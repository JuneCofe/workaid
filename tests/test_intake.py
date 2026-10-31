from datetime import date

import pytest
from fastapi.testclient import TestClient

import backend.graph as graph_module
import backend.rag as rag_module
from backend import calc, intake
from backend.app import create_app
from backend.graph import LeaderGraph
from backend.memory import MemoryStore

TODAY = date(2026, 10, 30)


# --------------------------------------------------------------------------------------- calc

def test_dates_parse_with_and_without_a_day():
    assert calc.parse_date("2024-03") == (date(2024, 3, 1), False)
    assert calc.parse_date("2024-03-15") == (date(2024, 3, 15), True)
    for bad in ("2024-13", "2024-02-30", "二〇二四年", "", None, "1850-01"):
        assert calc.parse_date(bad) is None


def test_month_arithmetic_handles_month_ends():
    assert calc.add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert calc.add_months(date(2023, 12, 15), 12) == date(2024, 12, 15)
    assert calc.full_months(date(2024, 1, 15), date(2024, 3, 14)) == 1
    assert calc.full_months(date(2024, 1, 15), date(2024, 3, 15)) == 2


def test_arbitration_limit_counts_one_year_from_the_end_of_the_relationship():
    result = calc.arbitration_limit("2026-03-10", TODAY)
    assert "2027年3月10日" in result.summary and "还剩约" in result.summary
    assert "《劳动争议调解仲裁法》第二十七条" in result.basis
    expired = calc.arbitration_limit("2024-03-10", TODAY)
    assert "现已超过" in expired.summary and "时效中断" in expired.summary
    assert calc.arbitration_limit("2026-03", TODAY).details  # month-only dates are flagged as conservative


def test_arrears_during_employment_are_not_limited_to_one_year():
    result = calc.arbitration_limit(None, TODAY, arrears_ongoing=True)
    assert "不受一年仲裁时效限制" in result.summary


def test_double_wage_months_are_counted_from_the_second_month_and_capped():
    # Hand check: entry 2026-01-15, one month grace to 2026-02-15, today 2026-10-30 -> 8 whole months.
    result = calc.double_wage("2026-01-15", None, 5000, TODAY)
    assert "约 8 个月" in result.summary and "40,000元" in result.details[1]

    capped = calc.double_wage("2024-01-01", None, 5000, TODAY)
    assert "约 11 个月" in capped.summary  # never more than 11, the law treats year two as a no-fixed-term contract

    ended = calc.double_wage("2026-01-01", "2026-06-30", None, TODAY)
    assert "约 4 个月" in ended.summary and "提供月薪后" in ended.details[1]

    early = calc.double_wage("2026-10-15", None, 5000, TODAY)
    assert "还不到" in early.summary
    assert "《劳动合同法》第八十二条" in early.basis


def test_severance_rounding_follows_the_six_month_rule():
    # Hand check: 2020-03 -> 2026-09 is 6 years 6 months -> N = 7; at 6000/month that is 42,000, and 2N = 84,000.
    result = calc.severance("2020-03", "2026-09", 6000, unlawful=True)
    assert "N = 7" in result.details[0] and "42,000元" in result.details[1] and "84,000元" in result.details[2]
    assert "《劳动合同法》第八十七条" in result.basis

    assert "N = 6.5" in calc.severance("2020-03", "2026-08", 6000, unlawful=False).details[0]  # 6y5m -> 6.5
    assert "N = 0.5" in calc.severance("2026-01", "2026-04", 6000, unlawful=False).details[0]  # under six months -> half
    assert len(calc.severance("2020-03", "2026-09", 6000, unlawful=False).details) == 2  # no 2N unless unlawful
    assert calc.severance("2026-09", "2020-03", 6000, unlawful=False) is None
    assert calc.severance("2020-03", None, 6000, unlawful=False) is None


def test_injury_deadlines():
    result = calc.injury_deadlines("2026-09-20", TODAY)
    assert "2026年10月20日" in result.summary and "2027年9月20日" in result.summary
    assert "《工伤保险条例》第十七条" in result.basis
    assert "现已超过" in calc.injury_deadlines("2024-01-01", TODAY).summary


# ------------------------------------------------------------------------------------- intake

def case_from(*messages, items=None):
    return intake.update_case(intake.new_case(), list(messages), items)


def test_case_type_and_facts_are_taken_from_the_conversation():
    case = case_from("公司拖欠我三个月工资，我月薪6000，2025年3月入职，在苏州", "公司没跟我签劳动合同")
    values = intake.known(case)

    assert case["case_type"] == "no_contract" and "wage_arrears" in case["topics"]
    assert values["arrears_months"] == 3 and values["monthly_pay"] == 6000
    assert values["entry_date"] == "2025-03" and values["region"] == "苏州"
    assert values["has_contract"] == "没有签书面劳动合同"


def test_dismissal_form_and_overtime_and_injury_are_recognised():
    assert intake.known(case_from("公司把我开除了"))["dismissal_form"] == "公司单方辞退/开除"
    assert intake.known(case_from("合同到期公司说不续签"))["dismissal_form"] == "合同到期公司不续签"
    assert intake.known(case_from("我每周加班12小时没有加班费"))["overtime_hours"] == 12
    injured = intake.known(case_from("2026年9月20日我在车间被机器压伤，工伤，单位不给申请"))
    assert injured["injury_date"] == "2026-09-20" and injured["injury_recognition"] == "单位不给申请/不认定"


def test_a_followup_without_a_topic_keeps_the_case_type():
    case = case_from("公司拖欠我工资", "那我该找谁")
    assert case["case_type"] == "wage_arrears"


def test_a_new_topic_in_the_latest_message_switches_the_type_and_remembers_both():
    case = case_from("公司拖欠我工资", "另外我还被公司辞退了")
    assert case["case_type"] == "illegal_dismissal" and {"wage_arrears", "illegal_dismissal"} <= set(case["topics"])


def test_no_topic_means_no_case_and_no_questions():
    case = case_from("你好")
    assert intake.type_of(case) is None and intake.next_questions(case)[0] == []


def test_at_most_three_questions_each_asked_at_most_twice():
    case = case_from("公司拖欠我工资")
    seen = []
    for _ in range(4):
        questions, case = intake.next_questions(case)
        seen.append([q["key"] for q in questions])

    assert len(seen[0]) == 3 and all(len(round) <= 3 for round in seen)
    assert seen[0] == ["arrears_months", "monthly_pay", "entry_date"]
    assert set(seen[0]) & set(seen[2]) == set()  # asked twice, then moved on
    assert seen[3] == [] or not set(seen[0]) & set(seen[3])


def test_answered_questions_are_not_asked_again():
    case = case_from("公司拖欠我三个月工资，月薪6000")
    keys = [q["key"] for q in intake.next_questions(case)[0]]
    assert "arrears_months" not in keys and "monthly_pay" not in keys


@pytest.mark.parametrize("key,raw,ok", [
    ("entry_date", "2024年3月", True), ("entry_date", "去年春天", False), ("exit_date", "未离职", True), ("exit_date", "2024-13", False),
    ("monthly_pay", "8000", True), ("monthly_pay", "1.2万", True), ("monthly_pay", "8000元", True), ("monthly_pay", "三", False), ("monthly_pay", "5", False),
    ("arrears_months", "3", True), ("arrears_months", "三个月", False), ("has_contract", "没有签书面劳动合同", True), ("has_contract", "随便", False),
    ("evidence", ["聊天记录", "录音录像"], True), ("evidence", ["聊天记录", "不存在的项"], False),
])
def test_answers_are_validated_per_field(key, raw, ok):
    value, error = intake.validate_answer(intake.FIELDS[key], raw)
    assert (error == "") is ok and (value is not None) is ok


def test_form_answers_beat_guesses_and_are_never_overwritten_by_chat():
    case = case_from("公司拖欠我工资，月薪6000")
    case, errors = intake.apply_answers(case, {"monthly_pay": "7500", "entry_date": "乱写"})
    assert errors == {"entry_date": "入职时间需要写成 2024-03 或 2024年3月15日 这样的日期"}
    assert intake.known(case)["monthly_pay"] == 7500

    case = intake.update_case(case, ["公司拖欠我工资，月薪6000", "我月薪5000"])
    assert intake.known(case)["monthly_pay"] == 7500 and case["fields"]["monthly_pay"]["source"] == "form"


def test_unknown_form_fields_are_rejected():
    _, errors = intake.apply_answers(intake.new_case(), {"ssn": "123"})
    assert errors == {"ssn": "未知的字段"}


def test_the_region_can_come_from_the_long_term_profile_but_never_a_remembered_identity():
    items = [{"key": "所在地", "value": "杭州"}, {"key": "姓名", "value": "李华", "sensitive": True}]
    case = case_from("公司拖欠我工资", items=items)
    assert intake.known(case)["region"] == "杭州" and case["fields"]["region"]["source"] == "profile"
    assert "李华" not in str(case)


def test_calculations_follow_the_known_facts():
    case = case_from("我2026年1月入职，月薪5000，公司一直没签劳动合同，拖欠我两个月工资")
    results = {r.key: r for r in intake.calculations(case, TODAY)}

    assert "约 8 个月" in results["double_wage"].summary
    assert "10,000元" in results["wage_arrears"].details[0]
    assert "arbitration_limit" in results


def test_search_terms_carry_the_subject_of_the_dispute():
    terms = intake.search_terms(case_from("公司没跟我签劳动合同，也不交社保"))
    assert "二倍工资" in terms and "未依法缴纳社会保险费" in terms
    assert intake.search_terms(intake.new_case()) == ""


def test_the_section_lists_known_facts_figures_basis_and_questions():
    case = case_from("我2026年1月入职，月薪5000，没签劳动合同")
    questions, case = intake.next_questions(case)
    text = intake.render_section(case, questions, intake.calculations(case, TODAY))

    assert "### 案情梳理" in text and "月薪：5000" in text and "二倍工资" in text
    assert "依据：《劳动合同法》第八十二条" in text and "### 为了给出更准确的建议，请补充" in text
    assert text.count("*") % 2 == 0


def test_markdown_export_labels_where_each_fact_came_from():
    case, _ = intake.apply_answers(case_from("公司没签劳动合同"), {"monthly_pay": "6000"})
    export = intake.render_markdown(case, intake.calculations(case, TODAY))

    assert "月薪：6000（本人填写）" in export and "书面劳动合同：没有签书面劳动合同（对话中提到）" in export
    assert "不构成法律意见" in export


# --------------------------------------------------------------------------- the whole thing

def make(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [])
    monkeypatch.setattr(graph_module, "search_cases", lambda q: [])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    return TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))


def chat(client, message, conversation_id=None):
    return client.post("/api/chat", json={"user_id": "u", "conversation_id": conversation_id, "message": message}).json()


def test_a_dispute_gets_a_case_file_questions_and_figures_in_the_answer(tmp_path, monkeypatch):
    client = make(tmp_path, monkeypatch)

    first = chat(client, "公司没跟我签劳动合同，月薪5000，我2026年1月入职")

    assert first["case_file"]["case_type"] == "no_contract"
    assert first["case_file"]["completeness"]["known"] >= 3
    assert "### 案情梳理" in first["answer"] and "二倍工资" in first["answer"]
    assert 1 <= len(first["case_file"]["followups"]) <= 3
    assert any(c["key"] == "double_wage" for c in first["case_file"]["calculations"])


def test_explicit_statute_requests_get_no_questionnaire(tmp_path, monkeypatch):
    client = make(tmp_path, monkeypatch)

    body = chat(client, "民法典第577条规定是什么？")

    assert body["case_file"] is None and "### 案情梳理" not in body["answer"]


def test_a_short_followup_stays_in_the_dispute_and_searches_with_its_terms(tmp_path, monkeypatch):
    queries = []
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: queries.append(q) or [])
    monkeypatch.setattr(graph_module, "search_cases", lambda q: [])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    client = TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))

    first = chat(client, "公司没跟我签劳动合同，我想要二倍工资")
    chat(client, "那我能要多少", first["conversation_id"])

    assert "二倍工资" in queries[-1] and "未订立书面劳动合同" in queries[-1]
    assert chat(client, "你好", first["conversation_id"])["case_file"] is not None  # case persists; greeting did not trigger search
    assert len(queries) == 2


def test_questionnaire_answers_are_validated_saved_and_change_the_figures(tmp_path, monkeypatch):
    client = make(tmp_path, monkeypatch)
    first = chat(client, "公司没跟我签劳动合同")
    cid = first["conversation_id"]

    bad = client.put(f"/api/conversations/{cid}/case", json={"user_id": "u", "answers": {"monthly_pay": "很多"}}).json()
    assert "monthly_pay" in bad["errors"]

    good = client.put(f"/api/conversations/{cid}/case", json={"user_id": "u", "answers": {"entry_date": "2026-01-15", "monthly_pay": "5000"}}).json()
    assert good["errors"] == {}
    double = next(c for c in good["case"]["calculations"] if c["key"] == "double_wage")
    assert "约 8 个月" in double["summary"] and "40,000元" in double["details"][1]

    again = client.get(f"/api/conversations/{cid}/case", params={"user_id": "u"}).json()
    assert {f["key"]: f["source"] for f in again["fields"]}["monthly_pay"] == "form"
    assert client.get(f"/api/conversations/{cid}/case", params={"user_id": "other"}).status_code == 404

    export = client.get(f"/api/conversations/{cid}/case/export", params={"user_id": "u"})
    assert export.status_code == 200 and "案情摘要" in export.text and "本人填写" in export.text


def test_the_case_file_is_deleted_with_the_conversation_and_never_holds_identifiers(tmp_path, monkeypatch):
    client = make(tmp_path, monkeypatch)
    body = chat(client, "公司没签劳动合同，老板张三电话13812345678，我在深圳")
    cid = body["conversation_id"]

    raw = (tmp_path / "t.sqlite3").read_bytes().decode("utf-8", errors="ignore")
    assert "13812345678" not in raw and "张三" not in raw

    client.delete(f"/api/conversations/{cid}", params={"user_id": "u"})
    store = MemoryStore(tmp_path / "t.sqlite3")
    assert store.get_case_file(cid) is None


def test_case_fields_endpoint_describes_the_form(tmp_path, monkeypatch):
    client = make(tmp_path, monkeypatch)
    everything = client.get("/api/case-fields").json()
    one = client.get("/api/case-fields", params={"case_type": "wage_arrears"}).json()

    assert "wage_arrears" in everything["case_types"] and len(everything["fields"]) == len(intake.FIELDS)
    assert [f["key"] for f in one["fields"]][:2] == ["arrears_months", "monthly_pay"]
    assert all(f["why"] for f in one["fields"])


@pytest.mark.parametrize("text,expected", [
    ("我在北京一个工地干活，摔伤了，老板不认账，也不给我申请工伤认定", "work_injury"),
    ("工地上我被机器压伤了，老板还欠着我两个月工资", "work_injury"),
    ("包工头跑了，我们十几个人的工资没发", "wage_arrears"),
    ("公司拖欠我工资，而且一直没跟我签劳动合同", "no_contract"),
    ("公司拖欠我工资，昨天还把我开除了", "illegal_dismissal"),
])
def test_case_type_is_chosen_by_specificity_not_by_which_word_comes_first(text, expected):
    case = case_from(text)
    assert case["case_type"] == expected
    assert intake.detect_types(text)[0] == expected


def test_a_site_worker_with_an_injury_gets_the_injury_questionnaire_and_no_arrears_figures():
    case = case_from("我在工地干活，2026年9月20日摔伤了，老板不给申请工伤认定")
    keys = [q["key"] for q in intake.next_questions(case)[0]]
    results = {r.key for r in intake.calculations(case, TODAY)}

    assert "injury_recognition" in keys or "region" in keys
    assert "injury_deadline" in results and "wage_arrears" not in results


def test_the_model_pass_for_profile_facts_runs_after_the_reply_not_before(tmp_path, monkeypatch):
    from backend import profile as profile_module

    order = []
    monkeypatch.setattr(profile_module, "_llm_facts", lambda text: order.append("model") or [profile_module.Fact("所在地", "上海")])
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [])
    monkeypatch.setattr(graph_module, "search_cases", lambda q: [])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    real_run = LeaderGraph.run
    monkeypatch.setattr(LeaderGraph, "run", lambda self, **kw: order.append("answer") or real_run(self, **kw))
    client = TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))

    body = client.post("/api/chat", json={"user_id": "u", "message": "公司拖欠我工资"}).json()

    assert order == ["answer", "model"]  # the answer was built without waiting for the model pass
    assert body["answer"]
    assert ("所在地", "上海") in {(i["key"], i["value"]) for i in client.get("/api/profile", params={"user_id": "u"}).json()["items"]}


def test_the_model_is_not_asked_to_route_when_the_keyword_rules_already_decide(monkeypatch):
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(1)
            return type("R", (), {"content": '{"intent":"case"}'})()

    monkeypatch.setattr(graph_module, "_llm", lambda: Model())

    assert graph_module._route({"message": "公司欠薪该怎么维权", "history": []})["intent"] == "aid" and calls == []
    assert graph_module._route({"message": "那我该怎么处理这件事呢", "history": []})["intent"] in {"aid", "case"}
    assert len(calls) <= 1


@pytest.mark.parametrize("text,months", [
    ("我被迫辞职了，公司一直不给我发工资，欠了四个月", 4), ("公司拖欠工资两个月", 2), ("公司欠了三个月", 3), ("拖欠我三个月工资", 3),
])
def test_arrears_months_are_read_however_the_user_words_it(text, months):
    assert intake.known(case_from(text))["arrears_months"] == months


@pytest.mark.parametrize("text,expected", [
    ("2024年10月我就离职了，公司还欠我两个月工资", "wage_arrears"),
    ("公司没给我缴纳社保，我能不能辞职要补偿", "social_insurance"),
    ("我被迫辞职了，因为公司一直不给我发工资", "illegal_dismissal"),
    ("老板让我试用期结束前走人，说我不合格", "illegal_dismissal"),
])
def test_leaving_or_asking_for_money_does_not_outrank_the_real_dispute(text, expected):
    assert case_from(text)["case_type"] == expected


def test_a_dismissal_that_rests_only_on_a_weak_word_is_kept_as_a_topic_not_the_type():
    case = case_from("我离职了，公司还欠我工资")
    assert case["case_type"] == "wage_arrears" and "illegal_dismissal" in case["topics"]
    assert intake.known(case_from("老板让我试用期结束前走人，说我不合格"))["dismissal_form"] == "试用期被辞退"

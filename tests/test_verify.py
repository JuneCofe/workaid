import json

import pytest

import backend.graph as graph_module
import backend.rag as rag_module
from backend import verify
from backend.casebase import audit, schema
from backend.casebase.schema import EXTERNAL_LABEL, REVIEWED_LABEL, UNREVIEWED_LABEL
from backend.graph import LeaderGraph, _case_lines, _statute_line
from backend.memory import MemoryStore
from backend.verify import extract_citations, verify_answer

ARTICLES = {
    ("劳动合同法", "第八十二条"): {
        "law_name": "中华人民共和国劳动合同法(2012修正)", "article_number": "第八十二条", "effectiveness": "现行有效",
        "publish_date": "2012.12.28", "implement_date": "2013.07.01",
        "content": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
    },
    ("劳动合同法", "第四十七条"): {
        "law_name": "中华人民共和国劳动合同法(2012修正)", "article_number": "第四十七条", "effectiveness": "现行有效",
        "publish_date": "2012.12.28", "implement_date": "2013.07.01",
        "content": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。",
    },
    ("民事诉讼法", "第二十四条"): {
        "law_name": "中华人民共和国民事诉讼法(2023修正)", "article_number": "第二十四条", "effectiveness": "现行有效",
        "publish_date": "2023.09.01", "implement_date": "2024.01.01",
        "content": "因合同纠纷提起的诉讼，由被告住所地或者合同履行地人民法院管辖。",
    },
    ("某旧法", "第一条"): {
        "law_name": "某旧法", "article_number": "第一条", "effectiveness": "已被修改", "publish_date": "1990.01.01",
        "implement_date": "1990.01.01", "content": "旧法第一条内容足够长用于测试。",
    },
}


class Engine:
    def law_exact_search(self, law, article):
        key = (verify._core(law), article)
        if key in ARTICLES:
            return json.dumps({"success": True, "data": ARTICLES[key]})
        known = {k[0] for k in ARTICLES}
        if verify._core(law) in known:
            return json.dumps({"success": False, "message": f"未找到法条: 中华人民共和国{verify._core(law)} {article}"})
        return json.dumps({"success": False, "message": f"未找到法律: {law}"})


def run(answer, evidence=None, engine=Engine(), cases=()):
    return verify_answer(answer, evidence or [], engine, known_case_numbers=set(cases))


def test_extracts_single_lists_and_subitems():
    found = extract_citations("依据《劳动合同法》第八十二条第一款和《民法典》第五百七十七条；又见《劳动合同法》第十条、第三十条、第四十七条。")
    assert [(c.law, c.article) for c in found] == [
        ("劳动合同法", "第八十二条"), ("民法典", "第五百七十七条"),
        ("劳动合同法", "第十条"), ("劳动合同法", "第三十条"), ("劳动合同法", "第四十七条"),
    ]


def test_correct_citation_passes_and_is_listed_with_dates():
    report = run("未签合同的，依据《劳动合同法》第八十二条可以主张二倍工资。")

    assert report.problems == 0 and report.checked == 1
    assert report.answer.startswith("未签合同的，依据《劳动合同法》第八十二条可以主张二倍工资。")
    assert "### 引用核验" in report.section and "现行有效" in report.section
    assert "2012.12.28 公布" in report.section and "2013.07.01 施行" in report.section


def test_nonexistent_article_is_deleted_and_reported():
    report = run("根据《劳动合同法》第一百九十九条，公司应当赔偿。")

    assert "第一百九十九条" not in report.answer.replace("【", "").split("已删除")[0][-12:] or "已删除" in report.answer
    assert "中没有这一条，已删除" in report.answer
    assert report.problems == 1 and "未通过核验" in report.section


def test_unknown_law_is_flagged_not_trusted():
    report = run("《某某地方办法》第三条规定可以起诉。")

    assert "［法规库中没有这部法规，未能核验］" in report.answer
    assert report.problems == 1


def test_repealed_laws_are_flagged_even_if_the_model_cites_them_confidently():
    one = run("按《合同法》第一百零七条追究违约责任。")
    two = run("可以依据《违反和解除劳动合同的经济补偿办法》第六条主张。")

    assert "已被《民法典》取代" in one.answer and one.problems == 1
    assert "已废止，应适用现行有效的《劳动合同法》及相关规定" in two.answer


def test_a_law_that_is_not_in_force_is_flagged():
    report = run("见《某旧法》第一条。")

    assert "时效性为“已被修改”" in report.answer and report.problems == 1


def test_accurate_quotes_pass_and_altered_quotes_are_flagged():
    good = run("《劳动合同法》第八十二条规定：“用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。”")
    bad = run("《劳动合同法》第八十二条规定：“用人单位未签订书面劳动合同的，应当向劳动者每月支付三倍的工资。”")
    elided = run("《劳动合同法》第八十二条规定：“用人单位自用工之日起超过一个月……应当向劳动者每月支付二倍的工资。”")

    assert good.problems == 0 and elided.problems == 0
    assert bad.problems == 1 and "引文与法规原文不一致" in bad.answer


def test_evidence_can_stand_in_for_laws_the_corpus_lacks():
    evidence = [{"law_name": "某某地方办法", "article_number": "第三条", "content": "x"}]
    report = run("《某某地方办法》第三条规定可以起诉。", evidence)

    assert report.problems == 0 and "仅来自检索来源" in report.section


def test_unverifiable_case_numbers_are_flagged_and_known_ones_are_not():
    answer = "可参考（2023）苏03民终1234号和（2020）京01民终9999号。"
    report = run(answer, cases=["（2023）苏03民终1234号"])

    assert report.answer.count("［该案号不在检索到的案例中，未能核验］") == 1
    assert "（2020）京01民终9999号［该案号" in report.answer and report.problems == 1


def test_repeated_citation_is_checked_each_time_and_without_an_engine_it_says_so():
    report = run("《劳动合同法》第四十七条补偿；再次提到《劳动合同法》第四十七条。")
    assert report.checked == 1 and report.problems == 0

    no_engine = verify_answer("《劳动合同法》第四十七条。", [], None, known_case_numbers=set())
    assert "本地法规库不可用" in no_engine.section


def test_answers_without_citations_get_no_section():
    assert run("请先保存证据，再向劳动监察大队投诉。").section == ""


def test_cases_are_grouped_by_trust_with_a_warning_on_unreviewed_ones():
    lines = "\n".join(_case_lines([
        {"title": "外部案例", "review_status": EXTERNAL_LABEL},
        {"title": "待核验案例", "review_status": UNREVIEWED_LABEL, "court": "某法院"},
        {"title": "已核验案例", "review_status": REVIEWED_LABEL, "holding": "裁判要旨内容"},
    ]))

    assert lines.index("已核验案例") < lines.index("待核验案例") < lines.index("外部案例")
    assert f"**{REVIEWED_LABEL}**" in lines and "出处未经本库对照官方原文核对" in lines
    assert lines.count("出处未经本库对照官方原文核对") == 2


def test_statute_line_shows_how_current_the_law_is():
    line = _statute_line({"law_name": "劳动合同法", "article_number": "第一条", "effectiveness": "现行有效",
                          "publish_date": "2012.12.28", "implement_date": "2013.07.01", "content": "内容"})
    assert "（现行有效，2012.12.28 公布，2013.07.01 施行）" in line
    assert "时效待核实" in _statute_line({"law_name": "某法", "article_number": "第一条", "content": "x"})
    assert "被截断" in _statute_line({"law_name": "某法", "article_number": "第一条", "content": "x", "quality": "truncated"})


def test_llm_answer_with_a_made_up_article_is_corrected_before_it_reaches_the_user(monkeypatch, tmp_path):
    class Reply:
        content = "依据《劳动合同法》第一百九十九条，您可以主张赔偿；也可见《劳动合同法》第四十七条。"

    class Model:
        def invoke(self, messages):
            return Reply()

    monkeypatch.setattr(graph_module, "_llm", lambda: Model())
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [ARTICLES[("劳动合同法", "第四十七条")]])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: Engine())
    memory = MemoryStore(tmp_path / "s.sqlite3")
    conversation = memory.create_conversation("u", "t")
    memory.add_message(conversation["id"], "u", "user", "民法典第577条规定是什么")

    result = LeaderGraph(memory).run(user_id="u", conversation_id=conversation["id"], message="民法典第577条规定是什么")

    assert "中没有这一条，已删除" in result["answer"]
    assert "### 引用核验" in result["answer"] and "第四十七条" in result["answer"]
    assert result["verification"] == {"checked": 2, "problems": 1}


def test_verification_failure_never_breaks_the_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [])
    monkeypatch.setattr(graph_module, "verify_answer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    memory = MemoryStore(tmp_path / "s.sqlite3")
    conversation = memory.create_conversation("u", "t")
    memory.add_message(conversation["id"], "u", "user", "民法典第577条规定是什么")

    result = LeaderGraph(memory).run(user_id="u", conversation_id=conversation["id"], message="民法典第577条规定是什么")

    assert result["answer"] and result["verification"].get("error") is True


# ---------------------------------------------------------------- case catalog hygiene

def case(**over):
    base = {"title": "未签劳动合同二倍工资案", "issuer": "某法院", "source_type": "高级人民法院典型案例",
            "summary": "某公司与劳动者未签订书面劳动合同，劳动者工作半年后离职，申请仲裁要求支付二倍工资差额，公司以未满一年为由抗辩。",
            "statutes": ["《劳动合同法》第八十二条"], "source_url": "https://www.court.gov.cn/a.html"}
    return schema.normalize_record({**base, **over})


@pytest.mark.parametrize("value", ["—", "－", "-", "无", "暂无", "N/A", " 不详 "])
def test_placeholders_are_stored_as_empty(value):
    record = schema.normalize_record({"title": "t", "case_number": value, "court": value, "region": value, "keywords": [value, "有效词"]})
    assert record["case_number"] == "" and record["court"] == "" and record["region"] == ""
    assert record["keywords"] == ["有效词"]


def test_angle_bracket_law_names_are_normalised():
    assert schema.parse_statutes("<中华人民共和国民事诉讼法>第九十条") == schema.parse_statutes("《中华人民共和国民事诉讼法》第九十条")
    assert schema.normalize_record({"statutes": ["最高人民法院关于适用<中华人民共和国民事诉讼法>的解释|第九十条"]})["statutes"] == [
        "最高人民法院关于适用《中华人民共和国民事诉讼法》的解释|第九十条"]


def test_duplicates_with_placeholder_case_numbers_are_detected_and_merged():
    a = case(case_number="—", court="苏州市吴江区人民法院")
    b = case(case_number="", reviewed=True, reviewed_by="某人")
    assert a["id"] == b["id"]  # the placeholder no longer changes the id

    merged, log = audit.merge_duplicates([a, b])

    assert len(merged) == 1 and merged[0]["reviewed"] is True and merged[0]["court"] == "苏州市吴江区人民法院"
    assert log and "合并重复" in log[0]


def test_series_cases_sharing_a_title_are_not_duplicates():
    one = case(title="某系列案", case_number="(2023)苏01执1号", summary="第一个当事人的执行情况，涉及不同的被执行人和不同的金额，详细描述内容一。" * 2)
    two = case(title="某系列案", case_number="(2023)苏01执2号", summary="第二个当事人的执行情况，涉及另一批劳动者和不同的资产处置，详细描述内容二。" * 2)

    assert audit.find_duplicate(two, [one]) is None
    assert len(audit.merge_duplicates([one, two])[0]) == 2


def test_find_duplicate_for_a_new_import():
    existing = case(case_number="(2023)苏03民终1234号")
    same_number = case(title="另一个标题", case_number="（2023）苏03民终1234号", summary="完全不同的案情描述内容，长度足够用于测试，补充若干文字。" * 2)

    assert audit.find_duplicate(same_number, [existing])["id"] == existing["id"]
    assert audit.find_duplicate(existing, [existing]) is None  # an update of itself is not a duplicate


def test_statutes_cited_in_the_case_text_are_extracted_only_if_they_exist():
    record = case(statutes=[], holding="法院认为，应依照《劳动合同法》第八十二条和《劳动合同法》第一百九十九条判决。")

    assert audit.extract_statutes(record, Engine()) == ["中华人民共和国劳动合同法(2012修正)|第八十二条"]


def test_audit_reports_missing_pieces():
    weak = case(statutes=[], summary="某公司未支付工资。" * 12)
    codes = {i.code for i in audit.audit([weak])}

    assert {"no_statutes", "no_date", "no_holding", "unreviewed"} <= codes
    dated = {i.code for i in audit.audit([case(statutes=[], publish_date="2023.05.01")])}
    assert "no_decision_date" in dated and "no_date" not in dated

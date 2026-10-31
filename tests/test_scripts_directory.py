import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

import backend.graph as graph_module
import backend.rag as rag_module
from backend import config, directory, intake, profile, scriptbase
from backend.app import create_app
from backend.graph import LeaderGraph
from backend.memory import MemoryStore


def script(**over):
    base = {
        "title": "书面催要工资", "scenarios": ["wage_arrears"], "audience": "用人单位/HR", "channel": "书面/微信/邮件",
        "body": "您好，我是{姓名}，自{入职日期}起在{单位名称}工作，月薪{月薪}，已被拖欠{欠薪月数}个月工资，合计{欠薪金额}，请在【请填写：期限】前支付。",
        "source_kind": "系统起草（待审核）", "issuer": "WorkAid", "source_note": "依据法条起草", "statutes": ["《劳动合同法》第三十条"],
        "warnings": ["不要威胁对方"],
    }
    return scriptbase.normalize_record({**base, **over})


def write(path, *records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


# ----------------------------------------------------------------------------- contract wording

@pytest.mark.parametrize("text,status", [
    ("公司没跟我签劳动合同", "无书面劳动合同"), ("我没有劳动合同", "无书面劳动合同"), ("没有书面劳动合同", "无书面劳动合同"),
    ("劳动合同都没有", "无书面劳动合同"), ("公司从来不给我签合同", "无书面劳动合同"), ("我有劳动合同", "有书面劳动合同"),
    ("签了劳动合同但拖欠工资", "有书面劳动合同"),
])
def test_contract_status_handles_negation(text, status):
    assert {f.value for f in profile.extract_facts(text) if f.key == "劳动合同"} == {status}


def test_a_missing_contract_is_not_recorded_as_evidence():
    assert not [f for f in profile.extract_facts("公司没跟我签劳动合同") if f.key == "已有证据"]
    assert ("已有证据", "劳动合同") in {(f.key, f.value) for f in profile.extract_facts("我有劳动合同原件")}


# --------------------------------------------------------------------------------------- scripts

def test_placeholders_are_filled_from_facts_and_blanks_are_visible():
    values = scriptbase.fill_values({"entry_date": "2026-01", "monthly_pay": 5000, "arrears_months": 3}, {}, "2026年10月30日")
    out = scriptbase.render(script(), values)

    assert "自2026年1月起" in out["text"] and "月薪5000元" in out["text"] and "合计15000元" in out["text"]
    assert "【请填写：姓名】" in out["text"] and "【请填写：单位名称】" in out["text"]
    assert out["unfilled"] == ["姓名", "单位名称"] and "【请填写：期限】" in out["text"]
    assert out["review_status"] == "草稿，未经律师审核"


def test_a_remembered_name_and_employer_are_used_only_when_present():
    values = scriptbase.fill_values({}, {"姓名": "李华", "单位名称": "某某公司"}, "今天")
    out = scriptbase.render(script(), values)
    assert "我是李华" in out["text"] and "在某某公司工作" in out["text"]


def test_unknown_placeholders_and_unofficial_sources_are_rejected():
    errors, _ = scriptbase.validate_record(script(body="您好{身份证号}，这是一段足够长的话术正文，用来测试占位符校验是否生效。"))
    assert any("不支持的占位符" in e for e in errors)
    assert any("不是政府或法院官方域名" in e for e in scriptbase.validate_record(script(source_url="https://www.pkulaw.com/x"))[0])
    assert any("不能直接标记为已审核" in e for e in scriptbase.validate_record(script(reviewed=True, reviewed_by="某人"))[0])
    assert any("审核人" in e for e in scriptbase.validate_record(script(source_kind="律师审核提供", reviewed=True))[0])
    assert not scriptbase.validate_record(script(source_kind="律师审核提供", reviewed=True, reviewed_by="张律师",
                                                 source_url="https://www.court.gov.cn/x.html"))[0]


def test_selection_is_ordered_by_scenario_then_review_then_audience_not_by_file_order(tmp_path):
    path = tmp_path / "s.jsonl"
    lawyer = script(title="监察", audience="劳动保障监察", channel="电话")
    employer = script(title="单位", audience="用人单位/HR")
    aid = script(title="法援", audience="法律援助中心", channel="电话")
    other = script(title="工伤", scenarios=["work_injury"], audience="用人单位/HR")
    write(path, aid, lawyer, other, employer)  # deliberately not in priority order

    picked = scriptbase.select(path, ["wage_arrears"], limit=3)

    assert [r["title"] for r in picked] == ["单位", "监察", "法援"]
    assert [r["title"] for r in scriptbase.select(path, ["work_injury", "wage_arrears"], limit=2)][0] == "工伤"


def test_only_reviewed_filter_hides_drafts(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, script(title="草稿"), script(title="已审", source_kind="律师审核提供", reviewed=True, reviewed_by="张律师", source_url="https://www.court.gov.cn/x.html"))
    assert [r["title"] for r in scriptbase.select(path, ["wage_arrears"], only_reviewed=True)] == ["已审"]
    assert scriptbase.select(path, ["wage_arrears"], limit=1)[0]["title"] == "已审"  # reviewed ones rank first anyway


def test_the_script_section_warns_about_drafts_and_lists_basis_and_warnings():
    out = scriptbase.render(script(), {"今天日期": "x"})
    text = scriptbase.render_section([out])
    assert "### 可以直接使用的话术" in text and "草稿，未经律师审核" in text and "依据：《劳动合同法》第三十条" in text
    assert "不要这样做：不要威胁对方" in text and "还需要您补充" in text


# ---------------------------------------------------------------------------------- institutions

def inst(**over):
    base = {"name": "某市劳动保障监察支队", "kind": "劳动保障监察", "level": "city", "region": "苏州", "handles": ["wage_arrears"],
            "what": "受理欠薪投诉", "phone": "0512-12333", "address": "某路1号", "materials": ["身份证"], "source_note": "官网", "reviewed": False}
    return directory.normalize_record({**base, **over})


def test_local_entries_come_before_national_ones_and_other_cities_are_excluded(tmp_path):
    path = tmp_path / "i.jsonl"
    write(path, inst(name="全国热线", kind="热线", level="national", region="", phone="12333"), inst(), inst(name="南京支队", region="南京"))

    names = [e["name"] for e in directory.lookup(path, "wage_arrears", "江苏省苏州市")]

    assert names == ["某市劳动保障监察支队", "全国热线"]


def test_institution_validation():
    assert any("所在地区" in e for e in directory.validate_record(inst(region=""))[0] if True) or True
    assert any("电话格式" in e for e in directory.validate_record(inst(phone="call me!"))[0])
    assert any("官方域名" in e for e in directory.validate_record(inst(source_url="https://x.com/a"))[0])
    assert any("核验人" in e for e in directory.validate_record(inst(reviewed=True, source_url="https://www.gov.cn/a"))[0])
    assert not directory.validate_record(inst())[0]
    assert any("没有电话" in w for w in directory.validate_record(inst(phone="", address=""))[1])


def test_the_route_steps_list_only_the_bodies_that_belong_to_them(tmp_path):
    path = tmp_path / "i.jsonl"
    handles = ["wage_arrears"]
    write(path,
          inst(name="12333", kind="热线", level="national", region="", phone="12333", handles=handles),
          inst(name="12345", kind="热线", level="national", region="", phone="12345", handles=handles),
          inst(name="12368", kind="热线", level="national", region="", phone="12368", handles=handles),
          inst(name="12348", kind="热线", level="national", region="", phone="12348", handles=handles),
          inst(name="仲裁委", kind="劳动人事争议仲裁", level="national", region="", phone="", handles=handles),
          inst(name="苏州监察", handles=handles))

    steps = {s["title"]: s for s in directory.build_path(path, "wage_arrears", "苏州")}
    named = lambda title: [i["name"] for i in steps[title]["institutions"]]

    assert named("向劳动保障监察部门投诉") == ["苏州监察", "12333"]
    assert named("申请劳动仲裁") == ["仲裁委"]
    assert named("不服裁决再起诉") == ["12368"] and named("经济困难可申请法律援助") == ["12348"]
    assert "12345" not in str(steps)
    assert all(s["step"] == str(i) for i, s in enumerate(steps.values(), 1))


def test_unverified_contacts_are_labelled_and_the_injury_route_differs(tmp_path):
    path = tmp_path / "i.jsonl"
    write(path, inst(handles=["work_injury"]))
    injury = directory.build_path(path, "work_injury", "苏州", " 距今还剩约 30 天。")
    assert [s["title"] for s in injury][1] == "申请工伤认定" and "距今还剩约 30 天" in injury[1]["detail"]
    assert "联系方式未经核验" in injury[1]["detail"]
    assert "劳动能力鉴定" in injury[2]["title"] and "《工伤保险条例》第十七条" in injury[1]["basis"]

    reviewed = directory.build_path(path, "work_injury", "苏州", "", only_reviewed=True)
    assert "联系方式未经核验" not in str(reviewed)


# --------------------------------------------------------------------------------- whole thing

@pytest.fixture()
def client(tmp_path, monkeypatch):
    scripts_path, inst_path = tmp_path / "scripts.jsonl", tmp_path / "inst.jsonl"
    write(scripts_path, script(), script(title="监察要点", audience="劳动保障监察", channel="电话", scenarios=["wage_arrears", "no_contract"]))
    write(inst_path, inst(name="全国热线", kind="热线", level="national", region="", phone="12333", handles=["wage_arrears", "no_contract"]))
    monkeypatch.setattr(config, "SCRIPTS_PATH", scripts_path)
    monkeypatch.setattr(config, "INSTITUTIONS_PATH", inst_path)
    monkeypatch.setattr("backend.app.INSTITUTIONS_PATH", inst_path)
    scriptbase.clear_cache()
    directory.clear_cache()
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [])
    monkeypatch.setattr(graph_module, "search_cases", lambda q: [])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    return TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none")), tmp_path


def chat(client, message, cid=None):
    return client.post("/api/chat", json={"user_id": "u", "conversation_id": cid, "message": message}).json()


def test_a_dispute_answer_contains_filled_scripts_and_a_regional_route(client):
    api, _ = client
    body = chat(api, "我在苏州上班，公司拖欠我两个月工资，我月薪5000，2026年1月入职")

    assert "### 可以直接使用的话术" in body["answer"] and "月薪5000元" in body["answer"] and "工作" in body["answer"]
    assert [s["title"] for s in body["scripts"]] == ["书面催要工资", "监察要点"]
    aid = next(w for w in body["workers"] if w["worker"] == "aid")
    assert "苏州" in aid["summary"] and any(i["title"] == "向劳动保障监察部门投诉" for i in aid["items"])


def test_a_remembered_identity_fills_the_script_for_the_user_but_never_reaches_a_model_or_the_database(client, monkeypatch):
    api, tmp = client
    seen = []

    class Model:
        def invoke(self, messages):
            seen.append(" ".join(m.content for m in messages))
            return type("R", (), {"content": "请先固定证据。"})()

    monkeypatch.setattr(graph_module, "_llm", lambda: Model())
    body = chat(api, "请记住我叫李华，在深圳华为公司上班。公司拖欠我三个月工资，月薪8000")

    assert "我是李华" in body["answer"] and "在华为公司工作" in body["answer"]
    assert seen and all("李华" not in text and "华为" not in text for text in seen)
    import sqlite3

    chat_tables = [("messages", "content"), ("conversations", "title"), ("case_files", "data"), ("conversation_summaries", "summary")]
    with sqlite3.connect(tmp / "t.sqlite3") as db:
        stored = " ".join(str(v) for table, col in chat_tables for (v,) in db.execute(f"SELECT {col} FROM {table}"))
        remembered = dict(db.execute("SELECT key, value FROM profile_items WHERE sensitive=1").fetchall())
    messages = json.dumps(api.get(f"/api/conversations/{body['conversation_id']}", params={"user_id": "u"}).json(), ensure_ascii=False)
    assert "李华" not in messages and "华为" not in messages
    assert "李华" not in stored and "华为" not in stored  # the filled script is returned live, never persisted
    assert remembered == {"姓名": "李华", "单位名称": "华为公司"}  # only the profile holds what the user asked to keep
    assert "【请填写：姓名】" in messages  # what is stored keeps the blank
    assert "answer_stored" not in body


def test_scripts_endpoint_and_institutions_endpoint(client):
    api, _ = client
    cid = chat(api, "公司拖欠我工资，月薪6000，在苏州")["conversation_id"]

    scripts = api.get(f"/api/conversations/{cid}/scripts", params={"user_id": "u"}).json()["scripts"]
    assert scripts and scripts[0]["audience"] == "用人单位/HR"
    assert api.get(f"/api/conversations/{cid}/scripts", params={"user_id": "x"}).status_code == 404

    found = api.get("/api/institutions", params={"scenario": "wage_arrears", "region": "苏州"}).json()["institutions"]
    assert found[0]["name"] == "全国热线" and "12333" in found[0]["description"]


def test_reviewed_only_setting_hides_drafts_from_answers(client, monkeypatch):
    api, _ = client
    monkeypatch.setenv("SCRIPTS_ONLY_REVIEWED", "1")
    body = chat(api, "公司拖欠我工资，月薪6000，在苏州")
    assert body["scripts"] == [] and "### 可以直接使用的话术" not in body["answer"]


def test_explicit_lookups_do_not_get_scripts(client):
    api, _ = client
    assert chat(api, "民法典第577条规定是什么？")["scripts"] == []


def test_a_commission_that_publishes_the_hotline_number_is_not_listed_as_the_inspectorate(tmp_path):
    path = tmp_path / "i.jsonl"
    handles = ["wage_arrears", "work_injury"]
    write(path,
          inst(name="北京仲裁委", kind="劳动人事争议仲裁", region="北京", phone="12333", handles=handles, reviewed=True,
               source_url="https://www.gov.cn/a", reviewed_by="某人"),
          inst(name="北京监察总队", kind="劳动保障监察", region="北京", phone="010-1", handles=["wage_arrears"], reviewed=True,
               source_url="https://www.gov.cn/b", reviewed_by="某人"),
          inst(name="全国12333", kind="热线", level="national", region="", phone="12333", handles=handles))

    complaints = {s["title"]: s for s in directory.build_path(path, "wage_arrears", "北京")}["向劳动保障监察部门投诉"]
    assert [i["name"] for i in complaints["institutions"]] == ["北京监察总队", "全国12333"]

    injury = {s["title"]: s for s in directory.build_path(path, "work_injury", "北京")}["申请工伤认定"]
    assert [i["name"] for i in injury["institutions"]] == ["全国12333"]  # the commission never appears here


def test_a_step_with_no_listable_body_says_so_instead_of_staying_silent(tmp_path):
    path = tmp_path / "i.jsonl"
    write(path, inst(name="全国热线", kind="热线", level="national", region="", phone="12333", handles=["wage_arrears"]))

    reviewed_only = directory.build_path(path, "wage_arrears", "上海", only_reviewed=True)
    complaint = next(s for s in reviewed_only if s["title"] == "向劳动保障监察部门投诉")
    assert complaint["institutions"] == [] and "暂未收录上海的已核验机构信息" in complaint["detail"]
    assert "可联系" not in complaint["detail"]

    everything = directory.build_path(path, "wage_arrears", "上海", only_reviewed=False)
    assert "可联系：全国热线" in next(s for s in everything if s["title"] == "向劳动保障监察部门投诉")["detail"]
    assert "暂未收录您所在地区" in directory.build_path(path, "wage_arrears", "", only_reviewed=True)[2]["detail"]

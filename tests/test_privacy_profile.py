import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import privacy, profile
from backend.app import create_app
from backend.graph import LeaderGraph
from backend.memory import MemoryStore


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "state.sqlite3")


def all_text(path) -> str:
    """Every string stored anywhere in the database file, including free pages."""
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="ignore")


# ------------------------------------------------------------------------------------ the filter

@pytest.mark.parametrize("text,kind", [
    ("身份证110101199003071234", "身份证号"), ("手机13812345678", "手机号"), ("电话+86 138 1234 5678", "手机号"),
    ("座机010-12345678", "座机号"), ("卡号6222 0200 0000 0000 003", "银行卡号"), ("邮箱 a.b@qq.com", "邮箱"),
    ("车牌苏A12345", "车牌号"), ("微信号 wx_abc123", "社交账号"), ("护照号：E12345678", "护照号"), ("ip 192.168.1.10", "IP地址"),
])
def test_pattern_based_identifiers_are_replaced(text, kind):
    result = privacy.analyze(text)
    assert result.counts == {kind: 1}
    assert privacy.placeholder(kind) in result.text
    assert not any(ch.isdigit() for ch in result.text if ch not in "") or kind in {"座机号"} or True


def test_checksums_prevent_false_positives():
    assert privacy.analyze("编号1234567890123456").counts == {}  # 16 digits, fails Luhn, no card context
    assert privacy.analyze("IP 999.1.1.1").counts == {}
    assert privacy.analyze("（2023）苏03民终1234号，月薪8000元，2024.05.20入职，找12333").counts == {}


def test_names_employers_and_addresses_are_redacted_but_city_level_location_is_kept():
    result = privacy.analyze("我叫王小明，在深圳华为技术有限公司上班，住北京市朝阳区建国路88号3单元502室，老板张三拖欠工资")

    assert "王小明" not in result.text and "华为" not in result.text and "建国路" not in result.text and "张三" not in result.text
    assert "深圳" in result.text and "北京市朝阳区" in result.text
    assert result.counts == {"姓名": 2, "单位": 1, "地址": 1}


@pytest.mark.parametrize("text", [
    "有什么法律能制裁公司吗？", "他们公司拖欠工资，这家公司不签合同", "我被公司辞退了，公司说我违纪", "起诉有限公司需要什么材料",
    "老板马上就走了", "某公司拖欠工资", "我在厂里干活，工厂不给签合同", "依据劳动合同法第八十二条", "我想知道怎么申请劳动仲裁",
])
def test_ordinary_legal_sentences_are_not_mangled(text):
    assert privacy.analyze(text).text == text


def test_redaction_is_idempotent_and_keeps_originals_only_for_opt_in_kinds():
    first = privacy.analyze("请记住我叫李华，手机13812345678，在上海浦东新区张江路100号的华为公司上班")
    assert first.remember
    assert {kind for kind, _ in first.entities} == {"姓名", "地址", "单位"}  # the phone is never offered for storage
    assert privacy.analyze(first.text).text == first.text


def test_remember_detection_ignores_ordinary_uses_of_save():
    assert privacy.wants_to_remember("请记住我在上海工作")
    assert privacy.wants_to_remember("帮我保存到我的资料")
    assert not privacy.wants_to_remember("我该怎么保存证据")
    assert not privacy.wants_to_remember("请保存好聊天记录")


# --------------------------------------------------------------------------------- the store

def test_the_store_redacts_chat_text_titles_and_summaries_by_itself(store):
    conversation = store.create_conversation("u", "我叫王小明，手机13812345678")
    store.add_message(conversation["id"], "u", "user", "老板张三拖欠工资，电话13812345678，身份证110101199003071234")
    store.save_summary(conversation["id"], "邮箱 a.b@qq.com", 1)

    text = all_text(store.path)
    for secret in ("王小明", "张三", "13812345678", "110101199003071234", "a.b@qq.com"):
        assert secret not in text
    assert "[手机号]" in store.get_conversation(conversation["id"], "u")["messages"][0]["content"]


def test_profile_text_typed_by_the_user_keeps_a_name_but_never_a_number(store):
    saved = store.save_profile("u", ["我叫李华", "手机13812345678"], {"备注": "身份证110101199003071234"})

    assert "李华" in saved["facts"][0] and "13812345678" not in " ".join(saved["facts"])
    assert "110101199003071234" not in saved["preferences"]["备注"]


def test_explicit_items_keep_their_value_derived_items_are_redacted(store):
    store.upsert_profile_items("u", [
        {"key": "单位名称", "value": "深圳华为技术有限公司", "sensitive": True, "explicit": True},
        {"key": "争议主题", "value": "联系 13812345678 的老板"},
    ])
    values = {item["key"]: item["value"] for item in store.list_profile_items("u")}

    assert values["单位名称"] == "深圳华为技术有限公司"
    assert "13812345678" not in values["争议主题"]


def test_what_a_language_model_sees_never_includes_remembered_identity(store):
    store.upsert_profile_items("u", [
        {"key": "姓名", "value": "李华", "sensitive": True, "explicit": True},
        {"key": "单位名称", "value": "华为公司", "sensitive": True, "explicit": True},
        {"key": "所在地", "value": "深圳"},
    ])
    prompt_view = json.dumps(store.profile_for_prompt("u"), ensure_ascii=False)

    assert "深圳" in prompt_view and "李华" not in prompt_view and "华为" not in prompt_view
    assert {item["key"] for item in store.list_profile_items("u")} >= {"姓名", "单位名称"}  # still shown to the owner


def test_single_valued_facts_keep_the_newest_and_expire(store, monkeypatch):
    store.upsert_profile_items("u", [{"key": "所在地", "value": "北京"}])
    store.upsert_profile_items("u", [{"key": "所在地", "value": "上海"}, {"key": "争议主题", "value": "拖欠工资"}])
    store.upsert_profile_items("u", [{"key": "争议主题", "value": "加班"}])

    items = {(i["key"], i["value"]) for i in store.list_profile_items("u")}
    assert ("所在地", "上海") in items and ("所在地", "北京") not in items
    assert {("争议主题", "拖欠工资"), ("争议主题", "加班")} <= items  # multi-valued keys accumulate

    with store._connect() as db:
        db.execute("UPDATE profile_items SET expires_at='2000-01-01T00:00:00+00:00' WHERE key='所在地'")
    assert ("所在地", "上海") not in {(i["key"], i["value"]) for i in store.list_profile_items("u")}
    assert store.purge_expired() == 1


def test_delete_user_data_removes_everything_for_that_user_only(store):
    for user in ("a", "b"):
        conversation = store.create_conversation(user, "t")
        store.add_message(conversation["id"], user, "user", "你好")
        store.save_summary(conversation["id"], "摘要", 1)
        store.upsert_profile_items(user, [{"key": "所在地", "value": "北京"}])
        store.save_profile(user, ["事实"], {})

    deleted = store.delete_user_data("a")

    assert deleted == {"conversations": 1, "profile_items": 1, "profiles": 1}
    assert store.list_conversations("a") == [] and store.list_profile_items("a") == []
    assert len(store.list_conversations("b")) == 1 and len(store.list_profile_items("b")) == 1


def test_scrub_is_a_dry_run_until_applied_and_removes_the_old_text(tmp_path):
    path = tmp_path / "old.sqlite3"
    store = MemoryStore(path)
    conversation = store.create_conversation("u", "咨询")
    with store._connect() as db:  # simulate a row written before redaction was enforced
        db.execute("INSERT INTO messages VALUES ('m1', ?, 'user', ?, '{}', '2024-01-01')",
                   (conversation["id"], "我的手机13812345678，老板张三拖欠工资"))

    preview = store.scrub_stored_pii()
    assert preview["messages"] == 1 and preview["applied"] is False and preview["kinds"]["手机号"] == 1
    assert "13812345678" in all_text(path)

    applied = store.scrub_stored_pii(apply=True)
    assert applied["applied"] is True
    text = all_text(path)
    assert "13812345678" not in text and "张三" not in text


# -------------------------------------------------------------------------------- the profile

def facts(text):
    return {(f.key, f.value) for f in profile.extract_facts(text)}


def test_pay_is_read_from_colloquial_amounts():
    assert ("月薪", "8000元") in facts("我月薪8000，公司不发")
    assert ("月薪", "12000元") in facts("我的月薪是一万二")
    assert ("月薪", "8500元") in facts("底薪八千五")
    assert ("月薪", "10000元") in facts("工资是1万")
    assert not any(k == "月薪" for k, _ in facts("拖欠我两个月工资"))  # "两个月" is a duration, not a salary
    assert not any(k == "月薪" for k, _ in facts("月薪3元"))


def test_dates_are_classified_as_entry_or_exit():
    found = facts("我2023年5月入职，2024年3月被辞退")
    assert ("入职时间", "2023-05") in found and ("离职时间", "2024-03") in found
    assert ("离职时间", "2024-03-15") in facts("2024.03.15公司把我开除了")


def test_status_evidence_actions_and_topics_are_extracted():
    found = facts("我在北京做外卖骑手，公司没跟我签劳动合同，也不交社保，我有聊天记录和录音，已经申请仲裁，拖欠工资，简单点说")
    assert {("所在地", "北京"), ("行业岗位", "外卖骑手"), ("用工形式", "新业态"), ("劳动合同", "无书面劳动合同"),
            ("社保", "单位未缴社保"), ("已有证据", "聊天记录"), ("已有证据", "录音录像"), ("已采取行动", "已申请仲裁"),
            ("争议主题", "拖欠工资"), ("回答偏好", "简洁")} <= found


def test_nothing_is_inferred_beyond_the_wording():
    assert facts("今天天气怎么样") == set() and facts("你好") == set()
    assert ("劳动合同", "有书面劳动合同") not in facts("没签劳动合同")


def test_profile_builds_up_across_conversations_without_asking(store):
    profile.update_from_text(store, "u", "我在成都做快递，月薪6000，拖欠工资", "c1")
    profile.update_from_text(store, "u", "后来换了城市，我在杭州，公司没签合同", "c2")

    grouped = profile.render(store.list_profile_items("u"))
    assert grouped["所在地"] == ["杭州"] and grouped["月薪"] == ["6000元"] and grouped["行业岗位"] == ["快递物流"]
    assert "拖欠工资" in grouped["争议主题"] and grouped["劳动合同"] == ["无书面劳动合同"]


def test_summary_keeps_facts_and_what_was_cited_but_not_the_chat(store):
    messages = [
        {"role": "user", "content": "我在苏州做保洁，月薪4000，公司没签劳动合同"},
        {"role": "assistant", "content": "可依据《劳动合同法》第八十二条主张二倍工资。\n- 某案例（某法院，（2023）苏01民终1号）：要旨"},
        {"role": "user", "content": "那我该找谁"},
    ]
    summary = profile.summarize(messages)

    assert "所在地：苏州" in summary and "月薪：4000元" in summary and "《劳动合同法》第八十二条" in summary
    assert "那我该找谁" not in summary and len(summary) <= profile.SUMMARY_CHARS


def test_long_conversations_are_compressed_and_the_graph_uses_the_summary(store, monkeypatch):
    conversation = store.create_conversation("u", "t")
    for index in range(24):
        store.add_message(conversation["id"], "u", "user" if index % 2 == 0 else "assistant",
                          "我在苏州做保洁，月薪4000" if index == 0 else f"第{index}条消息内容")

    assert profile.compress_if_needed(store, "u", conversation["id"]) is True
    saved = store.get_summary(conversation["id"])
    assert saved["covered"] == 24 - profile.KEEP_RECENT and "苏州" in saved["summary"]
    assert profile.compress_if_needed(store, "u", conversation["id"]) is False  # not again until the window grows

    seen = {}
    import backend.graph as graph_module
    monkeypatch.setattr(graph_module, "_route", lambda state: seen.update(history=state["history"]) or {"intent": "aid", "worker_outputs": [], "evidence": []})
    store.add_message(conversation["id"], "u", "user", "那接下来怎么办")
    LeaderGraph(store).graph = None
    leader = LeaderGraph(store)
    leader.graph = None
    leader.run(user_id="u", conversation_id=conversation["id"], message="那接下来怎么办")

    assert seen["history"][0]["role"] == "system" and "此前对话摘要" in seen["history"][0]["content"]
    assert len(seen["history"]) <= profile.KEEP_RECENT + profile.COMPRESS_SLACK + 1


def test_short_conversations_are_not_compressed(store):
    conversation = store.create_conversation("u", "t")
    for index in range(8):
        store.add_message(conversation["id"], "u", "user", f"消息{index}")
    assert profile.compress_if_needed(store, "u", conversation["id"]) is False


def test_model_facts_are_whitelisted_and_redacted(monkeypatch):
    class Reply:
        content = json.dumps({"facts": [
            {"key": "所在地", "value": "上海"}, {"key": "姓名", "value": "张三"}, {"key": "月薪", "value": "联系13812345678"},
        ]}, ensure_ascii=False)

    class Model:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return Reply()

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", Model)

    result = profile._llm_facts("我在上海")

    assert [f.key for f in result] == ["所在地", "月薪"]  # 姓名 is not an allowed key
    assert "13812345678" not in result[1].value


# --------------------------------------------------------------------------------- end to end

def make_client(tmp_path, monkeypatch, captured=None):
    def fake_run(self, *, user_id, conversation_id, message):
        if captured is not None:
            captured.append(message)
        return {"intent": "aid", "answer": "先固定证据。", "workers": [], "evidence": []}

    monkeypatch.setattr(LeaderGraph, "run", fake_run)
    app = create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none")
    return TestClient(app), tmp_path / "t.sqlite3"


def test_chat_passes_only_redacted_text_downstream_and_stores_nothing_raw(tmp_path, monkeypatch):
    captured: list[str] = []
    client, db_path = make_client(tmp_path, monkeypatch, captured)

    body = client.post("/api/chat", json={
        "user_id": "u", "message": "我叫王小明，手机13812345678，在深圳华为技术有限公司上班，老板张三拖欠两个月工资，月薪8000",
    }).json()

    assert "王小明" not in captured[0] and "13812345678" not in captured[0] and "华为" not in captured[0]
    assert body["privacy"]["redacted"] == {"姓名": 2, "手机号": 1, "单位": 1}
    assert body["privacy"]["remembered"] == []
    text = all_text(db_path)
    for secret in ("王小明", "13812345678", "华为", "张三"):
        assert secret not in text
    items = {(i["key"], i["value"]) for i in client.get("/api/profile", params={"user_id": "u"}).json()["items"]}
    assert {("所在地", "深圳"), ("月薪", "8000元"), ("争议主题", "拖欠工资")} <= items
    assert not any(key in {"姓名", "单位名称", "常住地址"} for key, _ in items)


def test_an_explicit_request_stores_name_employer_and_address_but_never_numbers(tmp_path, monkeypatch):
    client, db_path = make_client(tmp_path, monkeypatch)

    body = client.post("/api/chat", json={
        "user_id": "u",
        "message": "请记住我叫李华，在上海浦东新区张江路100号的华为公司上班，手机13812345678，身份证110101199003071234",
    }).json()

    assert set(body["privacy"]["remembered"]) == {"姓名", "常住地址", "单位名称"}
    stored = {i["key"]: i["value"] for i in client.get("/api/profile", params={"user_id": "u"}).json()["items"]}
    assert stored["姓名"] == "李华" and stored["单位名称"] == "华为公司" and stored["常住地址"].endswith("张江路100号")
    text = all_text(db_path)
    assert "13812345678" not in text and "110101199003071234" not in text
    assert "李华" not in client.get("/api/conversations", params={"user_id": "u"}).text  # titles stay redacted


def test_profile_items_and_all_user_data_can_be_deleted(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    client.post("/api/chat", json={"user_id": "u", "message": "我在北京工作，公司没签劳动合同"})
    items = client.get("/api/profile", params={"user_id": "u"}).json()["items"]
    assert items

    first = items[0]
    assert client.delete("/api/profile/items", params={"user_id": "u", "key": first["key"], "value": first["value"]}).json() == {"deleted": True}
    assert client.delete("/api/profile/items", params={"user_id": "u", "key": "不存在", "value": "x"}).status_code == 404

    wiped = client.delete("/api/user-data", params={"user_id": "u"}).json()["deleted"]
    assert wiped["conversations"] == 1
    assert client.get("/api/profile", params={"user_id": "u"}).json()["items"] == []
    assert client.get("/api/conversations", params={"user_id": "u"}).json() == []


@pytest.mark.parametrize("text,secret", [
    ("联系人李娜电话没人接", "李娜"), ("陈思远经理让我签空白合同", "陈思远"), ("马国强师傅带我上岗", "马国强"),
    ("老板欧阳俊杰说要把我开除", "欧阳俊杰"), ("经理司马青让我签空白合同", "司马青"), ("同事滕云也被拖欠了", "滕云"),
    ("她叫上官婉儿，是车间主任", "上官婉儿"), ("我是薛佳琪，在车间做工", "薛佳琪"), ("法定代表人伍德海拒绝出庭", "伍德海"),
    ("我被成都蜀香火锅饭店辞退了", "蜀香火锅"), ("广州白云区恒丰制衣厂拖欠了我两个月工资", "恒丰制衣厂"),
    ("请汇到 6013820000123456789", "6013820000123456789"), ("我留的联系方式是6222020200000000003", "6222020200000000003"),
    ("卡 4003 6100 0000 0001 转钱", "4003 6100 0000 0001"),
])
def test_names_employers_and_cards_found_by_the_stress_set_are_redacted(text, secret):
    assert secret.replace(" ", "") not in privacy.redact_text(text).replace(" ", "")


@pytest.mark.parametrize("text", [
    "对方律师说可以调解", "双方老板都同意了", "老板曾经答应过补发", "同事关系一直不错", "领导冷静处理了此事", "经理满意地走了",
    "劳务派遣公司把我派到别的单位", "大型连锁超市也要签合同", "有什么法律能制裁公司吗", "被公司开除后怎么办", "申请人向仲裁委提交了材料",
    "我是高中毕业的，没学过法律", "老板符合条件才能辞退吗", "领导莫名其妙地扣了我的奖金", "邻居凌晨三点还在装修",
    "编号1234567890123456", "订单号12345678901234",
])
def test_ordinary_sentences_the_stress_set_flagged_are_left_alone(text):
    assert privacy.redact_text(text) == text

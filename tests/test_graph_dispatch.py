import json

import backend.graph as graph_module
import backend.rag as rag_module
from backend.graph import NO_RESULT_MARK, LeaderGraph, _has_legal_signal, _search_text, no_result_notice
from backend.memory import MemoryStore


def _leader(tmp_path):
    memory = MemoryStore(tmp_path / "state.sqlite3")
    return memory, LeaderGraph(memory)


def _ask(memory, leader, message):
    conversation = memory.create_conversation("u", message[:20])
    memory.add_message(conversation["id"], "u", "user", message)
    return leader.run(user_id="u", conversation_id=conversation["id"], message=message)


HIT = {"law_name": "劳动合同法", "article_number": "第三十条", "content": "及时足额支付劳动报酬。"}


def test_aid_questions_also_search_statutes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(graph_module, "search_statutes", lambda query: calls.append(query) or [HIT])
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "公司欠薪该怎么维权")

    assert result["intent"] == "aid"
    assert {worker["worker"] for worker in result["workers"]} == {"statute", "aid"}
    assert calls == ["公司欠薪该怎么维权"]
    assert NO_RESULT_MARK not in result["answer"]


def test_small_talk_does_not_search_or_trigger_the_remote(monkeypatch, tmp_path):
    def must_not_run(query):
        raise AssertionError("small talk must not reach statute search")

    monkeypatch.setattr(graph_module, "search_statutes", must_not_run)
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "你好")

    assert {worker["worker"] for worker in result["workers"]} == {"aid"}
    assert NO_RESULT_MARK not in result["answer"]


def test_legal_signal_detection():
    assert _has_legal_signal("我被公司口头辞退，没有给赔偿")
    assert _has_legal_signal("公司欠薪该怎么维权")
    assert not _has_legal_signal("你好")
    assert not _has_legal_signal("今天天气怎么样")


def test_remote_is_used_after_a_local_miss_and_suppresses_the_notice(monkeypatch, tmp_path):
    class EmptyEngine:
        def law_fuzzy_search(self, query, limit=5):
            return json.dumps({"success": False, "data": []})

    remote_calls = []
    monkeypatch.setattr(rag_module, "_load_engine", lambda: EmptyEngine())
    monkeypatch.setattr(
        "backend.yuandian_mcp.search_statutes",
        lambda query, limit: remote_calls.append(query) or [{**HIT, "source_name": "华宇元典 MCP"}],
    )
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "公司欠薪该怎么维权")

    assert remote_calls == ["公司欠薪该怎么维权"]
    assert result["evidence"][0]["source_name"] == "华宇元典 MCP"
    assert NO_RESULT_MARK not in result["answer"]


def test_last_resort_tells_the_user_nothing_was_found_and_to_see_a_lawyer(monkeypatch, tmp_path):
    class EmptyEngine:
        def law_fuzzy_search(self, query, limit=5):
            return json.dumps({"success": False, "data": []})

    monkeypatch.setattr(rag_module, "_load_engine", lambda: EmptyEngine())
    monkeypatch.setattr("backend.yuandian_mcp.search_statutes", lambda query, limit: [])
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "公司欠薪该怎么维权")

    assert result["evidence"] == []
    assert NO_RESULT_MARK in result["answer"]
    assert "没有找到相关条文" in result["answer"]
    assert "律师" in result["answer"] and "12348" in result["answer"]


def test_notice_names_each_kind_that_came_back_empty():
    both = no_result_notice({"statute": {"items": []}, "case": {"items": []}, "aid": {"items": [1]}})
    only_cases = no_result_notice({"statute": {"items": [HIT]}, "case": {"items": []}})

    assert "相关条文、相关案例" in both
    assert "相关案例" in only_cases and "相关条文" not in only_cases
    assert no_result_notice({"statute": {"items": [HIT]}, "aid": {"items": [1]}}) == ""
    assert no_result_notice({"aid": {"items": [1]}}) == ""


def test_notice_is_appended_even_when_the_model_ignores_the_instruction(monkeypatch, tmp_path):
    class Reply:
        content = "根据我的理解，劳动合同法第三十八条规定……"

    class Model:
        def invoke(self, messages):
            self.prompt = messages[-1].content
            return Reply()

    model = Model()
    monkeypatch.setattr(graph_module, "_llm", lambda: model)
    monkeypatch.setattr(graph_module, "search_statutes", lambda query: [])
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "民法典第577条规定是什么")

    assert result["answer"].startswith("根据我的理解")
    assert NO_RESULT_MARK in result["answer"]
    assert "不得凭记忆列举条文或案例" in model.prompt


def test_notice_is_not_duplicated_when_the_model_already_wrote_it(monkeypatch, tmp_path):
    class Reply:
        content = f"{NO_RESULT_MARK}\n未找到。"

    class Model:
        def invoke(self, messages):
            return Reply()

    monkeypatch.setattr(graph_module, "_llm", lambda: Model())
    monkeypatch.setattr(graph_module, "search_statutes", lambda query: [])
    memory, leader = _leader(tmp_path)

    result = _ask(memory, leader, "民法典第577条规定是什么")

    assert result["answer"].count(NO_RESULT_MARK) == 1


def test_short_follow_up_borrows_the_previous_user_question():
    state = {
        "message": "那赔多少",
        "history": [
            {"role": "user", "content": "公司拖欠我两个月工资"},
            {"role": "assistant", "content": "建议先固定证据。"},
        ],
    }

    assert _search_text(state) == "公司拖欠我两个月工资 那赔多少"
    assert _search_text({"message": "公司拖欠我两个月工资怎么办", "history": state["history"]}) == "公司拖欠我两个月工资怎么办"
    assert _search_text({"message": "那赔多少", "history": []}) == "那赔多少"

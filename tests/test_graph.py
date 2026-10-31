from pathlib import Path
from uuid import uuid4

from backend.graph import LeaderGraph
from backend.memory import MemoryStore


def test_langgraph_routes_and_resets_worker_outputs_between_turns(monkeypatch):
    import backend.graph as graph_module

    monkeypatch.setattr(
        graph_module,
        "search_statutes",
        lambda query: [{"law_name": "民法典", "article_number": "第五百七十七条", "content": "依法承担违约责任。"}],
    )
    monkeypatch.setattr(
        graph_module,
        "search_cases",
        lambda query: [{"title": "某合同纠纷案", "summary": "支持继续履行。"}],
    )
    database = Path("data/test_state")
    database.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(database / f"{uuid4().hex}.sqlite3")
    conversation = memory.create_conversation("user-a", "混合问题")
    leader = LeaderGraph(memory)
    assert leader.graph is not None

    first = "民法典合同违约的法条和类似案例有哪些？"
    memory.add_message(conversation["id"], "user-a", "user", first)
    result = leader.run(user_id="user-a", conversation_id=conversation["id"], message=first)
    assert result["intent"] == "mixed"
    assert {item["worker"] for item in result["workers"]} == {"statute", "case", "aid"}

    second = "帮我找类似判例"
    memory.add_message(conversation["id"], "user-a", "user", second)
    next_result = leader.run(user_id="user-a", conversation_id=conversation["id"], message=second)
    assert next_result["intent"] == "case"
    assert [item["worker"] for item in next_result["workers"]] == ["case"]
    assert len(next_result["evidence"]) == 1

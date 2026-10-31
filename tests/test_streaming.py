import json

import pytest
from fastapi.testclient import TestClient

import backend.graph as graph_module
import backend.rag as rag_module
from backend.app import create_app
from backend.graph import LeaderGraph

LAW = {"law_name": "中华人民共和国劳动合同法(2012修正)", "article_number": "第三十条", "effectiveness": "现行有效",
       "publish_date": "2012.12.28", "implement_date": "2013.07.01", "content": "用人单位应当按照约定向劳动者及时足额支付劳动报酬。"}


class Chunk:
    def __init__(self, text):
        self.content = text


class StreamingModel:
    """Streams a fixed answer in pieces; invoke is the non-streaming path."""

    def __init__(self, pieces):
        self.pieces = pieces

    def stream(self, messages):
        for piece in self.pieces:
            yield Chunk(piece)

    def invoke(self, messages):
        return Chunk("".join(self.pieces))


def parse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_module, "search_statutes", lambda q: [LAW])
    monkeypatch.setattr(graph_module, "search_cases", lambda q: [])
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    return TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))


def post(client, message, conversation_id=None):
    response = client.post("/api/chat/stream", json={"user_id": "u", "conversation_id": conversation_id, "message": message})
    return response, parse(response.text)


def test_the_stream_reports_progress_then_tokens_then_the_complete_reply(client, monkeypatch):
    monkeypatch.setattr(graph_module, "_llm", lambda: StreamingModel(["根据", "劳动合同法", "第三十条，", "应当及时支付工资。"]))

    response, events = post(client, "公司拖欠我两个月工资，月薪5000")

    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-2:] == ["final", "done"]
    assert names.index("route") < names.index("worker") < names.index("delta") < names.index("final")
    assert [d["text"] for n, d in events if n == "delta"] == ["根据", "劳动合同法", "第三十条，", "应当及时支付工资。"]
    stages = [d["name"] for n, d in events if n == "stage"]
    assert stages == ["understand", "retrieve", "compose", "verify"]
    final = dict(events)["final"]
    assert final["answer"].startswith("根据劳动合同法第三十条，应当及时支付工资。") and "### 案情梳理" in final["answer"]
    assert final["verification"]["checked"] >= 1 and final["case_file"]["case_type"] == "wage_arrears"


def test_workers_arrive_with_their_evidence_before_the_answer_starts(client, monkeypatch):
    monkeypatch.setattr(graph_module, "_llm", lambda: StreamingModel(["好的。"]))

    _, events = post(client, "公司拖欠我工资怎么办")

    workers = [d for n, d in events if n == "worker"]
    assert {o["worker"] for w in workers for o in w["outputs"]} == {"statute", "aid"}
    assert any(e["law_name"].endswith("(2012修正)") for w in workers for e in w["evidence"])


def test_the_final_reply_replaces_streamed_text_when_verification_edits_it(client, monkeypatch):
    streamed = ["依据《劳动合同法》第一百九十九条，", "您可以主张赔偿。"]
    monkeypatch.setattr(graph_module, "_llm", lambda: StreamingModel(streamed))
    from backend import verify

    class Engine:
        def law_exact_search(self, law, article):
            return json.dumps({"success": False, "message": "未找到法条: 中华人民共和国劳动合同法 " + article})

    monkeypatch.setattr(rag_module, "_load_engine", lambda: Engine())

    _, events = post(client, "劳动合同法第一百九十九条怎么规定")

    assert "".join(d["text"] for n, d in events if n == "delta") == "".join(streamed)
    final = dict(events)["final"]
    assert "已删除" in final["answer"] and final["verification"]["problems"] == 1


def test_streaming_persists_the_same_conversation_as_the_plain_endpoint(client, monkeypatch):
    monkeypatch.setattr(graph_module, "_llm", lambda: StreamingModel(["请先固定证据。"]))

    _, events = post(client, "公司拖欠我两个月工资")
    cid = dict(events)["meta"]["conversation_id"]

    stored = client.get(f"/api/conversations/{cid}", params={"user_id": "u"}).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant"] and stored[1]["content"].startswith("请先固定证据。")
    assert stored[1]["metadata"]["intent"] == dict(events)["final"]["intent"]
    again = post(client, "那我最多能要多少", cid)[1]
    assert dict(again)["final"]["conversation_id"] == cid
    assert len(client.get(f"/api/conversations/{cid}", params={"user_id": "u"}).json()["messages"]) == 4


def test_privacy_is_reported_up_front_and_nothing_raw_is_streamed(client, monkeypatch):
    seen = []

    class Spy(StreamingModel):
        def stream(self, messages):
            seen.append(" ".join(m.content for m in messages))
            return super().stream(messages)

    monkeypatch.setattr(graph_module, "_llm", lambda: Spy(["好的。"]))

    _, events = post(client, "我叫王小明，手机13812345678，公司拖欠我工资")

    assert dict(events)["meta"]["privacy"]["redacted"] == {"姓名": 1, "手机号": 1}
    assert "13812345678" not in seen[0] and "王小明" not in seen[0] and "13812345678" not in client.post("/api/chat/stream", json={"user_id": "u", "message": "x"}).text


def test_a_failure_while_streaming_ends_with_an_error_event_not_a_broken_connection(client):
    def boom(**kwargs):
        raise RuntimeError("model down")
        yield  # makes this a generator, like the real stream

    client.app.state.leader.stream = boom

    response, events = post(client, "公司拖欠我工资")

    assert response.status_code == 200 and [n for n, _ in events][-2:] == ["error", "done"]
    assert "model down" not in response.text  # the user sees a plain message, not the exception


def test_an_unknown_conversation_is_a_plain_404_before_any_stream_starts(client):
    response = client.post("/api/chat/stream", json={"user_id": "u", "conversation_id": "nope", "message": "你好"})
    assert response.status_code == 404 and response.headers["content-type"].startswith("application/json")


def test_without_a_model_the_reply_still_streams_to_a_final_event(client, monkeypatch):
    monkeypatch.setattr(graph_module, "_llm", lambda: None)

    _, events = post(client, "公司拖欠我工资")

    names = [n for n, _ in events]
    assert "delta" not in names and names[-2:] == ["final", "done"]
    assert "### 法条检索 Worker" in dict(events)["final"]["answer"]

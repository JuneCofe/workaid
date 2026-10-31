from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app


def test_chat_creates_persisted_conversation_and_returns_worker_results(monkeypatch):
    from backend.graph import LeaderGraph

    def fake_run(self, *, user_id, conversation_id, message):
        return {
            "intent": "aid",
            "answer": "先固定证据，再确认仲裁时限。",
            "workers": [{"worker": "aid", "label": "法律援助路径 Worker"}],
            "evidence": [],
        }

    monkeypatch.setattr(LeaderGraph, "run", fake_run)
    database = Path("data/test_state")
    database.mkdir(parents=True, exist_ok=True)
    app = create_app(
        database_path=database / f"{uuid4().hex}.sqlite3",
        frontend_dir=database / "no-frontend",
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"user_id": "user-a", "message": "拖欠工资怎么办？"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "aid"
    assert payload["workers"][0]["worker"] == "aid"
    conversation = client.get(
        f"/api/conversations/{payload['conversation_id']}?user_id=user-a"
    ).json()
    assert [item["role"] for item in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][1]["metadata"]["workers"][0]["worker"] == "aid"


def test_static_route_does_not_serve_files_outside_the_frontend_directory(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>", encoding="utf-8")
    (dist / "app.txt").write_text("INSIDE", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("OUTSIDE-STATIC-DIR", encoding="utf-8")
    client = TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=dist))

    assert client.get("/app.txt").text == "INSIDE"
    for attack in ("/..%2fsecret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt", "/assets/..%2f..%2fsecret.txt"):
        response = client.get(attack)
        assert "OUTSIDE-STATIC-DIR" not in response.text, attack


def test_statute_trace_endpoint_reports_pipeline_stages(monkeypatch, tmp_path):
    import backend.app as app_module

    monkeypatch.setattr(
        app_module,
        "search_statutes_traced",
        lambda query, limit=5, remote=True: ([{"law_name": "某法", "article_number": "第一条", "score": 0.9}], {"concepts": ["拖欠工资"], "remote": remote}),
    )
    client = TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))

    payload = client.get("/api/statutes/trace", params={"q": "公司拖欠工资"}).json()

    assert payload["items"][0]["score"] == 0.9
    assert payload["trace"] == {"concepts": ["拖欠工资"], "remote": False}


def test_sources_report_yuandian_fallback_without_exposing_credentials(monkeypatch):
    monkeypatch.delenv("YUANDIAN_MCP_API_KEY", raising=False)
    database = Path("data/test_state")
    database.mkdir(parents=True, exist_ok=True)
    app = create_app(
        database_path=database / f"{uuid4().hex}.sqlite3",
        frontend_dir=database / "no-frontend",
    )

    response = TestClient(app).get("/api/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mcp_status"]["configured"] is False
    assert payload["mcp_status"]["services"]["statute"]["configured"] is False
    assert payload["mcp_status"]["services"]["case"]["configured"] is False
    assert [item["kind"] for item in payload["remote"]] == [
        "yuandian-statute",
        "yuandian-case",
    ]
    assert "YUANDIAN_MCP_API_KEY" not in response.text

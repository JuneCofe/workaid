import json
import time

from fastapi.testclient import TestClient

import backend.rag as rag_module
from backend.app import create_app
from backend.warmup import Warmup


class FakeEngine:
    def __init__(self, fail_stats=False):
        self.fail_stats = fail_stats
        self.terms = []

    def law_article_count(self):
        return 123

    def law_term_doc_freq(self, term):
        if self.fail_stats:
            raise RuntimeError("stats down")
        self.terms.append(term)
        return 1

    def law_fuzzy_search(self, query, limit=5):
        return json.dumps({"success": False, "data": []})


def _wait(warmup, timeout=10):
    deadline = time.time() + timeout
    while warmup.snapshot()["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    return warmup.snapshot()


def test_warmup_loads_engine_statistics_and_index_in_the_background(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(rag_module, "_load_engine", lambda: engine)
    warmup = Warmup()

    warmup.start()
    snapshot = _wait(warmup)

    assert snapshot["status"] == "ready" and snapshot["error"] is None
    assert set(snapshot["steps"]) == {"statute_engine", "term_statistics", "vector_index", "first_query"}
    assert snapshot["steps"]["statute_engine"]["detail"] == {"articles": 123}
    assert "拖欠劳动报酬" in engine.terms


def test_warmup_failure_degrades_without_raising(monkeypatch):
    monkeypatch.setattr(rag_module, "_load_engine", lambda: FakeEngine(fail_stats=True))
    warmup = Warmup()

    warmup.start()
    snapshot = _wait(warmup)

    assert snapshot["status"] == "degraded"
    assert snapshot["steps"]["term_statistics"]["ok"] is False
    assert "term_statistics" in snapshot["error"]


def test_warmup_is_skipped_when_the_engine_is_missing_or_disabled(monkeypatch):
    monkeypatch.setattr(rag_module, "_load_engine", lambda: None)
    warmup = Warmup()
    warmup.start()
    assert _wait(warmup)["status"] == "skipped"

    monkeypatch.setenv("WORKAID_WARMUP", "0")
    disabled = Warmup()
    disabled.start()
    assert disabled.snapshot()["status"] == "idle"


def test_warmup_starts_with_the_app_and_health_reports_it(monkeypatch, tmp_path):
    from backend.retrieval import vector

    monkeypatch.setattr(vector, "INDEX_PATH", tmp_path / "none.npy")
    monkeypatch.setattr(vector, "META_PATH", tmp_path / "none.json")
    vector._STATE.update(mtime=None, index=None)
    monkeypatch.setattr(rag_module, "_load_engine", lambda: FakeEngine())
    app = create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none")

    with TestClient(app) as client:
        _wait(app.state.warmup)
        health = client.get("/api/health").json()

    assert health["warmup"]["status"] == "ready"
    assert health["vector_index"]["index_ready"] is False
    assert "EMBEDDING_API_KEY" not in json.dumps(health)

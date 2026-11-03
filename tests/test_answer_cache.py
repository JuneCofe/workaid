import json

import pytest
from fastapi.testclient import TestClient

from backend import answer_cache
from backend import app as app_module
from backend.answer_cache import AnswerCache, eligible, key_for, normalize
from backend.graph import LeaderGraph


class FakeRedis:
    """The subset of redis-py the cache uses, in memory."""

    def __init__(self):
        self.data = {}
        self.ttls = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttls[key] = ex
        return True

    def ping(self):
        return True


class DownRedis:
    def get(self, key):
        raise ConnectionError("refused")

    set = get

    def ping(self):
        raise ConnectionError("refused")


REPLY = {
    "intent": "statute",
    "scripts": [],
    "answer": "根据《劳动合同法》第三十条，用人单位应当及时足额支付劳动报酬。",
    "answer_stored": "根据《劳动合同法》第三十条，用人单位应当及时足额支付劳动报酬。",
    "workers": [{"worker": "statute", "label": "法条检索 Worker", "summary": "检索到本地法规条文。", "items": []}],
    "evidence": [],
    "verification": {"checked": 1, "problems": 0},
    "case_file": None,
}


def test_normalize_ignores_case_width_and_punctuation():
    assert normalize("公司拖欠工资，怎么办？") == normalize("公司拖欠工资怎么办")
    assert normalize("ＡＢＣ 劳动合同") == normalize("abc劳动合同")


def test_key_changes_with_the_corpus_version(monkeypatch):
    before = key_for("公司拖欠工资怎么办")
    monkeypatch.setattr(answer_cache, "corpus_version", lambda: "rebuilt")
    assert key_for("公司拖欠工资怎么办") != before


def test_only_first_neutral_questions_are_shareable():
    question = "公司拖欠工资怎么办"
    assert eligible(question, first_turn=True, profile_empty=True, redacted=False)
    assert not eligible(question, first_turn=False, profile_empty=True, redacted=False)
    assert not eligible(question, first_turn=True, profile_empty=False, redacted=False)
    assert not eligible(question, first_turn=True, profile_empty=True, redacted=True)
    assert not eligible("你好", first_turn=True, profile_empty=True, redacted=False)


def test_cache_round_trip_and_ttl():
    cache = AnswerCache("redis://fake", ttl=120)
    cache._client = FakeRedis()
    cache.set("k", REPLY)

    assert cache.get("k") == REPLY
    assert cache._client.ttls[answer_cache.KEY_PREFIX + "k"] == 120
    assert cache.status() == {"enabled": True, "reachable": True, "ttl_seconds": 120}


def test_unreachable_redis_is_a_miss_and_opens_a_cooldown(monkeypatch):
    cache = AnswerCache("redis://fake", ttl=60)
    cache._client = DownRedis()
    clock = [1000.0]
    monkeypatch.setattr(answer_cache.time, "monotonic", lambda: clock[0])

    assert cache.get("k") is None
    assert cache.status()["reachable"] is False
    cache.set("k", REPLY)  # swallowed; no second network call during the cooldown
    clock[0] += answer_cache.COOLDOWN_SECONDS + 1
    assert cache.get("k") is None


def test_disabled_cache_never_touches_the_network():
    cache = AnswerCache("", ttl=60)
    assert cache.enabled is False
    assert cache.get("k") is None
    assert cache.status()["reachable"] is False


@pytest.fixture()
def shared_cache(monkeypatch, tmp_path):
    cache = AnswerCache("redis://fake", ttl=600)
    cache._client = FakeRedis()
    monkeypatch.setattr(app_module, "AnswerCache", lambda url, ttl: cache)
    calls = []

    def fake_run(self, *, user_id, conversation_id, message):
        calls.append(message)
        return json.loads(json.dumps(REPLY))

    monkeypatch.setattr(LeaderGraph, "run", fake_run)
    client = TestClient(app_module.create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))
    return client, calls


def test_repeat_question_is_answered_from_cache(shared_cache):
    client, calls = shared_cache
    question = "公司拖欠工资怎么办"

    first = client.post("/api/chat", json={"user_id": "u1", "message": question}).json()
    second = client.post("/api/chat", json={"user_id": "u2", "message": question}).json()

    assert len(calls) == 1
    assert second["answer"] == first["answer"] == REPLY["answer"]
    assert second["conversation_id"] != first["conversation_id"]
    history = client.get(f"/api/conversations/{second['conversation_id']}", params={"user_id": "u2"}).json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


def test_personal_questions_are_never_cached(shared_cache):
    client, calls = shared_cache
    question = "我叫王小明，手机13812345678，公司拖欠工资怎么办"

    client.post("/api/chat", json={"user_id": "u1", "message": question})
    client.post("/api/chat", json={"user_id": "u1", "message": question})

    assert len(calls) == 2


def test_cached_stream_emits_the_same_event_sequence(shared_cache):
    client, calls = shared_cache
    question = "公司拖欠工资怎么办"
    client.post("/api/chat", json={"user_id": "u1", "message": question})

    with client.stream("POST", "/api/chat/stream", json={"user_id": "u1", "message": question}) as response:
        body = "".join(response.iter_text())

    events = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
    assert len(calls) == 1
    assert events[0] == "meta" and events[-2:] == ["final", "done"]
    assert "route" in events and "delta" in events
    assert "".join(
        json.loads(line.removeprefix("data: "))["text"]
        for line in body.splitlines() if line.startswith("data: ") and '"text"' in line
    ) == REPLY["answer"]


def test_health_reports_engine_and_cache_without_leaking_urls(shared_cache, monkeypatch):
    client, _ = shared_cache
    health = client.get("/api/health").json()

    assert health["memory"] == {"engine": "sqlite"}
    assert health["answer_cache"]["enabled"] is True
    assert "redis://" not in json.dumps(health)

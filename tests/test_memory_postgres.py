"""Runs the memory store against a real PostgreSQL server. Skipped unless TEST_POSTGRES_URL is set, e.g.

    TEST_POSTGRES_URL=postgresql://postgres@127.0.0.1:5433/postgres pytest tests/test_memory_postgres.py

Each test gets its own schema, which is dropped afterwards, so the server's other data is untouched.
"""
import os
from uuid import uuid4

import psycopg
import pytest

from backend import profile
from backend.memory import MemoryStore

BASE_URL = os.getenv("TEST_POSTGRES_URL", "").strip()
pytestmark = pytest.mark.skipif(not BASE_URL, reason="TEST_POSTGRES_URL is not set")


@pytest.fixture()
def pg_store():
    schema = f"wa_test_{uuid4().hex[:12]}"
    with psycopg.connect(BASE_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    separator = "&" if "?" in BASE_URL else "?"
    url = f"{BASE_URL}{separator}options=-csearch_path%3D{schema}"
    store = MemoryStore(url=url)
    try:
        yield store
    finally:
        store.db.close()
        with psycopg.connect(BASE_URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_engine_label_is_postgresql(pg_store):
    assert pg_store.db.label == "postgresql"
    assert pg_store.path is None


def test_history_and_profile_are_user_scoped(pg_store):
    conversation = pg_store.create_conversation("user-a", "租房纠纷")
    pg_store.add_message(conversation["id"], "user-a", "user", "房东不退押金")
    pg_store.add_message(conversation["id"], "user-a", "assistant", "先保留租赁合同。", {"intent": "aid"})
    pg_store.save_profile("user-a", ["常住地：上海", "常住地：上海"], {"style": "简洁"})

    history = pg_store.get_conversation(conversation["id"], "user-a")
    assert [item["role"] for item in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["metadata"] == {"intent": "aid"}
    assert pg_store.get_conversation(conversation["id"], "user-b") is None
    assert pg_store.profile("user-a")["facts"] == ["常住地：上海"]
    assert pg_store.profile("user-b")["facts"] == []


def test_delete_conversation_cascades_messages(pg_store):
    conversation = pg_store.create_conversation("user-a", "Test")
    pg_store.add_message(conversation["id"], "user-a", "user", "question")

    assert pg_store.delete_conversation(conversation["id"], "other-user") is False
    assert pg_store.delete_conversation(conversation["id"], "user-a") is True
    assert pg_store.get_conversation(conversation["id"], "user-a") is None


def test_profile_items_upsert_keep_one_value_for_single_valued_keys(pg_store):
    pg_store.upsert_profile_items("u", [{"key": "所在地", "value": "北京", "confidence": 0.6}])
    pg_store.upsert_profile_items("u", [{"key": "所在地", "value": "上海", "confidence": 0.7}])
    pg_store.upsert_profile_items("u", [{"key": "所在地", "value": "上海", "confidence": 0.5, "explicit": True}])

    items = pg_store.list_profile_items("u")
    assert [(i["value"], i["explicit"], i["confidence"]) for i in items] == [("上海", 1, 0.7)]
    assert items[0]["expires_at"] is None  # an explicit item never expires


def test_sensitive_items_stay_out_of_the_prompt_profile(pg_store):
    pg_store.upsert_profile_items("u", [
        {"key": "姓名", "value": "王小明", "explicit": True, "sensitive": True},
        {"key": "所在地", "value": "北京", "confidence": 0.6},
    ])

    prompt = pg_store.profile_for_prompt("u")
    assert "王小明" not in prompt["已了解的情况"]
    assert "北京" in prompt["已了解的情况"]


def test_delete_user_data_and_expiry(pg_store):
    conversation = pg_store.create_conversation("u", "t")
    pg_store.add_message(conversation["id"], "u", "user", "hi")
    pg_store.upsert_profile_items("u", [{"key": "月薪", "value": "5000", "confidence": 0.6}])
    pg_store.save_case_file(conversation["id"], {"types": ["wage_arrears"]})
    pg_store.save_summary(conversation["id"], "摘要", 1)

    report = pg_store.delete_user_data("u")

    assert report == {"conversations": 1, "profile_items": 1, "profiles": 0}
    assert pg_store.get_case_file(conversation["id"]) is None  # cascaded from the conversation
    assert pg_store.get_summary(conversation["id"]) is None


def test_derived_profile_items_expire(pg_store):
    pg_store.upsert_profile_items("u", [{"key": "行业", "value": "餐饮", "confidence": 0.6}])
    with pg_store._connect() as db:
        db.execute("UPDATE profile_items SET expires_at=? WHERE user_id=?", ("2000-01-01T00:00:00+00:00", "u"))

    assert pg_store.list_profile_items("u") == []
    assert pg_store.purge_expired() == 1


def test_profile_extraction_runs_on_postgres(pg_store):
    conversation = pg_store.create_conversation("u", "t")
    profile.update_from_text(pg_store, "u", "我在北京做服务员，月薪5000，公司拖欠工资", conversation["id"], use_model=False)

    keys = {item["key"] for item in pg_store.list_profile_items("u")}
    assert {"所在地", "月薪"} <= keys

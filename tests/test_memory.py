from pathlib import Path
from uuid import uuid4

from backend.memory import MemoryStore


def _database_path() -> Path:
    root = Path("data/test_state")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid4().hex}.sqlite3"


def test_conversation_history_and_profile_are_user_scoped():
    store = MemoryStore(_database_path())
    conversation = store.create_conversation("user-a", "租房纠纷")
    store.add_message(conversation["id"], "user-a", "user", "房东不退押金")
    store.add_message(
        conversation["id"],
        "user-a",
        "assistant",
        "先保留租赁合同。",
        {"intent": "aid"},
    )
    store.save_profile("user-a", ["常住地：上海", "常住地：上海"], {"style": "简洁"})

    history = store.get_conversation(conversation["id"], "user-a")
    assert [item["role"] for item in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["metadata"] == {"intent": "aid"}
    assert store.get_conversation(conversation["id"], "user-b") is None
    assert store.profile("user-a")["facts"] == ["常住地：上海"]
    assert store.profile("user-b")["facts"] == []


def test_delete_conversation_cascades_messages():
    store = MemoryStore(_database_path())
    conversation = store.create_conversation("user-a", "Test")
    store.add_message(conversation["id"], "user-a", "user", "question")

    assert store.delete_conversation(conversation["id"], "other-user") is False
    assert store.delete_conversation(conversation["id"], "user-a") is True
    assert store.get_conversation(conversation["id"], "user-a") is None

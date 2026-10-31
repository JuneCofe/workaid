from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from . import privacy

_LOCK = threading.RLock()
ITEM_TTL_DAYS = 180
SINGLE_VALUED = frozenset({"所在地", "行业岗位", "入职时间", "离职时间", "工作时长", "月薪", "劳动合同", "社保", "用工形式", "回答偏好"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _redact_values(value: Any) -> Any:
    """Redact every string inside nested data without touching its structure or keys."""
    if isinstance(value, str):
        return privacy.redact_text(value)
    if isinstance(value, list):
        return [_redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_values(item) for key, item in value.items()}
    return value


def _expiry() -> str:
    return (datetime.now(UTC) + timedelta(days=ITEM_TTL_DAYS)).isoformat()


class MemoryStore:
    """SQLite conversation history plus durable, user-scoped profile memory."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_conversations_user_updated
                    ON conversations(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    facts TEXT NOT NULL DEFAULT '[]',
                    preferences TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profile_items (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    explicit INTEGER NOT NULL DEFAULT 0,
                    sensitive INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.6,
                    source TEXT NOT NULL DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    expires_at TEXT,
                    PRIMARY KEY (user_id, key, value)
                );
                CREATE TABLE IF NOT EXISTS case_files (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    covered INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_conversation(self, user_id: str, title: str) -> dict[str, Any]:
        conversation_id, now = str(uuid4()), _now()
        title = privacy.redact_text(title)
        with _LOCK, self._connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, title[:90] or "新对话", now, now),
            )
        return {"id": conversation_id, "title": title[:90] or "新对话", "updated_at": now}

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.id, c.title, c.updated_at, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id=c.id
                WHERE c.user_id=?
                GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 100
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            messages = db.execute(
                "SELECT role, content, metadata, created_at FROM messages "
                "WHERE conversation_id=? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return {
            "id": row["id"],
            "title": row["title"],
            "updated_at": row["updated_at"],
            "messages": [
                {
                    "role": item["role"],
                    "content": item["content"],
                    "metadata": json.loads(item["metadata"]),
                    "created_at": item["created_at"],
                }
                for item in messages
            ],
        }

    def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Enforced here rather than at each call site: no code path can store un-redacted chat text.
        content = privacy.redact_text(content)
        now = _now()
        with _LOCK, self._connect() as db:
            exists = db.execute(
                "SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            ).fetchone()
            if exists is None:
                raise KeyError("会话不存在")
            db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    conversation_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with _LOCK, self._connect() as db:
            cursor = db.execute(
                "DELETE FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            )
            return cursor.rowcount > 0

    def profile(self, user_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT facts, preferences FROM user_profiles WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            return {"user_id": user_id, "facts": [], "preferences": {}}
        return {
            "user_id": user_id,
            "facts": json.loads(row["facts"]),
            "preferences": json.loads(row["preferences"]),
        }

    def save_profile(
        self, user_id: str, facts: list[str], preferences: dict[str, str]
    ) -> dict[str, Any]:
        # A person typing into their own profile may keep a name or employer; ID/phone/card numbers never.
        facts = [privacy.redact_stored(fact) for fact in facts]
        preferences = {k: privacy.redact_stored(str(v)) for k, v in preferences.items()}
        normalized_facts = list(dict.fromkeys(f.strip()[:300] for f in facts if f.strip()))[:30]
        safe_preferences = {
            str(key)[:80]: str(value)[:300]
            for key, value in preferences.items()
            if str(key).strip()
        }
        with _LOCK, self._connect() as db:
            db.execute(
                """
                INSERT INTO user_profiles(user_id, facts, preferences, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    facts=excluded.facts, preferences=excluded.preferences,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(normalized_facts, ensure_ascii=False),
                    json.dumps(safe_preferences, ensure_ascii=False),
                    _now(),
                ),
            )
        return {"user_id": user_id, "facts": normalized_facts, "preferences": safe_preferences}

    # ------------------------------------------------------------------ derived profile

    def upsert_profile_items(self, user_id: str, items: list[dict[str, Any]]) -> int:
        """Insert or refresh derived/explicit profile items; returns how many rows were written.

        Values are redacted unless the item is an explicit, sensitive one (a name, employer or address
        the user asked to be remembered). A single-valued key keeps only its newest derived value.
        """
        written = 0
        now, expires = _now(), _expiry()
        with _LOCK, self._connect() as db:
            for item in items:
                key = str(item["key"])[:40]
                sensitive = bool(item.get("sensitive"))
                explicit = bool(item.get("explicit")) or sensitive
                value = str(item["value"]).strip()
                value = privacy.redact_stored(value) if sensitive else privacy.redact_text(value)
                value = value[:200]
                if not key or not value:
                    continue
                if key in SINGLE_VALUED and not explicit:
                    db.execute(
                        "DELETE FROM profile_items WHERE user_id=? AND key=? AND value<>? AND explicit=0",
                        (user_id, key, value),
                    )
                db.execute(
                    """
                    INSERT INTO profile_items(user_id, key, value, explicit, sensitive, confidence, source,
                                              first_seen, last_seen, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, key, value) DO UPDATE SET
                        last_seen=excluded.last_seen, expires_at=excluded.expires_at,
                        confidence=MAX(confidence, excluded.confidence),
                        explicit=MAX(explicit, excluded.explicit)
                    """,
                    (user_id, key, value, int(explicit), int(sensitive), float(item.get("confidence", 0.6)),
                     str(item.get("source", ""))[:60], now, now, None if explicit else expires),
                )
                written += 1
        return written

    def list_profile_items(self, user_id: str) -> list[dict[str, Any]]:
        now = _now()
        with self._connect() as db:
            rows = db.execute(
                "SELECT key, value, explicit, sensitive, confidence, source, first_seen, last_seen, expires_at "
                "FROM profile_items WHERE user_id=? AND (expires_at IS NULL OR expires_at>?) "
                "ORDER BY explicit DESC, key, last_seen DESC",
                (user_id, now),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_profile_item(self, user_id: str, key: str, value: str) -> bool:
        with _LOCK, self._connect() as db:
            cursor = db.execute(
                "DELETE FROM profile_items WHERE user_id=? AND key=? AND value=?", (user_id, key, value)
            )
            return cursor.rowcount > 0

    def purge_expired(self) -> int:
        with _LOCK, self._connect() as db:
            return db.execute(
                "DELETE FROM profile_items WHERE expires_at IS NOT NULL AND expires_at<=?", (_now(),)
            ).rowcount

    def profile_for_prompt(self, user_id: str, limit: int = 600) -> dict[str, Any]:
        """Profile as sent to a language model: stored facts and derived items, never the values of
        names, employers or addresses the user asked to remember."""
        base = self.profile(user_id)
        parts: dict[str, list[str]] = {}
        for item in self.list_profile_items(user_id):
            if item["sensitive"]:
                continue
            parts.setdefault(item["key"], []).append(item["value"])
        digest = "；".join(f"{key}：{'、'.join(values)}" for key, values in parts.items())[:limit]
        return {"facts": base["facts"], "preferences": base["preferences"], "已了解的情况": digest}

    # ------------------------------------------------------------------ conversation summaries

    def get_summary(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT summary, covered, updated_at FROM conversation_summaries WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_summary(self, conversation_id: str, summary: str, covered: int) -> None:
        with _LOCK, self._connect() as db:
            db.execute(
                """
                INSERT INTO conversation_summaries(conversation_id, summary, covered, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary=excluded.summary, covered=excluded.covered, updated_at=excluded.updated_at
                """,
                (conversation_id, privacy.redact_text(summary)[:1200], covered, _now()),
            )

    # ------------------------------------------------------------------ case file

    def get_case_file(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT data FROM case_files WHERE conversation_id=?", (conversation_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    def save_case_file(self, conversation_id: str, data: dict[str, Any]) -> None:
        """Case data holds only whitelisted fields and already-redacted text; text values are redacted again."""
        raw = json.dumps(_redact_values(data), ensure_ascii=False)
        with _LOCK, self._connect() as db:
            db.execute(
                """
                INSERT INTO case_files(conversation_id, data, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
                """,
                (conversation_id, raw, _now()),
            )

    # ------------------------------------------------------------------ deletion and clean-up

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Remove everything stored for a user: conversations, messages, summaries, profile and items."""
        with _LOCK, self._connect() as db:
            conversations = db.execute("DELETE FROM conversations WHERE user_id=?", (user_id,)).rowcount
            items = db.execute("DELETE FROM profile_items WHERE user_id=?", (user_id,)).rowcount
            profiles = db.execute("DELETE FROM user_profiles WHERE user_id=?", (user_id,)).rowcount
        return {"conversations": conversations, "profile_items": items, "profiles": profiles}

    def scrub_stored_pii(self, apply: bool = False) -> dict[str, Any]:
        """Redact chat text and titles that were stored before redaction was enforced.

        Dry-run by default: reports what would change without writing. With apply=True the rows are
        rewritten and the file is vacuumed so the old text does not linger in free pages.
        """
        report: dict[str, Any] = {"messages": 0, "titles": 0, "kinds": {}}
        with _LOCK, self._connect() as db:
            for table, column, label in (("messages", "content", "messages"), ("conversations", "title", "titles")):
                rows = db.execute(f"SELECT rowid, {column} FROM {table}").fetchall()
                for row in rows:
                    result = privacy.analyze(row[column])
                    if not result.changed:
                        continue
                    report[label] += 1
                    for kind, number in result.counts.items():
                        report["kinds"][kind] = report["kinds"].get(kind, 0) + number
                    if apply:
                        db.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (result.text, row["rowid"]))
        if apply:
            db = sqlite3.connect(self.path)
            try:
                db.execute("VACUUM")
            finally:
                db.close()
        report["applied"] = apply
        return report

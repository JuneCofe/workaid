from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# One schema for both engines. REAL is spelled per dialect where the two disagree.
SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conversations_user_updated ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, created_at);
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
    confidence {REAL} NOT NULL DEFAULT 0.6,
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

TABLES = ("conversations", "messages", "user_profiles", "profile_items", "case_files", "conversation_summaries")


class Session:
    """One connection for one unit of work. Placeholders are written as `?` and translated for Postgres."""

    def __init__(self, conn: Any, dialect: str):
        self.conn = conn
        self.dialect = dialect

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        if self.dialect == "postgres":
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, tuple(params))


class Database:
    """SQLite file by default; PostgreSQL when a URL is given (MEMORY_DATABASE_URL)."""

    def __init__(self, path: Path | None = None, url: str = ""):
        self.url = url.strip()
        self.dialect = "postgres" if self.url else "sqlite"
        self.path = None if self.url else Path(path)
        self._pool = None
        if self.dialect == "postgres":
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self.url,
                min_size=1,
                max_size=int(os.getenv("MEMORY_POOL_MAX", "5")),
                kwargs={"row_factory": dict_row},
                open=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        """Engine name for health output; never the URL, which may carry a password."""
        return "postgresql" if self.dialect == "postgres" else "sqlite"

    @contextmanager
    def connect(self) -> Iterator[Session]:
        if self._pool is not None:
            with self._pool.connection() as conn:
                yield Session(conn, "postgres")
            return
        conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield Session(conn, "sqlite")
        finally:
            conn.close()

    def create_schema(self) -> None:
        ddl = SCHEMA.replace("{REAL}", "DOUBLE PRECISION" if self.dialect == "postgres" else "REAL")
        with self.connect() as session:
            for statement in ddl.split(";"):
                if statement.strip():
                    session.execute(statement)

    def vacuum(self) -> None:
        """Reclaim free pages so erased text does not linger (SQLite only; Postgres autovacuum handles it)."""
        if self.dialect != "sqlite":
            return
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

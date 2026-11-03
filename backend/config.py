from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path_setting(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    path = Path(configured).expanduser() if configured else default
    return path if path.is_absolute() else ROOT / path


DATABASE_PATH = _path_setting("DATABASE_PATH", ROOT / "data" / "workaid.sqlite3")
CASE_CATALOG_PATH = _path_setting("CASE_CATALOG_PATH", ROOT / "data" / "cases.jsonl")
SCRIPTS_PATH = _path_setting("SCRIPTS_PATH", ROOT / "data" / "scripts.jsonl")
INSTITUTIONS_PATH = _path_setting("INSTITUTIONS_PATH", ROOT / "data" / "institutions.jsonl")
RAG_DATA_DIR = ROOT / "rag" / "data"
RAG_CACHE_DIR = ROOT / "rag" / "cache"
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "24"))
MEMORY_DATABASE_URL = os.getenv("MEMORY_DATABASE_URL", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()
ANSWER_CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", "86400"))

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import CASE_CATALOG_PATH, INSTITUTIONS_PATH, RAG_CACHE_DIR, SCRIPTS_PATH

logger = logging.getLogger(__name__)

# Bump when the answer prompt or the shape of a cached reply changes; old entries then stop matching.
PROMPT_VERSION = "1"
KEY_PREFIX = "workaid:answer:"
MIN_QUESTION_CHARS = 4
# After a failed Redis call the cache stays off for this long, so an unreachable server costs one timeout.
COOLDOWN_SECONDS = 30.0
_PUNCTUATION = re.compile(r"[\W_]+")


def normalize(question: str) -> str:
    """Case, width and punctuation do not make a different question."""
    return _PUNCTUATION.sub("", unicodedata.normalize("NFKC", question).lower())


def _stamp(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "-"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def corpus_version() -> str:
    """Everything an answer depends on besides the question and the user. A rebuilt statute index, a changed
    case catalogue or a different model or prompt version gives a new version, so old answers stop matching."""
    parts = [
        PROMPT_VERSION,
        os.getenv("OPENAI_MODEL", ""),
        os.getenv("EMBEDDING_MODEL", ""),
        os.getenv("ANSWER_CACHE_VERSION", ""),
        _stamp(RAG_CACHE_DIR / "law.db"),
        _stamp(RAG_CACHE_DIR / "manifest.json"),
        _stamp(RAG_CACHE_DIR / "chroma" / "chroma.sqlite3"),
        _stamp(CASE_CATALOG_PATH),
        _stamp(SCRIPTS_PATH),
        _stamp(INSTITUTIONS_PATH),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def key_for(question: str) -> str:
    digest = hashlib.sha256(f"{corpus_version()}\0{normalize(question)}".encode("utf-8")).hexdigest()
    return digest


def eligible(question: str, *, first_turn: bool, profile_empty: bool, redacted: bool) -> bool:
    """Only answers that do not depend on this user are shared: a first question, no profile facts after it
    was read, and nothing redacted (a redacted question may still carry the user's situation)."""
    return (
        first_turn
        and profile_empty
        and not redacted
        and len(normalize(question)) >= MIN_QUESTION_CHARS
    )


class AnswerCache:
    """Full replies for common questions, in Redis. Every failure is a miss, never an error for the user."""

    def __init__(self, url: str, ttl: int):
        self.url = url.strip()
        self.ttl = max(1, int(ttl))
        self._client: Any = None
        self._down_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(
                self.url, decode_responses=True, socket_timeout=0.3, socket_connect_timeout=0.3
            )
        return self._client

    def _available(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    def _failed(self, exc: Exception) -> None:
        self._down_until = time.monotonic() + COOLDOWN_SECONDS
        logger.warning("answer cache unavailable: %s", type(exc).__name__)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._available():
            return None
        try:
            raw = self._redis().get(KEY_PREFIX + key)
        except Exception as exc:
            self._failed(exc)
            return None
        return json.loads(raw) if raw else None

    def set(self, key: str, reply: dict[str, Any]) -> None:
        if not self._available():
            return
        payload = json.dumps(reply, ensure_ascii=False)
        try:
            self._redis().set(KEY_PREFIX + key, payload, ex=self.ttl)
        except Exception as exc:
            self._failed(exc)

    def status(self) -> dict[str, Any]:
        reachable = False
        if self._available():
            try:
                reachable = bool(self._redis().ping())
            except Exception as exc:
                self._failed(exc)
        return {"enabled": self.enabled, "reachable": reachable, "ttl_seconds": self.ttl}

from __future__ import annotations

import os
import threading
import time
from typing import Any

from .retrieval import vector
from .retrieval.lexicon import CONCEPTS


class Warmup:
    """Loads the statute engine, term statistics and vector index in the background at startup.

    Requests that arrive before it finishes still work; they just wait on the same engine lock the
    warm-up holds instead of paying the cost themselves later.
    """

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"status": "idle", "steps": {}, "seconds": None, "error": None}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if os.getenv("WORKAID_WARMUP", "1") == "0" or self._thread is not None:
            return
        self.state["status"] = "running"
        self._thread = threading.Thread(target=self._run, name="workaid-warmup", daemon=True)
        self._thread.start()

    def _step(self, name: str, action: Any) -> None:
        began = time.perf_counter()
        try:
            detail = action()
            self.state["steps"][name] = {"ok": True, "seconds": round(time.perf_counter() - began, 2), "detail": detail}
        except Exception as exc:  # warm-up must never take the service down
            self.state["steps"][name] = {"ok": False, "seconds": round(time.perf_counter() - began, 2), "detail": str(exc)[:200]}

    def _run(self) -> None:
        began = time.perf_counter()
        from . import rag

        module = rag._load_engine()
        if module is None:
            self.state.update(status="skipped", error="local statute engine unavailable")
            return
        self._step("statute_engine", lambda: {"articles": module.law_article_count()})
        self._step("term_statistics", lambda: self._term_statistics(module))
        self._step("vector_index", lambda: vector.status())
        self._step("first_query", lambda: len(rag.search_statutes_local("公司拖欠工资", 1)))
        failed = [name for name, step in self.state["steps"].items() if not step["ok"]]
        self.state.update(
            status="degraded" if failed else "ready",
            seconds=round(time.perf_counter() - began, 2),
            error=f"failed steps: {', '.join(failed)}" if failed else None,
        )

    @staticmethod
    def _term_statistics(module: Any) -> int:
        """Document frequencies feed the IDF weights; computing them here keeps the first query fast."""
        terms = {t for concept in CONCEPTS for t in (*concept.formal, *concept.keywords)}
        for term in terms:
            module.law_term_doc_freq(term)
        return len(terms)

    def snapshot(self) -> dict[str, Any]:
        return {**self.state, "steps": dict(self.state["steps"])}

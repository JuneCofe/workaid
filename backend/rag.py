from __future__ import annotations

import json
import re
from typing import Any

from .config import RAG_CACHE_DIR, RAG_DATA_DIR
from .retrieval import search_local


def _load_engine() -> Any | None:
    """Load WorkAid's local SQLite/FTS-backed statute search engine."""
    if not RAG_DATA_DIR.is_dir():
        return None
    try:
        from rag import law_data_search

        return law_data_search
    except Exception:
        return None


def engine_status() -> dict[str, Any]:
    if not RAG_DATA_DIR.exists():
        return {"available": False, "source": str(RAG_DATA_DIR), "documents": 0}
    count = sum(1 for item in RAG_DATA_DIR.rglob("*") if item.is_file())
    return {
        "available": count > 0,
        "source": str(RAG_DATA_DIR),
        "documents": count,
        "indexed": (RAG_CACHE_DIR / "law.db").exists(),
    }


def search_statutes_traced(
    query: str, limit: int = 5, *, remote: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Local retrieval first (exact citation, then rewrite/recall/rerank); remote only on a miss."""
    module = _load_engine()
    exact: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}
    citation = re.search(
        r"《([^》\n]{2,80})》\s*(第[一二三四五六七八九十百千万零〇两0-9]+条(?:之[一二三四五六七八九十百千万零〇两0-9]+)?)",
        query,
    )
    if module is not None and citation:
        try:
            payload = json.loads(module.law_exact_search(citation.group(1), citation.group(2)))
            item = payload.get("data") if payload.get("success") else None
            if isinstance(item, dict):
                exact.append(item)
        except Exception:
            pass
    if module is not None:
        try:
            ranked, trace = search_local(module, query, limit)
            seen = {(item.get("law_name", ""), item.get("article_number", "")) for item in exact}
            exact.extend(
                item for item in ranked
                if (item.get("law_name", ""), item.get("article_number", "")) not in seen
            )
        except Exception:
            pass
    if exact:
        return exact[:limit], trace
    if not remote:
        return [], trace

    from .yuandian_mcp import search_statutes as search_remote_statutes

    return search_remote_statutes(query, limit), {**trace, "remote_fallback": True}


def search_statutes(query: str, limit: int = 5) -> list[dict[str, Any]]:
    return search_statutes_traced(query, limit)[0]


def search_statutes_local(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Local corpus only; never sends the query to a remote provider."""
    return search_statutes_traced(query, limit, remote=False)[0]


def exact_statute(title: str, article_number: str) -> dict[str, Any] | None:
    module = _load_engine()
    if module is None:
        return None
    try:
        payload = json.loads(module.law_exact_search(title, article_number))
    except Exception:
        return None
    item = payload.get("data") if payload.get("success") else None
    return item if isinstance(item, dict) else None


def statute_sources() -> list[dict[str, Any]]:
    status = engine_status()
    if not status["available"]:
        return []
    categories: dict[str, int] = {}
    for path in RAG_DATA_DIR.iterdir():
        if path.is_dir():
            categories[path.name] = sum(1 for item in path.rglob("*") if item.is_file())
    return [
        {
            "title": "WorkAid 本地法律法规库",
            "category": "法规总库",
            "article_count": status["documents"],
            "path": str(RAG_DATA_DIR),
            "kind": "statute",
            "categories": categories,
        }
    ]

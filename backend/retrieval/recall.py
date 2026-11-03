from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .query import QueryPlan

RRF_K = 60
PER_VARIANT_LIMIT = 15
BM25_TOP_K = 10
VECTOR_WEIGHT = 1.0
# Whole clauses are not word lists, so a BM25 phrase query over them only adds noise.
_BM25_SKIP_KINDS = {"original", "chunk"}


def rrf_k() -> int:
    return int(os.getenv("RRF_K", RRF_K))


def bm25_top_k() -> int:
    return int(os.getenv("BM25_TOP_K", BM25_TOP_K))


def fuzzy_enabled() -> bool:
    return os.getenv("RECALL_FUZZY", "1") != "0"


@dataclass
class Candidate:
    item: dict[str, Any]
    rrf: float = 0.0
    bm25: float = 0.0
    vec: float = 0.0
    routes: set[str] = field(default_factory=set)
    lists: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return self.item["law_name"], self.item["article_number"]


def _call(module: Any, name: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    function = getattr(module, name, None)
    if function is None:
        return []
    try:
        payload = json.loads(function(*args, **kwargs))
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    return [item for item in data or [] if isinstance(item, dict)]


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "law_name": str(item.get("law_name", "")),
        "article_number": str(item.get("article_number", "")),
        "content": str(item.get("content", "")),
        "category": str(item.get("category", "")),
        "url": str(item.get("url", "")),
        "source_file": str(item.get("source_file", "")),
        "publish_date": str(item.get("publish_date", "")),
        "implement_date": str(item.get("implement_date", "")),
        "effectiveness": str(item.get("effectiveness", "")),
    }


def _absorb(pool: dict[tuple[str, str, str], Candidate], raw_hits: dict[str, int], route: str, weight: float,
            items: list[dict[str, Any]], k: int) -> None:
    raw_hits[route] += len(items)
    for rank, raw in enumerate(items, start=1):
        item = _normalize(raw)
        # Content prefix keeps a corrupted twin of an article from shadowing the real one.
        key = (item["law_name"], item["article_number"], item["content"][:30])
        candidate = pool.setdefault(key, Candidate(item))
        candidate.rrf += weight / (k + rank)
        candidate.routes.add(route)
        candidate.lists += 1
        if route == "bm25" and isinstance(raw.get("bm25"), (int, float)):
            # SQLite reports BM25 as a negative cost; flip it so larger means better.
            candidate.bm25 = max(candidate.bm25, weight * -float(raw["bm25"]))
        if route == "vector" and isinstance(raw.get("vec"), (int, float)):
            candidate.vec = max(candidate.vec, float(raw["vec"]))


class Recall:
    """Candidate pool shared by the keyword, BM25 and vector routes, fused with weighted RRF."""

    def __init__(self, k: int | None = None) -> None:
        self.k = rrf_k() if k is None else k
        self.pool: dict[tuple[str, str, str], Candidate] = {}
        self.raw_hits = {"keyword": 0, "bm25": 0, "vector": 0}

    def add(self, route: str, weight: float, items: list[dict[str, Any]]) -> None:
        _absorb(self.pool, self.raw_hits, route, weight, items, self.k)

    def add_vector(self, lists: list[list[dict[str, Any]]]) -> None:
        for items in lists:
            self.add("vector", VECTOR_WEIGHT, items)

    def candidates(self) -> list[Candidate]:
        return sorted(self.pool.values(), key=lambda c: -c.rrf)


def recall(
    module: Any, plan: QueryPlan, limit: int = PER_VARIANT_LIMIT, state: Recall | None = None,
    bm25_limit: int | None = None,
) -> tuple[list[Candidate], dict[str, int]]:
    """Run every query variant through the FTS5 BM25 route (plus the optional LIKE route) and fuse with RRF."""
    state = state or Recall()
    bm25_limit = bm25_top_k() if bm25_limit is None else bm25_limit
    for variant in plan.variants:
        if fuzzy_enabled():
            state.add("keyword", variant.weight, _call(module, "law_fuzzy_search", variant.text, limit=limit))
        if variant.kind in _BM25_SKIP_KINDS:
            continue
        phrases = [part for part in variant.text.split() if len(part) >= 3]
        if phrases:
            state.add("bm25", variant.weight, _call(module, "law_bm25_search", phrases, limit=bm25_limit))
    return state.candidates(), state.raw_hits

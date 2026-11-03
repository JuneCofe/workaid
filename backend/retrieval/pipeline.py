from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import vector
from .query import QueryPlan, understand
from .recall import Candidate, Recall, recall
from .rerank import POOL_SIZE, rerank

VECTOR_TIMEOUT_SECONDS = 6
VECTOR_TOP_K = 20
RERANK_POOL = POOL_SIZE


def vector_top_k() -> int:
    return int(os.getenv("VECTOR_TOP_K", VECTOR_TOP_K))


def rerank_pool() -> int:
    return int(os.getenv("RRF_TOP_K", RERANK_POOL))


def _vector_texts(plan: QueryPlan) -> list[str]:
    """The raw sentence plus its formal rewrite; embeddings handle paraphrase, so a few texts suffice."""
    formal = next((v.text for v in plan.variants if v.kind == "formal_sentence"), "")
    concept = next((v.text for v in plan.variants if v.kind == "formal"), "")
    llm = next((v.text for v in plan.variants if v.kind == "llm"), "")
    return [plan.original, formal, concept, llm]


def gather(module: Any, query: str) -> tuple[QueryPlan, list[Candidate], dict[str, Any]]:
    """Rewrite the query, run the two recall routes in parallel, and fuse them with RRF.

    Route A is embedding search in ChromaDB; route B is FTS5 BM25 over the rewritten terms. The embedding
    request is network-bound, so it runs in a worker while the local route runs on this thread.
    """
    started = time.perf_counter()
    plan: QueryPlan = understand(query)
    planned = time.perf_counter()

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(vector.search, module, _vector_texts(plan), vector_top_k())
    state = Recall()
    recall(module, plan, state=state)
    try:
        vec_result = future.result(timeout=VECTOR_TIMEOUT_SECONDS)
    except Exception:
        vec_result = None
    pool.shutdown(wait=False)
    vec_done = time.perf_counter()

    if vec_result is not None:
        state.add_vector(vec_result.lists)
    candidates = state.candidates()
    if vec_result is not None:
        missing = [c.key for c in candidates if c.vec == 0.0]
        scores = vector.score_candidates(vec_result, missing)
        for candidate in candidates:
            candidate.vec = max(candidate.vec, scores.get(candidate.key, 0.0))
    recalled = time.perf_counter()
    trace = {
        **plan.trace(),
        "recalled": state.raw_hits,
        "vector": vec_result is not None,
        "candidates": len(candidates),
        "timings_ms": {
            "rewrite": round((planned - started) * 1000),
            "recall": round((recalled - planned) * 1000),
            "vector_wait": round((vec_done - planned) * 1000),
        },
    }
    return plan, candidates, trace


def search_local(module: Any, query: str, limit: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rewrite -> two-route recall -> RRF -> cross-encoder rerank -> top-n, with a stage trace."""
    plan, candidates, trace = gather(module, query)
    started = time.perf_counter()
    items, rerank_trace = rerank(plan, candidates, limit, module, pool_size=rerank_pool())
    trace["timings_ms"]["rerank"] = round((time.perf_counter() - started) * 1000)
    return items, {**trace, **rerank_trace}

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import vector
from .query import QueryPlan, understand
from .recall import Recall, recall
from .rerank import rerank

VECTOR_TIMEOUT_SECONDS = 6


def _vector_texts(plan: QueryPlan) -> list[str]:
    """The raw sentence plus its formal rewrite; embeddings handle paraphrase, so a few texts suffice."""
    formal = next((v.text for v in plan.variants if v.kind == "formal_sentence"), "")
    concept = next((v.text for v in plan.variants if v.kind == "formal"), "")
    llm = next((v.text for v in plan.variants if v.kind == "llm"), "")
    return [plan.original, formal, concept, llm]


def search_local(module: Any, query: str, limit: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rewrite -> multi-route recall (keyword + BM25 + vector) -> rerank, with a stage trace."""
    started = time.perf_counter()
    plan: QueryPlan = understand(query)
    planned = time.perf_counter()

    # The embedding request is network-bound, so it overlaps with the local keyword routes.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(vector.search, module, _vector_texts(plan))
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
    raw_hits = state.raw_hits
    recalled = time.perf_counter()

    items, rerank_trace = rerank(plan, candidates, limit, module)
    done = time.perf_counter()
    trace = {
        **plan.trace(),
        "recalled": raw_hits,
        "vector": vec_result is not None,
        "candidates": len(candidates),
        **rerank_trace,
        "timings_ms": {
            "rewrite": round((planned - started) * 1000),
            "recall": round((recalled - planned) * 1000),
            "vector_wait": round((vec_done - planned) * 1000),
            "rerank": round((done - recalled) * 1000),
        },
    }
    return items, trace

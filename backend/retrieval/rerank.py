from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from .lexicon import CATEGORY_PRIOR, DEFAULT_PRIOR, NOTICE_STATUTE, PSEUDO_STATUTE
from .quality import classify
from .query import QueryPlan, clean
from .recall import Candidate

POOL_SIZE = 120
API_POOL_SIZE = 40
MAX_PER_LAW = 3
DEFAULT_MIN_SCORE = 0.28
OPEN_MIN_SCORE = 0.6
OPEN_VEC_MIN_SCORE = 0.55
API_MIN_SCORE = 0.2
API_BLEND = 0.15
TRUNCATED_PENALTY = 0.1
# Cosine range mapped onto 0..1; bge-m3 puts related Chinese legal text around 0.5-0.75.
VEC_FLOOR = 0.30
VEC_SPAN = 0.40

_WEIGHTS_WITH_CONCEPTS = {
    "concept": 0.22, "terms": 0.13, "law": 0.12, "rrf": 0.13, "bm25": 0.15, "prior": 0.08, "bigram": 0.17,
}
_WEIGHTS_LLM_TERMS_ONLY = {"terms": 0.35, "rrf": 0.15, "bm25": 0.15, "prior": 0.08, "bigram": 0.27}
# Without a lexicon or model signal the relative rrf/bm25 features always crown some top hit, so
# literal overlap with the sentence has to carry the decision.
_WEIGHTS_OPEN = {"rrf": 0.15, "bm25": 0.15, "prior": 0.10, "bigram": 0.60}

_API_CACHE: OrderedDict[str, dict[int, float]] = OrderedDict()
_API_CACHE_SIZE = 256
_API_DOWN_UNTIL = [0.0]


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^\w]", "", text)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _term_match(term: str, text: str, text_bigrams: set[str]) -> float:
    if term in text:
        return 1.0
    grams = _bigrams(term)
    if not grams:
        return 0.0
    fraction = len(grams & text_bigrams) / len(grams)
    return fraction * fraction * 0.8 if fraction >= 0.6 else 0.0


def _law_match(law_name: str, plan: QueryPlan) -> float:
    best = 0.0
    for concept in plan.concepts:
        for index, law in enumerate(concept.laws):
            if law in law_name:
                best = max(best, 1.0 if index < 2 else 0.7)
    return best


@dataclass
class _Scored:
    candidate: Candidate
    score: float
    base: float


def make_idf(module: Any) -> Callable[[str], float]:
    """Term specificity in 0..1 from corpus document frequency; flat 1.0 when the engine lacks stats."""
    df_of = getattr(module, "law_term_doc_freq", None)
    total_of = getattr(module, "law_article_count", None)
    if df_of is None or total_of is None:
        return lambda term: 1.0
    try:
        total = max(1, int(total_of()))
    except Exception:
        return lambda term: 1.0

    def idf(term: str) -> float:
        try:
            frequency = int(df_of(term))
        except Exception:
            return 1.0
        return math.log(1 + total / (1 + frequency)) / math.log(1 + total)

    return idf


def heuristic_scores(
    plan: QueryPlan, candidates: list[Candidate], idf: Callable[[str], float] | None = None
) -> list[float]:
    if not candidates:
        return []
    max_rrf = max(c.rrf for c in candidates) or 1.0
    max_bm25 = max(c.bm25 for c in candidates) or 1.0
    query_bigrams = _bigrams(clean(plan.original))
    concept_terms = [(*concept.formal, *concept.keywords) for concept in plan.concepts]
    all_terms = list(dict.fromkeys([t for group in concept_terms for t in group] + plan.llm_terms))
    weight_of = {term: idf(term) for term in all_terms} if idf else {term: 1.0 for term in all_terms}
    total_weight = sum(weight_of.values()) or 1.0
    if plan.concepts:
        weights, vec_share = _WEIGHTS_WITH_CONCEPTS, 0.20
    elif all_terms:
        weights, vec_share = _WEIGHTS_LLM_TERMS_ONLY, 0.25
    else:
        weights, vec_share = _WEIGHTS_OPEN, 0.35
    if any(c.vec > 0 for c in candidates):
        # Semantic similarity is a first-class signal once an index exists; the shares are not tuned
        # against real embeddings yet, so re-run eval/run_statute_eval.py after the first index build.
        weights = {**{name: w * (1 - vec_share) for name, w in weights.items()}, "vec": vec_share}

    scores: list[float] = []
    for candidate in candidates:
        item = candidate.item
        text = f"{item['law_name']} {item['content']}"
        text_bigrams = _bigrams(text)
        features = {
            "rrf": candidate.rrf / max_rrf,
            "bm25": candidate.bm25 / max_bm25,
            "vec": min(1.0, max(0.0, (candidate.vec - VEC_FLOOR) / VEC_SPAN)),
            "prior": CATEGORY_PRIOR.get(item["category"], DEFAULT_PRIOR),
            "bigram": min(1.0, 2 * len(query_bigrams & text_bigrams) / len(query_bigrams)) if query_bigrams else 0.0,
        }
        if all_terms:
            matched = {term: _term_match(term, text, text_bigrams) for term in all_terms}
            features["terms"] = min(1.0, 2 * sum(matched[t] * weight_of[t] for t in all_terms) / total_weight)
        if plan.concepts:
            per_concept = []
            for group in concept_terms:
                best = sorted((matched[t] * math.sqrt(weight_of[t]) for t in group), reverse=True)[:2]
                per_concept.append(sum(best) / 2)
            features["concept"] = sum(per_concept) / len(per_concept)
            features["law"] = _law_match(item["law_name"], plan)
        score = sum(weights[name] * features[name] for name in weights) / sum(weights.values())
        if PSEUDO_STATUTE.search(item["law_name"]):
            score -= 0.30
        elif NOTICE_STATUTE.search(item["law_name"]):
            score -= 0.12
        scores.append(max(0.0, min(1.0, score)))
    return scores


def _rerank_endpoint() -> str:
    url = os.getenv("RERANK_API_URL", "").strip()
    parsed = urlparse(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "https" or (parsed.scheme == "http" and local):
        return url
    return ""


def api_enabled() -> bool:
    return bool(_rerank_endpoint() and os.getenv("RERANK_MODEL", "").strip()) and time.monotonic() >= _API_DOWN_UNTIL[0]


def _api_scores(query: str, candidates: list[Candidate]) -> dict[int, float] | None:
    """Cross-encoder relevance from a Cohere/Jina/SiliconFlow-style /rerank endpoint.

    A failure opens a circuit for RERANK_COOLDOWN seconds so an unreachable service costs one timeout,
    not one per question.
    """
    digest = hashlib.sha1(
        (query + "\0" + "\0".join(f"{c.key[0]}{c.key[1]}" for c in candidates)).encode("utf-8")
    ).hexdigest()
    if digest in _API_CACHE:
        _API_CACHE.move_to_end(digest)
        return _API_CACHE[digest]
    documents = [
        f"{c.item['law_name']} {c.item['article_number']} {c.item['content'][:600]}" for c in candidates
    ]
    headers = {"Content-Type": "application/json"}
    if os.getenv("RERANK_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['RERANK_API_KEY']}"
    try:
        response = httpx.post(
            _rerank_endpoint(),
            json={"model": os.getenv("RERANK_MODEL", "").strip(), "query": query,
                  "documents": documents, "top_n": len(documents)},
            headers=headers,
            timeout=float(os.getenv("RERANK_TIMEOUT", "3")),
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results", payload.get("data")) if isinstance(payload, dict) else None
        raw = {
            int(row["index"]): float(row.get("relevance_score", row.get("score")))
            for row in rows or []
            if isinstance(row, dict) and "index" in row
        }
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        raw = {}
    if not raw:
        _API_DOWN_UNTIL[0] = time.monotonic() + float(os.getenv("RERANK_COOLDOWN", "60"))
        return None
    if any(score < 0 or score > 1 for score in raw.values()):
        raw = {index: 1 / (1 + math.exp(-score)) for index, score in raw.items()}
    _API_CACHE[digest] = raw
    if len(_API_CACHE) > _API_CACHE_SIZE:
        _API_CACHE.popitem(last=False)
    return raw


def rerank(
    plan: QueryPlan, candidates: list[Candidate], top_k: int, module: Any = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score the recalled pool, drop low-relevance hits, and keep the top_k with a per-law cap."""
    pool = candidates[:POOL_SIZE]
    base = heuristic_scores(plan, pool, make_idf(module) if module is not None else None)
    scores = list(base)
    reranker = "heuristic"
    if plan.concepts or plan.llm_terms:
        min_score = float(os.getenv("RETRIEVAL_MIN_SCORE", DEFAULT_MIN_SCORE))
    elif any(c.vec > 0 for c in pool):
        # Literal overlap cannot carry a paraphrase, so a strong cosine must be able to clear the gate.
        min_score = float(os.getenv("RETRIEVAL_MIN_SCORE_OPEN_VEC", OPEN_VEC_MIN_SCORE))
    else:
        min_score = float(os.getenv("RETRIEVAL_MIN_SCORE_OPEN", OPEN_MIN_SCORE))
    if pool and api_enabled():
        # Cross-encoder calls are priced per document, so only the heuristic front-runners are sent.
        front = sorted(range(len(pool)), key=lambda i: -base[i])[:API_POOL_SIZE]
        api = _api_scores(plan.original, [pool[i] for i in front])
        if api is None:
            reranker = "heuristic(api_failed)"
        else:
            reranker = "api+heuristic"
            blend = min(1.0, max(0.0, float(os.getenv("RERANK_BLEND", API_BLEND))))
            by_pool = {front[position]: value for position, value in api.items() if position < len(front)}
            scores = [
                blend * by_pool[i] + (1 - blend) * s if i in by_pool else s * (1 - blend)
                for i, s in enumerate(scores)
            ]

    # The relevance gate always uses the heuristic score: a cross-encoder ranks what is there but
    # cannot tell whether anything relevant exists, and a lowered gate lets small talk through.
    ranked = sorted(
        (_Scored(c, s, b) for c, s, b in zip(pool, scores, base)), key=lambda x: -x.score
    )
    selected: list[dict[str, Any]] = []
    per_law: dict[str, int] = {}
    dropped = 0
    damaged = 0
    for entry in ranked:
        item = entry.candidate.item
        defect = classify(item["content"])
        if defect == "fragment":
            damaged += 1
            continue
        penalty = TRUNCATED_PENALTY if defect else 0.0
        if entry.base - penalty < min_score:
            dropped += 1
            continue
        score = entry.score - penalty
        law = item["law_name"]
        if per_law.get(law, 0) >= MAX_PER_LAW:
            continue
        per_law[law] = per_law.get(law, 0) + 1
        selected.append({**item, "score": round(score, 3), **({"quality": defect} if defect else {})})
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda row: -row["score"])
    return selected, {
        "reranker": reranker,
        "pool": len(pool),
        "min_score": min_score,
        "below_threshold": dropped,
        "damaged_articles_skipped": damaged,
        "top_scores": [round(x.score, 3) for x in ranked[:5]],
    }

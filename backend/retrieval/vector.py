from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..config import RAG_CACHE_DIR
from . import embedding
from .lexicon import PSEUDO_STATUTE
from .quality import classify

INDEX_PATH = RAG_CACHE_DIR / "vectors.npy"
META_PATH = RAG_CACHE_DIR / "vectors_meta.json"
# Statutes, administrative regulations and judicial interpretations; the regional corpus is large and
# mostly irrelevant to an individual's case, so it stays keyword-only unless listed here explicitly.
DEFAULT_CATEGORIES = ("法律", "行政法规", "司法解释", "部门规章")
DEFAULT_MIN_COSINE = 0.35
QUERY_CACHE_SIZE = 256

_QUERY_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"mtime": None, "index": None}


def categories() -> tuple[str, ...]:
    raw = os.getenv("VECTOR_CATEGORIES", "")
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or DEFAULT_CATEGORIES


def min_cosine() -> float:
    return float(os.getenv("VECTOR_MIN_COSINE", DEFAULT_MIN_COSINE))


def _core_name(law_name: str) -> str:
    return law_name.removeprefix("中华人民共和国")


def embedding_text(item: dict[str, Any]) -> str:
    """Title and article number travel with the body so a hit is never a headless fragment."""
    return f"{_core_name(item['law_name'])} {item['article_number']} {item['content']}"


def indexable(item: dict[str, Any]) -> bool:
    if "English" in item["law_name"] or PSEUDO_STATUTE.search(item["law_name"]):
        return False
    return classify(item["content"]) != "fragment"


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class VectorIndex:
    matrix: np.ndarray
    rows: list[tuple[str, str, str]]
    model: str

    def __post_init__(self) -> None:
        self.row_of = {(law, article): i for i, (law, article, _) in enumerate(self.rows)}

    def top(self, queries: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        scores = queries @ self.matrix.T
        out: list[list[tuple[int, float]]] = []
        for row in scores:
            k_eff = min(k, row.shape[0])
            part = np.argpartition(-row, k_eff - 1)[:k_eff]
            part = part[np.argsort(-row[part])]
            out.append([(int(i), float(row[i])) for i in part])
        return out

    def similarity(self, queries: np.ndarray, keys: Iterable[tuple[str, str]]) -> dict[tuple[str, str], float]:
        found = {key: self.row_of[key] for key in keys if key in self.row_of}
        if not found:
            return {}
        cols = np.array(list(found.values()))
        best = (queries @ self.matrix[cols].T).max(axis=0)
        return {key: float(score) for key, score in zip(found, best)}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _read_raw() -> tuple[np.ndarray, list[tuple[str, str, str]], dict[str, Any]] | None:
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        matrix = np.load(INDEX_PATH)
        rows = [(a, b, c) for a, b, c in meta["rows"]]
    except (OSError, ValueError, KeyError):
        return None
    return (matrix, rows, meta) if matrix.shape[0] == len(rows) else None


def get_index() -> VectorIndex | None:
    """Load the on-disk index (reloading when rebuilt); None if missing or built with another model."""
    try:
        mtime = (INDEX_PATH.stat().st_mtime_ns, META_PATH.stat().st_mtime_ns)
    except OSError:
        return None
    with _LOCK:
        if _STATE["mtime"] == mtime:
            index = _STATE["index"]
        else:
            raw = _read_raw()
            index = None
            if raw is not None and raw[2].get("model") == embedding.model_name():
                index = VectorIndex(raw[0].astype(np.float32), raw[1], raw[2]["model"])
            _STATE.update(mtime=mtime, index=index)
    return index


def status() -> dict[str, Any]:
    index = get_index()
    meta: dict[str, Any] = {}
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {
        "embedding_configured": embedding.enabled(),
        "model": embedding.model_name(),
        "index_ready": index is not None,
        "articles": len(index.rows) if index else 0,
        "built_at": meta.get("built_at"),
        "index_model": meta.get("model"),
    }


def _save(rows: list[tuple[str, str, str]], matrix: np.ndarray, model: str, cats: tuple[str, ...]) -> None:
    RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_matrix, tmp_meta = INDEX_PATH.with_suffix(".tmp"), META_PATH.with_suffix(".tmp")
    with tmp_matrix.open("wb") as handle:
        np.save(handle, matrix.astype(np.float16))
    tmp_meta.write_text(
        json.dumps(
            {"model": model, "dim": int(matrix.shape[1]) if matrix.size else 0, "categories": list(cats),
             "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": [list(row) for row in rows]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp_matrix.replace(INDEX_PATH)
    tmp_meta.replace(META_PATH)


class BuildError(RuntimeError):
    pass


def plan_build(articles: list[dict[str, Any]]) -> tuple[list[tuple[str, str, str, str]], int]:
    """(law, article, hash, text) for every indexable article, plus how many still need embedding."""
    targets, seen = [], set()
    for item in articles:
        if not indexable(item):
            continue
        key = (item["law_name"], item["article_number"])
        if key in seen:
            continue
        seen.add(key)
        text = embedding_text(item)
        targets.append((*key, _text_hash(text), text))
    raw = _read_raw()
    have = {row[2] for row in raw[1]} if raw and raw[2].get("model") == embedding.model_name() else set()
    return targets, len({t[2] for t in targets if t[2] not in have})


def build_index(
    articles: list[dict[str, Any]],
    *,
    embedder: Callable[[list[str]], list[list[float]] | None] = embedding.embed,
    batch_size: int = 32,
    save_every: int = 25,
    progress: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Embed articles whose text hash is not already indexed; safe to re-run after an interruption."""
    model, cats = embedding.model_name(), categories()
    targets, _ = plan_build(articles)
    raw = _read_raw()
    vectors: dict[str, np.ndarray] = {}
    if raw and raw[2].get("model") == model:
        for (_, _, digest), vector in zip(raw[1], raw[0]):
            vectors[digest] = vector.astype(np.float32)
    reused = len(vectors)

    pending: dict[str, str] = {}
    for _, _, digest, text in targets:
        if digest not in vectors:
            pending.setdefault(digest, text)
    todo = list(pending.items())

    def snapshot() -> None:
        kept = [(law, article, digest) for law, article, digest, _ in targets if digest in vectors]
        if kept:
            _save(kept, _normalize_rows(np.stack([vectors[d] for _, _, d in kept])), model, cats)

    embedded = 0
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        result = embedder([text for _, text in chunk])
        if result is None or len(result) != len(chunk):
            snapshot()
            raise BuildError(f"embedding request failed after {embedded} new vectors; re-run to resume")
        for (digest, _), vector in zip(chunk, result):
            vectors[digest] = np.asarray(vector, dtype=np.float32)
        embedded += len(chunk)
        progress(f"embedded {embedded}/{len(todo)}")
        if (start // batch_size + 1) % save_every == 0:
            snapshot()
    snapshot()
    return {"articles": len([t for t in targets if t[2] in vectors]), "embedded": embedded, "reused": reused, "model": model}


@dataclass
class VectorResult:
    lists: list[list[dict[str, Any]]]
    queries: np.ndarray


def _embed_queries(texts: list[str]) -> np.ndarray | None:
    prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    missing = [t for t in dict.fromkeys(texts) if t not in _QUERY_CACHE]
    if missing:
        fresh = embedding.embed(
            [prefix + t for t in missing],
            timeout=float(os.getenv("EMBEDDING_QUERY_TIMEOUT", "4")),
            retries=1,
        )
        if fresh is None:
            return None
        for text, vector in zip(missing, fresh):
            _QUERY_CACHE[text] = vector
            if len(_QUERY_CACHE) > QUERY_CACHE_SIZE:
                _QUERY_CACHE.popitem(last=False)
    matrix = np.array([_QUERY_CACHE[t] for t in texts], dtype=np.float32)
    return _normalize_rows(matrix)


def embed_query_texts(texts: list[str]) -> np.ndarray | None:
    """Unit-normalised embeddings for query texts (cached); None when the service is unavailable."""
    return _embed_queries(texts)


def search(module: Any, texts: list[str], k: int = 20) -> VectorResult | None:
    """Nearest articles per query text, hydrated to full articles; None when vectors are unavailable."""
    index = get_index()
    texts = [t for t in dict.fromkeys(t.strip() for t in texts) if t]
    if index is None or not texts or not embedding.enabled():
        return None
    queries = _embed_queries(texts)
    if queries is None:
        return None
    floor = min_cosine()
    per_query = [[(i, s) for i, s in hits if s >= floor] for hits in index.top(queries, k)]
    wanted = list(dict.fromkeys((index.rows[i][0], index.rows[i][1]) for hits in per_query for i, _ in hits))
    if not wanted:
        return VectorResult([[] for _ in texts], queries)
    try:
        data = json.loads(module.law_fetch_articles(wanted)).get("data") or []
    except Exception:
        return None
    by_key = {(row["law_name"], row["article_number"]): row for row in data}
    lists = [
        [
            {**by_key[index.rows[i][:2]], "vec": score}
            for i, score in hits
            if index.rows[i][:2] in by_key
        ]
        for hits in per_query
    ]
    return VectorResult(lists, queries)


def score_candidates(result: VectorResult, keys: Iterable[tuple[str, str]]) -> dict[tuple[str, str], float]:
    index = get_index()
    return index.similarity(result.queries, keys) if index is not None else {}

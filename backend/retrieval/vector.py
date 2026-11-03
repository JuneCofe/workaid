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

CHROMA_DIR = RAG_CACHE_DIR / "chroma"
COLLECTION = "statutes"
# The numpy index written before ChromaDB; imported once so the corpus is not embedded a second time.
LEGACY_INDEX_PATH = RAG_CACHE_DIR / "vectors.npy"
LEGACY_META_PATH = RAG_CACHE_DIR / "vectors_meta.json"
# Statutes, administrative regulations and judicial interpretations; the regional corpus is large and
# mostly irrelevant to an individual's case, so it stays keyword-only unless listed here explicitly.
DEFAULT_CATEGORIES = ("法律", "行政法规", "司法解释", "部门规章")
DEFAULT_MIN_COSINE = 0.35
QUERY_CACHE_SIZE = 256
WRITE_BATCH = 500

_QUERY_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_LOCK = threading.Lock()
_CLIENTS: dict[str, Any] = {}


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


def _record_id(law: str, article: str) -> str:
    return f"{law}\x1f{article}"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _client() -> Any:
    import chromadb
    from chromadb.config import Settings

    path = str(CHROMA_DIR)
    with _LOCK:
        if path not in _CLIENTS:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _CLIENTS[path] = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        return _CLIENTS[path]


def _open(create: bool = False) -> Any | None:
    """The statute collection, or None when nothing has been built yet.

    A collection built with another embedding model is never queried: its vectors live in a different space.
    With create=True it is replaced when the model changed, and errors propagate so a failed build is visible.
    """
    model = embedding.model_name()
    metadata = {"hnsw:space": "cosine", "model": model}
    if create:
        client = _client()
        collection = client.get_or_create_collection(COLLECTION, metadata=metadata)
        if (collection.metadata or {}).get("model") != model:
            client.delete_collection(COLLECTION)
            collection = client.create_collection(COLLECTION, metadata=metadata)
        return collection
    try:
        return _client().get_collection(COLLECTION)
    except Exception:
        return None


def _usable(collection: Any | None) -> bool:
    return (
        collection is not None
        and (collection.metadata or {}).get("model") == embedding.model_name()
        and collection.count() > 0
    )


def _stored(collection: Any) -> dict[str, str]:
    """record id -> text hash for everything already in the collection, read in pages."""
    found: dict[str, str] = {}
    total = collection.count()
    for offset in range(0, total, WRITE_BATCH):
        rows = collection.get(include=["metadatas"], limit=WRITE_BATCH, offset=offset)
        found.update({record_id: meta["hash"] for record_id, meta in zip(rows["ids"], rows["metadatas"])})
    return found


def _upsert(collection: Any, rows: list[tuple[str, str, str]], vectors: np.ndarray) -> None:
    for start in range(0, len(rows), WRITE_BATCH):
        part = rows[start : start + WRITE_BATCH]
        collection.upsert(
            ids=[_record_id(law, article) for law, article, _ in part],
            embeddings=vectors[start : start + WRITE_BATCH].tolist(),
            metadatas=[{"law_name": law, "article_number": article, "hash": digest} for law, article, digest in part],
        )


def status() -> dict[str, Any]:
    collection = _open()
    meta = (collection.metadata or {}) if collection is not None else {}
    return {
        "embedding_configured": embedding.enabled(),
        "model": embedding.model_name(),
        "index_ready": _usable(collection),
        "articles": collection.count() if collection is not None else 0,
        "built_at": meta.get("built_at"),
        "index_model": meta.get("model"),
        "store": "chromadb",
    }


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
    collection = _open()
    have = set(_stored(collection).values()) if _usable(collection) else set()
    return targets, len({t[2] for t in targets if t[2] not in have})


def _import_legacy(collection: Any, targets: list[tuple[str, str, str, str]], stored: dict[str, str]) -> int:
    """Move vectors from the pre-ChromaDB numpy index when they still match the current text hashes."""
    if not (LEGACY_INDEX_PATH.exists() and LEGACY_META_PATH.exists()):
        return 0
    try:
        meta = json.loads(LEGACY_META_PATH.read_text(encoding="utf-8"))
        matrix = np.load(LEGACY_INDEX_PATH).astype(np.float32)
    except (OSError, ValueError):
        return 0
    if meta.get("model") != embedding.model_name() or matrix.shape[0] != len(meta["rows"]):
        return 0
    wanted = {(law, article, digest) for law, article, digest, _ in targets}
    keep = [
        (index, (law, article, digest))
        for index, (law, article, digest) in enumerate(meta["rows"])
        if (law, article, digest) in wanted and stored.get(_record_id(law, article)) != digest
    ]
    if keep:
        rows = [row for _, row in keep]
        _upsert(collection, rows, _normalize_rows(matrix[[index for index, _ in keep]]))
    return len(keep)


def build_index(
    articles: list[dict[str, Any]],
    *,
    embedder: Callable[[list[str]], list[list[float]] | None] = embedding.embed,
    batch_size: int = 32,
    progress: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Embed articles whose text hash is not stored yet; each batch is written before the next starts."""
    model = embedding.model_name()
    targets, _ = plan_build(articles)
    collection = _open(create=True)
    stored = _stored(collection)
    wanted = {_record_id(law, article) for law, article, _, _ in targets}
    gone = [record_id for record_id in stored if record_id not in wanted]
    for start in range(0, len(gone), WRITE_BATCH):
        collection.delete(ids=gone[start : start + WRITE_BATCH])

    imported = _import_legacy(collection, targets, stored)
    if imported:
        stored = _stored(collection)
    have = set(stored.values())
    reused = len({digest for _, _, digest, _ in targets if digest in have})
    pending = [
        (law, article, digest, text)
        for law, article, digest, text in targets
        if stored.get(_record_id(law, article)) != digest
    ]

    embedded = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        result = embedder([text for *_, text in chunk])
        if result is None or len(result) != len(chunk):
            raise BuildError(f"embedding request failed after {embedded} new vectors; re-run to resume")
        vectors = _normalize_rows(np.asarray(result, dtype=np.float32))
        _upsert(collection, [row[:3] for row in chunk], vectors)
        embedded += len(chunk)
        progress(f"embedded {embedded}/{len(pending)}")

    collection.modify(metadata={"model": model, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {
        "articles": collection.count(),
        "embedded": embedded,
        "imported": imported,
        "reused": reused,
        "model": model,
    }


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
    """Nearest articles per query text (cosine, ChromaDB HNSW), hydrated to full articles; None when unavailable."""
    texts = [t for t in dict.fromkeys(t.strip() for t in texts) if t]
    if not texts or not embedding.enabled():
        return None
    collection = _open()
    if not _usable(collection):
        return None
    queries = _embed_queries(texts)
    if queries is None:
        return None
    floor = min_cosine()
    found = collection.query(
        query_embeddings=queries.tolist(), n_results=min(k, collection.count()), include=["metadatas", "distances"]
    )
    per_query = [
        [(meta["law_name"], meta["article_number"], 1.0 - dist) for meta, dist in zip(metas, dists) if 1.0 - dist >= floor]
        for metas, dists in zip(found["metadatas"], found["distances"])
    ]
    wanted = list(dict.fromkeys((law, article) for hits in per_query for law, article, _ in hits))
    if not wanted:
        return VectorResult([[] for _ in texts], queries)
    try:
        data = json.loads(module.law_fetch_articles(wanted)).get("data") or []
    except Exception:
        return None
    by_key = {(row["law_name"], row["article_number"]): row for row in data}
    lists = [
        [{**by_key[(law, article)], "vec": score} for law, article, score in hits if (law, article) in by_key]
        for hits in per_query
    ]
    return VectorResult(lists, queries)


def score_candidates(result: VectorResult, keys: Iterable[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Best cosine between each article and any of the query vectors, for articles found by other routes."""
    collection = _open()
    keys = list(dict.fromkeys(keys))
    if not _usable(collection) or not keys:
        return {}
    ids = [_record_id(law, article) for law, article in keys]
    rows = collection.get(ids=ids, include=["embeddings"])
    stored = {record_id: vector for record_id, vector in zip(rows["ids"], rows["embeddings"])}
    scores: dict[tuple[str, str], float] = {}
    for key, record_id in zip(keys, ids):
        if record_id in stored:
            scores[key] = float(np.max(result.queries @ np.asarray(stored[record_id], dtype=np.float32)))
    return scores

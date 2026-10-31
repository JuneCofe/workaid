"""Local case catalog: JSONL records, lexical + concept + citation + vector retrieval."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..retrieval import embedding, vector
from ..retrieval.query import QueryPlan, clean, understand
from .schema import CITATION, PROCEDURAL_CONCEPTS, REVIEWED_LABEL, UNREVIEWED_LABEL, normalize_record

FIELD_WEIGHTS = {"title": 3, "cause": 3, "keywords": 3, "dispute_focus": 2, "holding": 2, "summary": 1,
                 "significance": 1, "result": 1}
# Measured on eval/case_queries.jsonl: correct cases scored 0.55-0.82, cases on an unrelated point 0.27-0.40.
DEFAULT_MIN_SCORE = 0.45
CITE_BONUS = 0.30
VEC_FLOOR, VEC_SPAN = 0.30, 0.40
K1, B = 1.5, 0.75
_TEXT_KINDS = {"formal_sentence", "formal", "keywords", "law_hint", "llm", "original"}

_LOCK = threading.Lock()
_CATALOGS: dict[Path, "Catalog"] = {}


def _bigrams(text: str) -> list[str]:
    compact = re.sub(r"[^\w]", "", text.lower())
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


@dataclass
class Doc:
    record: dict[str, Any]
    tf: Counter
    length: int
    plain: str
    cited: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class Catalog:
    docs: list[Doc]
    df: Counter
    avgdl: float
    mtime: int


def _core(law: str) -> str:
    return re.sub(r"[（(][^)）]*[)）]", "", law).strip().removeprefix("中华人民共和国")


def _build_doc(record: dict[str, Any]) -> Doc:
    tf: Counter = Counter()
    for name, weight in FIELD_WEIGHTS.items():
        value = record.get(name, "")
        text = " ".join(value) if isinstance(value, list) else str(value)
        for token in _bigrams(text):
            tf[token] += weight
    for token in _bigrams(" ".join(record.get("keywords", []))):
        tf[token] += FIELD_WEIGHTS["keywords"]
    plain = re.sub(r"[^\w]", "", " ".join(str(record.get(n, "")) for n in FIELD_WEIGHTS).lower())
    cited = set()
    for item in record.get("statutes", []):
        law, _, article = str(item).partition("|")
        cited.add((_core(law), article))
    return Doc(record, tf, sum(tf.values()), plain, cited)


def load(path: Path) -> Catalog:
    """Parse the catalog (tolerantly) and cache it until the file changes."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return Catalog([], Counter(), 1.0, 0)
    with _LOCK:
        cached = _CATALOGS.get(path)
        if cached and cached.mtime == mtime:
            return cached
        docs: list[Doc] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                docs.append(_build_doc(normalize_record(raw)))
        df: Counter = Counter()
        for doc in docs:
            df.update(doc.tf.keys())
        avgdl = (sum(d.length for d in docs) / len(docs)) if docs else 1.0
        catalog = Catalog(docs, df, avgdl or 1.0, mtime)
        _CATALOGS[path] = catalog
        return catalog


def records(path: Path) -> list[dict[str, Any]]:
    return [doc.record for doc in load(path).docs]


def stats(path: Path) -> dict[str, Any]:
    recs = records(path)
    return {
        "cases": len(recs),
        "reviewed": sum(1 for r in recs if r.get("reviewed")),
        "with_statutes": sum(1 for r in recs if r.get("statutes")),
        "vectors": vector_status(path),
    }


# ---------------------------------------------------------------- vectors

def _vector_paths(path: Path) -> tuple[Path, Path]:
    return Path(f"{path}.vectors.npy"), Path(f"{path}.vectors.json")


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embedding_text(record: dict[str, Any]) -> str:
    parts = [record.get("title", ""), record.get("cause", ""), record.get("dispute_focus", ""),
             record.get("holding", ""), record.get("summary", "")[:700]]
    return "\n".join(part for part in parts if part)


def _read_vectors(path: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    npy, meta_path = _vector_paths(path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        matrix = np.load(npy)
    except (OSError, ValueError):
        return None
    if meta.get("model") != embedding.model_name() or matrix.shape[0] != len(meta.get("rows", [])):
        return None
    return matrix.astype(np.float32), meta


def vector_status(path: Path) -> dict[str, Any]:
    loaded = _read_vectors(path)
    return {"ready": loaded is not None, "cases": len(loaded[1]["rows"]) if loaded else 0,
            "model": embedding.model_name()}


def build_vectors(
    path: Path,
    embedder: Callable[[list[str]], list[list[float]] | None] = embedding.embed,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Embed cases whose text changed since the last build; raises RuntimeError if a batch fails."""
    recs = records(path)
    targets = [(r["id"], embedding_text(r)) for r in recs]
    have: dict[str, np.ndarray] = {}
    loaded = _read_vectors(path)
    if loaded:
        for (case_key, digest), vec in zip(loaded[1]["rows"], loaded[0]):
            have[digest] = vec
    pending = [(cid, text) for cid, text in targets if _digest(text) not in have]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        result = embedder([text for _, text in chunk])
        if result is None or len(result) != len(chunk):
            raise RuntimeError(f"embedding failed after {start} new cases; re-run to resume")
        for (_, text), vec in zip(chunk, result):
            have[_digest(text)] = np.asarray(vec, dtype=np.float32)
    rows = [[cid, _digest(text)] for cid, text in targets]
    if rows:
        matrix = np.stack([have[d] for _, d in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = (matrix / np.where(norms == 0, 1, norms)).astype(np.float16)
        npy, meta_path = _vector_paths(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(f"{npy}.tmp").open("wb") as handle:
            np.save(handle, matrix)
        Path(f"{npy}.tmp").replace(npy)
        meta_path.write_text(json.dumps({"model": embedding.model_name(), "rows": rows,
                                         "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False),
                             encoding="utf-8")
    return {"cases": len(rows), "embedded": len(pending), "reused": len(rows) - len(pending)}


def _vector_scores(path: Path, texts: list[str]) -> dict[str, float] | None:
    if not embedding.enabled():
        return None
    loaded = _read_vectors(path)
    if loaded is None:
        return None
    queries = vector.embed_query_texts(texts)
    if queries is None:
        return None
    matrix, meta = loaded
    best = (queries @ matrix.T).max(axis=0)
    return {row[0]: float(score) for row, score in zip(meta["rows"], best)}


# ----------------------------------------------------------------- search

def _query_tokens(plan: QueryPlan) -> list[str]:
    texts = [clean(plan.original)] + [v.text for v in plan.variants if v.kind in _TEXT_KINDS]
    seen: dict[str, None] = {}
    for text in texts:
        for token in _bigrams(text):
            seen.setdefault(token, None)
    return list(seen)


def _idf(catalog: Catalog, token: str) -> float:
    n = len(catalog.docs)
    return math.log(1 + (n - catalog.df.get(token, 0) + 0.5) / (catalog.df.get(token, 0) + 0.5))


def min_score() -> float:
    return float(os.getenv("CASE_MIN_SCORE", DEFAULT_MIN_SCORE))


def to_result(record: dict[str, Any], score: float) -> dict[str, Any]:
    """Shape a catalog record for the agent; `source_name` is deliberately absent so the graph
    does not mistake a local hit for a remote fallback."""
    snippet = record["holding"] or record["summary"]
    return {
        "title": record["title"],
        "summary": snippet[:360],
        "content": "\n".join(part for part in (record["summary"], record["holding"], record["result"]) if part),
        "url": record["source_url"],
        "case_number": record["case_number"],
        "court": record["court"],
        "cause": record["cause"],
        "decision_date": record["decision_date"],
        "issuer": record["issuer"],
        "source_type": record["source_type"],
        "publish_date": record["publish_date"],
        "holding": record["holding"],
        "result": record["result"],
        "statutes": record["statutes"],
        "review_status": REVIEWED_LABEL if record["reviewed"] else UNREVIEWED_LABEL,
        "provenance_note": record["provenance_note"],
        "source": "local_casebase",
        "source_id": record["id"],
        "score": round(score, 3),
    }


def search(
    query: str,
    limit: int = 5,
    *,
    path: Path,
    use_vectors: bool = True,
    only_reviewed: bool | None = None,
) -> list[dict[str, Any]]:
    catalog = load(path)
    if not catalog.docs or not query.strip():
        return []
    only_reviewed = os.getenv("CASE_ONLY_REVIEWED", "0") == "1" if only_reviewed is None else only_reviewed
    plan = understand(query)
    tokens = _query_tokens(plan)
    if not tokens:
        return []
    idf = {token: _idf(catalog, token) for token in tokens}
    total_idf = sum(idf.values()) or 1.0
    concepts = {c.key for c in plan.concepts} - PROCEDURAL_CONCEPTS
    cites = {(_core(law), article) for law, article in CITATION.findall(query)}

    vec = None
    if use_vectors:
        vec = _vector_scores(path, [plan.original] + [v.text for v in plan.variants if v.kind == "formal_sentence"])

    bm25: dict[int, float] = {}
    coverage: dict[int, float] = {}
    for index, doc in enumerate(catalog.docs):
        score = matched = 0.0
        for token in tokens:
            tf = doc.tf.get(token, 0)
            if not tf:
                continue
            score += idf[token] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc.length / catalog.avgdl))
            matched += idf[token]
        bm25[index], coverage[index] = score, matched / total_idf
    best = max(bm25.values()) or 1.0

    weights = (
        {"lex": 0.25, "bm25": 0.10, "concept": 0.25, "vec": 0.30, "cite": 0.10}
        if vec is not None
        else {"lex": 0.35, "bm25": 0.15, "concept": 0.40, "cite": 0.10}
    )
    ranked: list[tuple[float, Doc]] = []
    for index, doc in enumerate(catalog.docs):
        record = doc.record
        if only_reviewed and not record["reviewed"]:
            continue
        overlap = concepts & set(record["scenarios"])
        features = {
            "lex": min(1.0, coverage[index] * 2.5),
            "bm25": bm25[index] / best,
            "concept": len(overlap) / len(concepts) if concepts else 0.0,
            "cite": 1.0 if cites & doc.cited else 0.0,
        }
        if vec is not None:
            features["vec"] = min(1.0, max(0.0, (vec.get(record["id"], 0.0) - VEC_FLOOR) / VEC_SPAN))
        score = sum(weights[name] * features[name] for name in weights)
        if features["cite"]:
            # A case that applies the very article the user cites is relevant even if the wording differs.
            score += CITE_BONUS
        has_signal = features["concept"] > 0 or features["cite"] > 0 or features.get("vec", 0) > 0.4 or coverage[index] >= 0.2
        if has_signal and score >= min_score():
            ranked.append((score, doc))
    ranked.sort(key=lambda pair: -pair[0])
    return [to_result(doc.record, score) for score, doc in ranked[:limit]]

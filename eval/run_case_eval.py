"""Score local case retrieval against eval/case_queries.jsonl.

    python -I eval/run_case_eval.py [-v] [--no-vector]

Each row lists substrings of case titles; a result counts as a hit if its title contains any of
them. `expect: []` means nothing should be returned (except titles matching `tolerate`). Labels written from the catalog titles are a
seed; replace them with lawyer-made judgements of which cases really answer the question.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["RETRIEVAL_LLM_REWRITE"] = "0"

from backend import casebase  # noqa: E402
from backend.config import CASE_CATALOG_PATH  # noqa: E402

PATH = Path(__file__).with_name("case_queries.jsonl")


def run(use_vectors: bool = True) -> dict:
    rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = {1: 0, 3: 0, 5: 0}
    positives = abstain_ok = negatives = 0
    reciprocal = 0.0
    detail = []
    for row in rows:
        results = casebase.search(row["query"], 5, path=CASE_CATALOG_PATH, use_vectors=use_vectors)
        titles = [r["title"] for r in results]
        if not row["expect"]:
            negatives += 1
            # `tolerate`: titles of cases on a neighbouring point that a reader would not call wrong.
            wrong = [t for t in titles if not any(w in t for w in row.get("tolerate", []))]
            abstain_ok += 0 if wrong else 1
            detail.append((row["id"], "abstain" if not wrong else "FALSE-HIT", row["query"], wrong[:2]))
            continue
        positives += 1
        rank = next((i for i, t in enumerate(titles, 1) if any(e in t for e in row["expect"])), None)
        for k in hits:
            hits[k] += 1 if rank and rank <= k else 0
        reciprocal += 1 / rank if rank else 0
        detail.append((row["id"], f"hit@{rank}" if rank else "MISS", row["query"], titles[:3]))
    return {"positives": positives, "hits": hits, "mrr": reciprocal / max(1, positives), "negatives": negatives,
            "abstain_ok": abstain_ok, "detail": detail}


if __name__ == "__main__":
    result = run(use_vectors="--no-vector" not in sys.argv)
    p = result["positives"]
    print(f"相关问题 {p} 条：hit@1 {result['hits'][1] / p:.2f}  hit@3 {result['hits'][3] / p:.2f}  hit@5 {result['hits'][5] / p:.2f}  MRR {result['mrr']:.2f}")
    print(f"无关问题 {result['negatives']} 条：正确不返回 {result['abstain_ok']}/{result['negatives']}")
    for row_id, status, query, titles in result["detail"]:
        if "-v" in sys.argv or status in {"MISS", "FALSE-HIT"}:
            print(f"  [{status}] {row_id} {query[:30]} -> {[t[:20] for t in titles]}")

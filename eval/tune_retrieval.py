"""Sweep the two-route recall and rerank sizes on the labelled statute queries.

Usage: python eval/tune_retrieval.py [--grid default|quick]

The dev set picks the setting (ties go to the cheaper one: smaller k, smaller pool). The holdout set is only
reported, never used to choose. Runs with .env, so the embedding route and the bge reranker are both live.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["RETRIEVAL_LLM_REWRITE"] = "0"

from backend.retrieval.evaluation import evaluate, load_cases, pipeline_search  # noqa: E402
from backend.rag import _load_engine  # noqa: E402

DEV = ROOT / "eval" / "statute_queries.jsonl"
HOLDOUT = ROOT / "eval" / "statute_holdout.jsonl"

GRIDS = {
    "default": {
        "VECTOR_TOP_K": [10, 20],
        "BM25_TOP_K": [10, 20, 40],
        "RRF_TOP_K": [30, 60, 120],
    },
    "quick": {
        "VECTOR_TOP_K": [20],
        "BM25_TOP_K": [20],
        "RRF_TOP_K": [30],
    },
}


def run(config: dict[str, int], dev, holdout) -> dict:
    os.environ.update({key: str(value) for key, value in config.items()})
    dev_report = evaluate("dev", pipeline_search(), dev)
    holdout_report = evaluate("holdout", pipeline_search(), holdout)
    return {
        **config,
        "dev": dev_report.summary(),
        "holdout": holdout_report.summary(),
    }


def main() -> None:
    grid = GRIDS[sys.argv[sys.argv.index("--grid") + 1]] if "--grid" in sys.argv else GRIDS["default"]
    dev, holdout = load_cases(DEV), load_cases(HOLDOUT)
    _load_engine()
    rows = []
    for values in itertools.product(*grid.values()):
        config = dict(zip(grid.keys(), values))
        row = run(config, dev, holdout)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    best = max(rows, key=lambda r: (r["dev"]["hit@5"], r["dev"]["mrr"], -r["VECTOR_TOP_K"] - r["BM25_TOP_K"] - r["RRF_TOP_K"]))
    print("\nselected on dev:", json.dumps({k: best[k] for k in grid}, ensure_ascii=False))
    print("dev:", best["dev"])
    print("holdout (reported only):", best["holdout"])
    out = ROOT / "eval" / ".tmp" / "tune_retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

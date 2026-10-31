"""Compare raw-sentence search against the rewrite/recall/rerank pipeline on the labelled queries.

Usage: python -I eval/run_statute_eval.py [-v] [--holdout] [--no-vector]

Reads .env, so the vector route is used when EMBEDDING_API_KEY is set and an index exists;
--no-vector turns it off for comparison. LLM query rewriting is always off here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if "--no-vector" in sys.argv[1:]:
    os.environ.pop("EMBEDDING_API_KEY", None)

from backend.retrieval import embedding, vector  # noqa: E402
from backend.retrieval.evaluation import load_cases, run_comparison  # noqa: E402

HOLDOUT = Path(__file__).with_name("statute_holdout.jsonl")


def main() -> None:
    verbose = "-v" in sys.argv[1:]
    holdout = "--holdout" in sys.argv[1:]
    print(f"vector route: {'on' if embedding.enabled() and vector.get_index() else 'off'}")
    baseline, pipeline = run_comparison(load_cases(HOLDOUT) if holdout else None)
    for report in (baseline, pipeline):
        print(f"{report.name}: {report.summary()}")
    if verbose:
        for result in pipeline.results:
            status = f"hit@{result.first_hit}" if result.first_hit else ("abstain" if not result.case["expect"] and not result.ranked else "MISS")
            print(f"\n[{status}] {result.case['id']}: {result.case['query']}")
            for law, article in result.ranked:
                print(f"    {law[:24]} {article}")


if __name__ == "__main__":
    main()

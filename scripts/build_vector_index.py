"""Build or refresh the vector index over the statute corpus.

    python -I scripts/build_vector_index.py --dry-run     # count articles and estimate tokens, no API calls
    python -I scripts/build_vector_index.py               # embed what is missing (resumable)

Needs EMBEDDING_API_KEY in .env (see .env.example). Re-running only embeds articles whose text changed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.rag import _load_engine  # noqa: E402
from backend.retrieval import embedding, vector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--categories", default="", help="comma separated corpus categories (default: 法律,行政法规,司法解释)")
    parser.add_argument("--batch", type=int, default=int(os.getenv("EMBEDDING_BATCH", "32")))
    args = parser.parse_args()
    if args.categories:
        os.environ["VECTOR_CATEGORIES"] = args.categories

    module = _load_engine()
    if module is None:
        print("local statute engine is unavailable")
        return 1
    cats = vector.categories()
    articles = module.law_iter_articles(cats)
    targets, to_embed = vector.plan_build(articles)
    chars = sum(len(text) for _, _, digest, text in targets)
    print(f"categories: {', '.join(cats)}")
    print(f"articles in corpus: {len(articles)}; indexable: {len(targets)}; still to embed: {to_embed}")
    print(f"embedding text: {chars:,} characters (roughly {int(chars * 0.7):,} tokens for the whole set)")
    print(f"model: {embedding.model_name()}  endpoint: {embedding.endpoint() or '(invalid)'}")
    if args.dry_run:
        return 0
    if not embedding.enabled():
        print("EMBEDDING_API_KEY is not set; add it to .env and re-run.")
        return 2
    try:
        result = vector.build_index(articles, batch_size=args.batch, progress=lambda m: print(m, end="\r", flush=True))
    except vector.BuildError as exc:
        print(f"\n{exc}")
        return 3
    print(f"\ndone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

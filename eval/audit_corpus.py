"""Audit the statute corpus for damaged or missing articles.

Reads the built SQLite index read-only and reports, per law, how many article bodies are fragments
or truncated (see backend/retrieval/quality.py) and which article numbers are missing.

Usage: python -I eval/audit_corpus.py [--top 25] [--law 劳动合同法]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import RAG_CACHE_DIR  # noqa: E402
from backend.retrieval.quality import classify  # noqa: E402


def audit(db: sqlite3.Connection, law_filter: str = "") -> list[dict]:
    # articles has no law_id index, so scan it once instead of once per law.
    laws = {
        law_id: (name, category)
        for law_id, name, category in db.execute(
            "SELECT id, law_name, category FROM laws WHERE law_name LIKE ?", (f"%{law_filter}%",)
        )
    }
    grouped: dict[int, list[tuple[str, str]]] = {law_id: [] for law_id in laws}
    for law_id, number, content in db.execute("SELECT law_id, normalized_article, content FROM articles"):
        if law_id in grouped:
            grouped[law_id].append((number, content))

    rows = []
    for law_id, (name, category) in laws.items():
        articles = grouped[law_id]
        numbers = {int(n.split("-")[0]) for n, _ in articles if n.split("-")[0].isdigit()}
        top = max(numbers, default=0)
        defects = [classify(content) for _, content in articles]
        if "English" in name:
            # Translated texts do not end in Chinese punctuation; the truncation check is meaningless.
            defects = ["fragment" if d == "fragment" else "" for d in defects]
        # Interpretations cite code articles (e.g. 第1211条) that get split into phantom numbers,
        # so gap counting is only meaningful for statutes and administrative regulations.
        gaps = top - len(numbers & set(range(1, top + 1))) if category in {"法律", "行政法规"} else None
        rows.append(
            {
                "law": name,
                "category": category,
                "articles": len(articles),
                "fragments": defects.count("fragment"),
                "truncated": defects.count("truncated"),
                "missing": gaps,
                "max_article": top,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--law", default="")
    args = parser.parse_args()

    db = sqlite3.connect(f"file:{RAG_CACHE_DIR / 'law.db'}?mode=ro", uri=True)
    rows = audit(db, args.law)
    total = sum(r["articles"] for r in rows)
    fragments = sum(r["fragments"] for r in rows)
    truncated = sum(r["truncated"] for r in rows)
    print(f"{len(rows)} laws, {total} articles: {fragments} fragments, {truncated} truncated")
    national = [r for r in rows if r["category"] in {"法律", "行政法规", "司法解释"}]
    statutes = [r for r in rows if r["missing"] is not None]
    print(
        f"national laws/regulations/interpretations: {len(national)} laws, "
        f"{sum(r['fragments'] for r in national)} fragments, {sum(r['truncated'] for r in national)} truncated"
    )
    print(
        f"statutes + administrative regulations: {len(statutes)} laws, "
        f"{sum(r['missing'] for r in statutes)} article numbers missing below each law's highest number"
    )
    print(f"\nWorst {args.top} statutes/regulations by damaged+missing articles:")
    worst = sorted(statutes, key=lambda r: -(r["fragments"] + r["truncated"] + r["missing"]))[: args.top]
    for r in worst:
        print(
            f"  {r['law'][:34]:<34} [{r['category']}] articles={r['articles']:>4} "
            f"fragments={r['fragments']:>3} truncated={r['truncated']:>3} missing={r['missing']:>3}"
        )


if __name__ == "__main__":
    main()

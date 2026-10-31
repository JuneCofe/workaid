"""Extract the cleanly structured labour-dispute roundup already in the corpus into an import file.

    python -I scripts/seed_cases_from_corpus.py

Writes data/case_import/seed_xuzhou_2026.jsonl (not the catalog). Review it, then run
scripts/import_cases.py. The corpus copy was republished by a commercial database, so the records
are marked unreviewed and carry no official URL until you add one.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_TITLE = "徐州中院、徐州市人社局联合发布劳动争议典型案例"
OUT = ROOT / "data" / "case_import" / "seed_xuzhou_2026.jsonl"
SECTION = re.compile(r"【(简要案情|典型意义)】")


def find_corpus_file() -> Path:
    for path in (ROOT / "rag" / "data").rglob("*.json"):
        if path.name == "index.json":
            continue
        with path.open(encoding="utf-8") as handle:
            if f'"law_name": "{CORPUS_TITLE}"' in handle.read(400):
                return path
    raise SystemExit(f"{CORPUS_TITLE} not found in rag/data")


def split_case(body: str) -> dict[str, str] | None:
    body = body.lstrip("：: ").strip()
    title, _, rest = body.partition("\n")
    parts = SECTION.split(rest)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not title.strip() or "简要案情" not in sections:
        return None
    paragraphs = [p.strip() for p in sections["简要案情"].split("\n") if p.strip()]
    split_at = next((i for i, p in enumerate(paragraphs) if "认为" in p[:40]), None)
    facts = paragraphs if split_at is None else paragraphs[:split_at]
    holding = [] if split_at is None else paragraphs[split_at:]
    return {
        "title": title.strip(),
        "summary": "\n".join(facts),
        "holding": "\n".join(holding),
        "significance": sections.get("典型意义", ""),
    }


def main() -> int:
    source = find_corpus_file()
    document = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for label, body in document["articles"].items():
        parsed = split_case(body)
        if parsed is None:
            print(f"skipped {label}: not in the expected structure")
            continue
        records.append({
            **parsed,
            "case_type": "劳动争议",
            "source_type": "中基层人民法院典型案例",
            "issuer": "徐州市中级人民法院、徐州市人力资源和社会保障局",
            "region": "江苏省徐州市",
            "publish_date": document.get("publish_date", ""),
            "source_url": "",
            "provenance_note": f"摘自 WorkAid 语料库《{CORPUS_TITLE}》（{label}），语料库转载自商业数据库；官方原文网址待补充并核验。",
            "reviewed": False,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} cases to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

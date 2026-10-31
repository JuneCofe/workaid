"""Self-check for the case catalog (data/cases.jsonl).

    python -I scripts/audit_cases.py                      # report only
    python -I scripts/audit_cases.py --fix                # clean placeholders ("—"), merge duplicates
    python -I scripts/audit_cases.py --extract-statutes   # add articles the case text itself cites (verified)
    python -I scripts/audit_cases.py --fix --extract-statutes --embed

Repairs back up the catalog to data/case_backups/ first. Nothing is invented: statutes are only added
when the case's own text cites them and the article exists in the local statute corpus; cases without
any citation are listed for a person to fill in.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.casebase import audit, schema, store  # noqa: E402
from backend.config import CASE_CATALOG_PATH  # noqa: E402

BACKUP_DIR = ROOT / "data" / "case_backups"
LEVELS = {"error": "错误", "warning": "警告", "info": "提示"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="clean placeholders and merge duplicate cases")
    parser.add_argument("--extract-statutes", action="store_true", help="add verified articles cited in the case text")
    parser.add_argument("--embed", action="store_true", help="rebuild the case vector index afterwards")
    parser.add_argument("--no-citation-check", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="also list info-level issues per case")
    parser.add_argument("--suggest", action="store_true",
                        help="for cases without statutes, list candidate articles found by search (not written)")
    args = parser.parse_args()

    raw = audit.load_raw(CASE_CATALOG_PATH)
    if not raw:
        print(f"{CASE_CATALOG_PATH} is empty or missing")
        return 1
    engine = None
    if not args.no_citation_check:
        from backend.rag import _load_engine

        engine = _load_engine()

    records = audit.clean_records(raw)
    changed = records != raw
    log: list[str] = []
    if args.fix:
        records, merged = audit.merge_duplicates(records)
        log += merged
        changed = changed or bool(merged)
        if changed and not merged:
            log.append("已清理占位符（—、无 等）和法律名称括号")
    if args.extract_statutes:
        if engine is None:
            print("需要本地法规库才能提取法条，请去掉 --no-citation-check")
            return 1
        for record in records:
            added = audit.extract_statutes(record, engine)
            if added:
                record["statutes"] = record["statutes"] + added
                record["content_hash"] = schema.content_hash(record)
                log.append(f"补充法条：{record['title'][:24]} ← {', '.join(a.replace('|', ' ') for a in added)}")
                changed = True

    issues = audit.audit(records, engine)
    counts = collections.Counter((i.level, i.code) for i in issues)
    print(f"案例 {len(records)} 条；已核验 {sum(r['reviewed'] for r in records)} 条\n")
    for (level, code), number in sorted(counts.items(), key=lambda kv: ("error", "warning", "info").index(kv[0][0])):
        print(f"  [{LEVELS[level]}] {code}: {number}")
    shown = [i for i in issues if args.verbose or i.level != "info"]
    if shown:
        print()
    for issue in shown:
        print(f"  [{LEVELS[issue.level]}] {issue.title[:26]} — {issue.message}")

    if args.suggest:
        from backend.rag import search_statutes_local

        print("\n候选法条（由检索得出，仅供人工对照案情后填写，未写入）：")
        for record in records:
            if record["statutes"]:
                continue
            query = f"{record['title']} {record['holding'][:200]}"
            found = search_statutes_local(query, 3)
            names = "；".join(f"《{i['law_name']}》{i['article_number']}" for i in found) or "（没有高相关条文）"
            print(f"  {record['title'][:26]} → {names}")

    if args.fix or args.extract_statutes:
        print()
        for line in log:
            print(" ", line)
        if not changed:
            print("  没有需要修复的内容")
        else:
            for record in records:
                record["content_hash"] = schema.content_hash(record)
            backup = audit.write_raw(CASE_CATALOG_PATH, records, BACKUP_DIR)
            print(f"\n已写入 {len(records)} 条（备份：{backup.relative_to(ROOT) if backup else '无'}）")
            if args.embed:
                from backend.retrieval import embedding

                store._CATALOGS.clear()
                if embedding.enabled():
                    print("向量索引：", store.build_vectors(CASE_CATALOG_PATH))
                else:
                    print("未配置 EMBEDDING_API_KEY，跳过向量索引")
    elif changed:
        print("\n（只读模式：发现可自动修复的问题，加 --fix 修复）")
    return 1 if any(i.level == "error" for i in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())

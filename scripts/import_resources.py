"""Import hand-collected assertion scripts (话术) or institutions (机构目录).

    python -I scripts/import_resources.py scripts --check        # validate data/script_import/*
    python -I scripts/import_resources.py scripts                # write data/scripts.jsonl
    python -I scripts/import_resources.py institutions --check   # validate data/institution_import/*
    python -I scripts/import_resources.py institutions

Inputs: .jsonl or .json. Files whose names start with "_" and the templates/ folder are ignored.
Existing records are never overwritten without --update, the target is backed up first, and one
failing record stops the whole batch. For scripts, cited articles are checked against the statute corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend import directory, scriptbase  # noqa: E402
from backend.casebase import audit  # noqa: E402
from backend.config import INSTITUTIONS_PATH, SCRIPTS_PATH  # noqa: E402

KINDS = {
    "scripts": {"module": scriptbase, "target": SCRIPTS_PATH, "inbox": ROOT / "data" / "script_import", "label": "话术"},
    "institutions": {"module": directory, "target": INSTITUTIONS_PATH, "inbox": ROOT / "data" / "institution_import", "label": "机构"},
}
BACKUP_DIR = ROOT / "data" / "case_backups"


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else [data]


def verify_statutes(record: dict, engine) -> tuple[list[str], list[str], list[str]]:
    errors, warnings, canonical = [], [], []
    for entry in record["statutes"]:
        law, _, article = entry.partition("|")
        found = json.loads(engine.law_exact_search(law, article))
        if found.get("success"):
            canonical.append(f"{found['data']['law_name']}|{found['data']['article_number']}")
        elif "未找到法律" in found.get("message", ""):
            warnings.append(f"《{law}》不在本地法规库中，无法核验 {article}")
            canonical.append(entry)
        else:
            errors.append(f"《{law}》{article} 在本地法规库中不存在，请核对条号")
    return errors, warnings, canonical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true", help="replace existing records whose content changed")
    parser.add_argument("--no-citation-check", action="store_true")
    args = parser.parse_args()
    spec = KINDS[args.kind]
    module, target, inbox = spec["module"], spec["target"], spec["inbox"]

    files = sorted(p for p in inbox.glob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"} and not p.name.startswith("_"))
    if not files:
        print(f"{inbox} 里没有 .json/.jsonl 文件")
        return 1
    engine = None
    if args.kind == "scripts" and not args.no_citation_check:
        from backend.rag import _load_engine

        engine = _load_engine()
        if engine is None:
            print("本地法规库不可用；加 --no-citation-check 跳过法条核验")
            return 1

    stored = {r["id"]: r for r in map(module.normalize_record, audit.load_raw(target))}
    batch: dict[str, dict] = {}
    failed = added = updated = same = 0
    for path in files:
        try:
            raws = read_records(path)
        except (ValueError, OSError) as exc:
            print(f"FAIL  {path.name}: 无法解析（{exc}）")
            failed += 1
            continue
        for number, raw in enumerate(raws, start=1):
            record = module.normalize_record(raw)
            label = f"{path.name}#{number} {(record.get('title') or record.get('name') or '')[:26]}"
            errors, warnings = module.validate_record(record)
            if engine is not None and record["statutes"]:
                more_errors, more_warnings, canonical = verify_statutes(record, engine)
                errors += more_errors
                warnings += more_warnings
                record["statutes"] = canonical
                record = module.normalize_record({**record, "statutes": canonical, "id": record["id"]})
            if record["id"] in batch:
                errors.append("本批次内重复")
            if errors:
                failed += 1
                print(f"FAIL  {label}")
                for item in errors:
                    print(f"        - {item}")
                continue
            existing = stored.get(record["id"])
            if existing is None:
                status, added = "NEW", added + 1
            elif existing["content_hash"] == record["content_hash"]:
                status, same = "SAME", same + 1
            elif args.update:
                status, updated = "UPDATE", updated + 1
            else:
                failed += 1
                print(f"FAIL  {label}\n        - 已存在且内容不同；确认要覆盖请加 --update")
                continue
            print(f"{status:<7}{label}  [{'已审核' if record['reviewed'] else '未审核'}]")
            for item in warnings:
                print(f"        ! {item}")
            if status != "SAME":
                batch[record["id"]] = record

    print(f"\n{added} 新增，{updated} 更新，{same} 未变，{failed} 失败")
    if args.check:
        print("只校验，没有写入（--check）")
        return 1 if failed else 0
    if failed:
        print("有记录未通过校验，整批没有写入")
        return 1
    if batch:
        stored.update(batch)
        audit.write_raw(target, list(stored.values()), BACKUP_DIR)
        print(f"{spec['label']}库现有 {len(stored)} 条（{target.relative_to(ROOT)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Import hand-collected cases into the local case catalog (data/cases.jsonl).

    1. Put case files in data/case_import/ (see data/case_import/README.md; templates are in templates/).
    2. python -I scripts/import_cases.py --check     # parse and validate, write nothing
       python -I scripts/import_cases.py             # add new cases (existing ones are never overwritten)
       python -I scripts/import_cases.py --update    # also replace cases whose content changed
       python -I scripts/import_cases.py --embed     # then build/refresh the case vector index
Supported inputs: .jsonl / .json (list or one object), .md / .txt / .docx (field-per-line template).
Files whose names start with "_" and the templates/ folder are ignored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.casebase import audit, schema, store  # noqa: E402
from backend.config import CASE_CATALOG_PATH  # noqa: E402
from backend.retrieval import embedding  # noqa: E402

IMPORT_DIR = ROOT / "data" / "case_import"
BACKUP_DIR = ROOT / "data" / "case_backups"
SEPARATOR = re.compile(r"^\s*={4,}\s*$")
FIELD_LINE = re.compile(r"^([^：:【\s]{2,8})\s*[：:]\s*(.*)$")
SECTION_LINE = re.compile(r"^【([^】]{2,8})】\s*(.*)$")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        import importlib.util

        spec = importlib.util.spec_from_file_location("import_laws", ROOT / "scripts" / "import_laws.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.read_text(path)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path.name}")


def parse_template(text: str) -> list[dict]:
    """Header lines 字段：值, then 【简要案情】-style sections; cases are separated by a ==== line."""
    cases, blocks, current = [], [], []
    for line in text.splitlines():
        if SEPARATOR.match(line):
            blocks.append(current)
            current = []
        else:
            current.append(line)
    blocks.append(current)
    for lines in blocks:
        record: dict[str, str] = {}
        section = None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            head = SECTION_LINE.match(stripped)
            if head and head.group(1) in schema.SECTION_LABELS:
                section = schema.SECTION_LABELS[head.group(1)]
                record[section] = head.group(2).strip()
                continue
            field = FIELD_LINE.match(stripped)
            if section is None and field and field.group(1) in schema.FIELD_LABELS:
                record[schema.FIELD_LABELS[field.group(1)]] = field.group(2).strip()
            elif section is not None:
                record[section] = (record[section] + "\n" + stripped).strip()
        if record:
            cases.append(record)
    return cases


def read_cases(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, list) else [data]
    return parse_template(read_text(path))


def check_citations(record: dict, engine) -> tuple[list[str], list[str], list[str]]:
    """Verify each cited article exists in the statute corpus; returns (errors, warnings, canonical refs)."""
    errors, warnings, canonical = [], [], []
    for entry in record["statutes"]:
        law, _, article = entry.partition("|")
        found = json.loads(engine.law_exact_search(law, article))
        if found.get("success"):
            data = found["data"]
            canonical.append(f"{data['law_name']}|{data['article_number']}")
        elif "未找到法律" in found.get("message", ""):
            warnings.append(f"适用法条《{law}》不在本地法规库中，无法核验 {article}")
            canonical.append(entry)
        else:
            errors.append(f"适用法条《{law}》{article} 在本地法规库中不存在，请核对条号")
    return errors, warnings, canonical


def load_catalog() -> dict[str, dict]:
    catalog = {}
    for record in audit.load_raw(CASE_CATALOG_PATH):
        normalized = schema.normalize_record(record)
        catalog[normalized["id"]] = {**record, **normalized}
    return catalog


def write_catalog(catalog: dict[str, dict]) -> None:
    audit.write_raw(CASE_CATALOG_PATH, list(catalog.values()), BACKUP_DIR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=IMPORT_DIR)
    parser.add_argument("--check", action="store_true", help="validate only; write nothing")
    parser.add_argument("--update", action="store_true", help="replace existing cases whose content changed")
    parser.add_argument("--embed", action="store_true", help="build/refresh the case vector index afterwards")
    parser.add_argument("--embed-only", action="store_true", help="only rebuild the vector index")
    parser.add_argument("--no-citation-check", action="store_true")
    args = parser.parse_args()

    if args.embed_only:
        return embed()

    files = sorted(p for p in args.dir.glob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".docx"}
                   and not p.name.startswith("_") and p.name.lower() != "readme.md")
    if not files:
        print(f"no case files in {args.dir}")
        return 1
    engine = None
    if not args.no_citation_check:
        from backend.rag import _load_engine

        engine = _load_engine()
        if engine is None:
            print("statute engine unavailable; use --no-citation-check to skip citation verification")
            return 1

    catalog = load_catalog()
    batch: dict[str, dict] = {}
    failed = added = updated = unchanged = 0
    for path in files:
        try:
            raws = read_cases(path)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"FAIL  {path.name}: cannot parse ({exc})")
            failed += 1
            continue
        for number, raw in enumerate(raws, start=1):
            record = schema.normalize_record(raw)
            label = f"{path.name}#{number} {record['title'][:24] or '(无标题)'}"
            errors, warnings = schema.validate_record(record)
            canonical = record["statutes"]
            if engine is not None and record["statutes"]:
                e, w, canonical = check_citations(record, engine)
                errors += e
                warnings += w
            if record["id"] in batch:
                errors.append("本批次内案号/标题重复")
            duplicate = audit.find_duplicate(record, list(catalog.values()) + list(batch.values()))
            if duplicate is not None:
                errors.append(f"与已有案例疑似重复：{duplicate['title'][:30]}（ID {duplicate['id']}）；确属同一案例请删除重复文件")
            if errors:
                failed += 1
                print(f"FAIL  {label}")
                for item in errors:
                    print(f"        - {item}")
                continue
            record["statutes"] = canonical
            record["content_hash"] = schema.content_hash(record)
            record["retrieved_at"] = time.strftime("%Y-%m-%d")
            existing = catalog.get(record["id"])
            if existing is None:
                status, added = "NEW", added + 1
            elif existing.get("content_hash") == record["content_hash"]:
                status, unchanged = "SAME", unchanged + 1
            elif args.update:
                status, updated = "UPDATE", updated + 1
            else:
                failed += 1
                print(f"FAIL  {label}\n        - 已存在且内容不同；确认要覆盖请加 --update")
                continue
            print(f"{status:<6}{label}  [{'已核验' if record['reviewed'] else '来源待核验'}]")
            for item in warnings:
                print(f"        ! {item}")
            if status != "SAME":
                batch[record["id"]] = record

    print(f"\n{added} new, {updated} updated, {unchanged} unchanged, {failed} failed")
    if args.check:
        print("nothing written (--check)")
        return 1 if failed else 0
    if failed:
        print("some cases failed validation; nothing was written")
        return 1
    if batch:
        catalog.update(batch)
        write_catalog(catalog)
        print(f"catalog now holds {len(catalog)} cases ({CASE_CATALOG_PATH.relative_to(ROOT)})")
    return embed() if args.embed else 0


def embed() -> int:
    if not embedding.enabled():
        print("EMBEDDING_API_KEY is not set; the catalog works without vectors, add the key to build them.")
        return 0
    store._CATALOGS.clear()
    try:
        result = store.build_vectors(CASE_CATALOG_PATH)
    except RuntimeError as exc:
        print(exc)
        return 3
    print(f"case vectors: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

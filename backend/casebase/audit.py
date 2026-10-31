"""Catalog self-check: duplicates, placeholders, weak records and unresolved citations.

Used by scripts/audit_cases.py (report and safe repairs) and by scripts/import_cases.py, which refuses
a new case that duplicates one already in the catalog.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CITATION, PROCEDURAL_CONCEPTS, normalize_record, parse_statutes

SHORT_SUMMARY_CHARS = 100
SIMILAR_PREFIX = 60
TEXT_FOR_CITATIONS = ("summary", "dispute_focus", "holding", "result", "significance")
FILLED_FIELDS = ("case_number", "court", "cause", "decision_date", "publish_date", "region", "dispute_focus",
                 "holding", "result", "significance", "source_url", "statutes", "keywords")


@dataclass
class Issue:
    level: str  # error | warning | info
    code: str
    case_id: str
    title: str
    message: str


def _norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text or "").lower()


def duplicate_keys(record: dict[str, Any]) -> set[tuple[str, str]]:
    """Keys that identify the same case. A shared title alone is not enough (series cases share one),
    so a title counts only together with the same opening of the facts."""
    keys = {("id", record["id"])}
    if record.get("case_number"):
        keys.add(("case_number", _norm(record["case_number"])))
    keys.add(("title_facts", _norm(record["title"]) + "|" + _norm(record["summary"])[:SIMILAR_PREFIX]))
    return keys


def find_duplicate(record: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    """An existing case with a different id that shares a case number or title+facts with this one.

    Same-id records are the importer's update/unchanged path, not duplicates.
    """
    mine = {key for key in duplicate_keys(record) if key[0] != "id"}
    for other in catalog:
        if other["id"] != record["id"] and mine & duplicate_keys(other):
            return other
    return None


def _richness(record: dict[str, Any]) -> tuple[int, int, int]:
    filled = sum(1 for name in FILLED_FIELDS if record.get(name))
    return (1 if record.get("reviewed") else 0, filled, len(record.get("summary", "")) + len(record.get("holding", "")))


def merge_duplicates(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse records that share an identifying key. The best record wins (reviewed first, then
    most fields filled); empty fields on the winner are filled from the losers."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in duplicate_keys(record):
            if key in owner:
                parent[find(index)] = find(owner[key])
            else:
                owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(find(index), []).append(index)

    kept, log = [], []
    for members in groups.values():
        members.sort(key=lambda i: _richness(records[i]), reverse=True)
        winner = dict(records[members[0]])
        for loser_index in members[1:]:
            loser = records[loser_index]
            for name in FILLED_FIELDS:
                if not winner.get(name) and loser.get(name):
                    winner[name] = loser[name]
            log.append(f"合并重复：保留 {winner['id']}，移除 {loser['id']}（{winner['title'][:24]}）")
        kept.append(winner)
    return kept, log


def extract_statutes(record: dict[str, Any], engine: Any) -> list[str]:
    """Articles the case text itself cites that are not yet listed, kept only if they exist in the corpus."""
    found: list[str] = []
    text = "\n".join(record.get(name, "") for name in TEXT_FOR_CITATIONS)
    for law, article in dict.fromkeys(CITATION.findall(text)):
        result = json.loads(engine.law_exact_search(law, article))
        if not result.get("success"):
            continue
        entry = f"{result['data']['law_name']}|{result['data']['article_number']}"
        if entry not in record["statutes"] and entry not in found:
            found.append(entry)
    return found


def check_citations(record: dict[str, Any], engine: Any) -> list[Issue]:
    issues = []
    for law, article in parse_statutes(record["statutes"]):
        result = json.loads(engine.law_exact_search(law, article))
        if not result.get("success"):
            issues.append(Issue("warning", "unresolved_statute", record["id"], record["title"],
                                f"适用法条《{law}》{article}：{result.get('message', '无法核验')}"))
    return issues


def audit(records: list[dict[str, Any]], engine: Any = None) -> list[Issue]:
    issues: list[Issue] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        rid, title = record["id"], record["title"]
        for key in duplicate_keys(record):
            if key in seen and seen[key] is not record:
                issues.append(Issue("error", "duplicate", rid, title, f"与 {seen[key]['id']}（{seen[key]['title'][:20]}）疑似重复（{key[0]}）"))
            seen.setdefault(key, record)
        if len(record["summary"]) < SHORT_SUMMARY_CHARS:
            issues.append(Issue("info", "short_summary", rid, title, f"简要案情仅 {len(record['summary'])} 字"))
        if not record["statutes"]:
            issues.append(Issue("warning", "no_statutes", rid, title, "没有适用法条，无法按法条关联检索"))
        if not record["decision_date"] and not record["publish_date"]:
            issues.append(Issue("info", "no_date", rid, title, "裁判日期和发布日期都缺失"))
        elif not record["decision_date"]:
            issues.append(Issue("info", "no_decision_date", rid, title, "缺少裁判日期"))
        if not record["holding"]:
            issues.append(Issue("info", "no_holding", rid, title, "缺少裁判要旨"))
        if not record["reviewed"]:
            issues.append(Issue("info", "unreviewed", rid, title, "来源待核验"))
        if not (set(record["scenarios"]) - PROCEDURAL_CONCEPTS):
            issues.append(Issue("warning", "no_scenario", rid, title, "没有识别出实体争议场景，只能靠文字匹配"))
        if engine is not None:
            issues.extend(check_citations(record, engine))
    return issues


def load_raw(path: Path) -> list[dict[str, Any]]:
    """Stored records exactly as written (unparseable lines are skipped)."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def write_raw(path: Path, records: list[dict[str, Any]], backup_dir: Path) -> Path | None:
    """Replace the catalog atomically, copying the previous file into backup_dir first."""
    backup = None
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"cases-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    ordered = sorted(records, key=lambda r: r["id"])
    temp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n", encoding="utf-8")
    temp.replace(path)
    return backup


def clean_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-normalise stored records (placeholders, bracket styles, derived id) keeping extra stored fields.

    The id is recomputed from the cleaned case number/title, because that is what a re-import of the
    same source would produce.
    """
    cleaned = []
    for raw in raw_records:
        normalized = normalize_record({key: value for key, value in raw.items() if key != "id"})
        cleaned.append({**raw, **normalized})
    return cleaned

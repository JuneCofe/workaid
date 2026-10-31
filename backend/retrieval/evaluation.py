from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "statute_queries.jsonl"
KS = (1, 3, 5)


@dataclass
class CaseResult:
    case: dict[str, Any]
    ranked: list[tuple[str, str]]
    first_hit: int | None
    seconds: float


@dataclass
class Report:
    name: str
    results: list[CaseResult] = field(default_factory=list)

    def positives(self) -> list[CaseResult]:
        return [r for r in self.results if r.case["expect"]]

    def hit_at(self, k: int) -> float:
        rows = self.positives()
        return sum(1 for r in rows if r.first_hit is not None and r.first_hit <= k) / max(1, len(rows))

    def mrr(self) -> float:
        rows = self.positives()
        return sum(1 / r.first_hit for r in rows if r.first_hit) / max(1, len(rows))

    def abstained(self) -> float:
        rows = [r for r in self.results if not r.case["expect"]]
        return sum(1 for r in rows if not r.ranked) / max(1, len(rows))

    def summary(self) -> dict[str, float]:
        return {
            **{f"hit@{k}": round(self.hit_at(k), 3) for k in KS},
            "mrr": round(self.mrr(), 3),
            "abstain_ok": round(self.abstained(), 3),
            "avg_seconds": round(sum(r.seconds for r in self.results) / max(1, len(self.results)), 3),
        }


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _core_name(law_name: str) -> str:
    name = re.sub(r"[（(][^)）]*[)）]|\s*English\s*$", "", law_name).strip()
    return name.removeprefix("中华人民共和国")


def _matches(expect: list[str], law: str, article: str) -> bool:
    core = _core_name(law)
    return any(
        core == law_part and article == article_part
        for law_part, article_part in (entry.split("|", 1) for entry in expect)
    )


def evaluate(
    name: str, search: Callable[[str], list[dict[str, Any]]], cases: list[dict[str, Any]], limit: int = 5
) -> Report:
    report = Report(name)
    for case in cases:
        started = time.perf_counter()
        items = search(case["query"])[:limit]
        elapsed = time.perf_counter() - started
        ranked = [(item.get("law_name", ""), item.get("article_number", "")) for item in items]
        first = next(
            (rank for rank, (law, article) in enumerate(ranked, start=1) if _matches(case["expect"], law, article)),
            None,
        )
        report.results.append(CaseResult(case, ranked, first, elapsed))
    return report


def baseline_search(module: Any) -> Callable[[str], list[dict[str, Any]]]:
    """The pre-pipeline behaviour: one fuzzy search over the raw sentence."""

    def run(query: str) -> list[dict[str, Any]]:
        payload = json.loads(module.law_fuzzy_search(query, limit=5))
        return [item for item in payload.get("data") or [] if isinstance(item, dict)]

    return run


def pipeline_search() -> Callable[[str], list[dict[str, Any]]]:
    from ..rag import search_statutes_local

    return search_statutes_local


def run_comparison(cases: list[dict[str, Any]] | None = None) -> tuple[Report, Report]:
    os.environ["RETRIEVAL_LLM_REWRITE"] = "0"
    from ..rag import _load_engine

    module = _load_engine()
    if module is None:
        raise RuntimeError("local statute engine is unavailable")
    cases = cases or load_cases()
    return (
        evaluate("baseline (raw sentence)", baseline_search(module), cases),
        evaluate("pipeline (rewrite+recall+rerank)", pipeline_search(), cases),
    )

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .lexicon import CONCEPTS, ENTITY_TERMS, Concept

MAX_QUERY_CHARS = 600
MAX_CHUNKS = 6
MAX_CONCEPTS = 5
MAX_VARIANTS = 16
LLM_TIMEOUT_SECONDS = 8

_CLAUSE_SPLIT = re.compile(
    r"[，,。！？!?；;、：:\n]+|(?:但是|可是|不过|而且|并且|另外|同时|还有|然后|后来|结果|所以|因此)"
)
_FILLER = re.compile(
    r"请问|想问一下|想问下|我想问|麻烦问下?|想咨询一下?|咨询一下?|求助|帮我看看|怎么办|怎么处理|如何处理"
    r"|怎么维权|如何维权|该怎么|应该怎么|能不能|可不可以|可以吗|行吗|吗|呢|吧|啊|呀"
)
_PRONOUN = re.compile(r"我们|咱们|他们|她们|我|你|他|她|它")
_ALWAYS_KEEP = ("original", "formal_sentence")


@dataclass(frozen=True)
class Variant:
    text: str
    kind: str
    weight: float


@dataclass
class QueryPlan:
    original: str
    chunks: list[str] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    llm_terms: list[str] = field(default_factory=list)
    llm_used: bool = False

    @property
    def law_hints(self) -> list[str]:
        hints: list[str] = []
        for concept in self.concepts:
            hints.extend(law for law in concept.laws if law not in hints)
        return hints

    def trace(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "concepts": [concept.label for concept in self.concepts],
            "variants": [{"text": v.text, "kind": v.kind} for v in self.variants],
            "llm_rewrite": self.llm_used,
        }


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()[:MAX_QUERY_CHARS]


def clean(text: str) -> str:
    """Drop question filler and pronouns so the remaining words are searchable."""
    return re.sub(r"\s+", " ", _PRONOUN.sub("", _FILLER.sub(" ", text))).strip()


def split_chunks(text: str) -> list[str]:
    pieces = [piece.strip() for piece in _CLAUSE_SPLIT.split(text) if piece and piece.strip()]
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(piece) < 4:
            chunks[-1] = f"{chunks[-1]}，{piece}"
        else:
            chunks.append(piece)
    if len(chunks) > 1 and len(chunks[0]) < 4:
        chunks[1] = f"{chunks[0]}，{chunks[1]}"
        chunks.pop(0)
    while len(chunks) > MAX_CHUNKS:
        chunks[-2] = f"{chunks[-2]}，{chunks[-1]}"
        chunks.pop()
    return chunks


def match_concepts(text: str) -> list[Concept]:
    """Concepts whose colloquial patterns occur in text, earliest mention first."""
    found: list[tuple[int, Concept]] = []
    for concept in CONCEPTS:
        match = concept.search(text)
        if match:
            found.append((match.start(), concept))
    found.sort(key=lambda pair: pair[0])
    return [concept for _, concept in found]


def formalize(text: str) -> str:
    """Rewrite everyday wording into the phrases statutes use, as space-separated terms."""
    rewritten = text
    for concept in match_concepts(text):
        match = concept.search(rewritten)
        if match:
            rewritten = rewritten[: match.start()] + f" {concept.formal[0]} " + rewritten[match.end():]
    for pattern, term in ENTITY_TERMS:
        rewritten = pattern.sub(f" {term} ", rewritten)
    parts = [part for part in re.split(r"[\s，,。；;、]+", clean(rewritten)) if len(part) >= 2]
    return " ".join(dict.fromkeys(parts))


def _concept_variants(concept: Concept) -> list[Variant]:
    law = concept.laws[0]
    variants = [
        Variant(f"{law} {concept.formal[0]}", "law_hint", 1.0),
        Variant(" ".join(concept.formal), "formal", 1.0),
        Variant(f"{law} {' '.join(concept.keywords[:3])}", "keywords", 0.8),
    ]
    if len(concept.laws) > 1:
        variants.append(Variant(f"{concept.laws[1]} {concept.formal[0]}", "law_hint", 0.7))
    return variants


def _ngram_variant(chunk: str) -> str:
    """Overlapping 4-character grams for clauses that match no lexicon concept."""
    compact = re.sub(r"[^\w]", "", chunk)
    if len(compact) < 4:
        return ""
    grams = [compact[i : i + 4] for i in range(0, len(compact) - 3, 2)]
    return " ".join(dict.fromkeys(grams[:8]))


def _dedupe(variants: list[Variant]) -> list[Variant]:
    seen: set[str] = set()
    unique: list[Variant] = []
    for variant in variants:
        key = variant.text.replace(" ", "")
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=128)
def _llm_rewrite(text: str) -> tuple[str, ...] | None:
    """Ask the chat model for formal rewrites; returns search phrases or None when unavailable."""
    if os.getenv("RETRIEVAL_LLM_REWRITE", "1") == "0" or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from .. import llm as llm_module

        model = llm_module.chat_model(
            model=os.getenv("RETRIEVAL_REWRITE_MODEL") or None, temperature=0, timeout=LLM_TIMEOUT_SECONDS, max_retries=0
        )
        if model is None:
            return None
        result = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是中国法规条文检索的查询改写器。把用户口语化的案情改写成法条里会出现的书面检索语。"
                        "只输出一个 JSON 对象，不要解释。字段：formal（一句书面改写）、"
                        "variants（3 到 5 条检索语，每条是空格分隔的法律术语短语，可含《法律名称》，"
                        "覆盖近义说法）、keywords（至多 8 个法律术语）。不要编造条文编号。"
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        payload = _parse_json_object(str(result.content))
    except Exception:
        return None
    if payload is None:
        return None
    phrases: list[str] = []
    for key in ("formal", "variants", "keywords"):
        value = payload.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                phrase = re.sub(r"[《》]", "", normalize(item))[:60]
                if len(phrase) >= 2 and phrase not in phrases:
                    phrases.append(phrase)
    return tuple(phrases[:10]) or None


def understand(text: str, *, use_llm: bool = True) -> QueryPlan:
    text = normalize(text)
    plan = QueryPlan(original=text, chunks=split_chunks(text))
    if not text:
        return plan

    variants: list[Variant] = []
    seen_concepts: list[Concept] = []
    chunk_concepts: list[list[Concept]] = []
    for chunk in plan.chunks:
        matched = match_concepts(chunk)
        chunk_concepts.append(matched)
        seen_concepts.extend(c for c in matched if c not in seen_concepts)
    seen_concepts.extend(c for c in match_concepts(text) if c not in seen_concepts)
    plan.concepts = seen_concepts[:MAX_CONCEPTS]

    for concept in plan.concepts:
        variants.extend(_concept_variants(concept))

    for chunk, matched in zip(plan.chunks, chunk_concepts):
        cleaned = clean(chunk)
        if matched:
            variants.append(Variant(cleaned, "chunk", 0.5))
        else:
            ngram = _ngram_variant(cleaned)
            if ngram:
                variants.append(Variant(ngram, "ngram", 0.4))
            variants.append(Variant(cleaned, "chunk", 0.6))

    formal = formalize(text)
    if formal:
        variants.append(Variant(formal, "formal_sentence", 0.9))
    variants.append(Variant(clean(text), "original", 0.5))

    # The lexicon already rewrites questions it recognises; a model call (3-4 s) only pays off for the
    # ones it cannot place. RETRIEVAL_LLM_REWRITE=always asks the model every time, 0 never.
    if use_llm and (not plan.concepts or os.getenv("RETRIEVAL_LLM_REWRITE") == "always"):
        phrases = _llm_rewrite(text)
        if phrases:
            plan.llm_used = True
            plan.llm_terms = [p for p in phrases if " " not in p]
            variants.extend(Variant(phrase, "llm", 0.9) for phrase in phrases)

    unique = _dedupe(variants)
    core = [v for v in unique if v.kind in _ALWAYS_KEEP]
    rest = sorted((v for v in unique if v.kind not in _ALWAYS_KEEP), key=lambda v: -v.weight)
    plan.variants = rest[: MAX_VARIANTS - len(core)] + core
    return plan

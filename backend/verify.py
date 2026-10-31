"""Verify the citations in a generated answer against the statute corpus and retrieved cases.

A language model can write a plausible-looking article number that does not exist, quote a provision
inaccurately, cite a repealed law, or invent a case number. Every 《法名》第X条 in the answer is looked up
in the local corpus (the retrieved evidence is the fallback for laws the corpus lacks); anything that
fails is deleted or marked in the text, and a deterministic section lists what was checked together
with each law's 时效性, 公布日期 and 施行日期.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rag.law_data_search import normalize_article_number

NUM = r"[一二三四五六七八九十百千零〇两0-9]+"
ARTICLE = rf"第{NUM}条(?:之{NUM})?"
SUBITEM = rf"(?:第{NUM}[款项]|[（(][一二三四五六七八九十]+[）)])*"
LAW_CITE = re.compile(rf"《([^》\n]{{2,80}})》\s*(?P<first>{ARTICLE})")
CONTINUATION = re.compile(rf"\s*[、和及与]\s*(?P<art>{ARTICLE})")
QUOTE_AFTER = re.compile(r"^[^。；\n“「]{0,20}[“「]([^”」\n]{8,300})[”」]")
CASE_NUMBER = re.compile(r"[（(]\d{4}[）)]\s*[^\s，。；、（）()：:]{1,14}?\d+号")
MIN_QUOTE_PIECE = 6

# Laws replaced when the 民法典 took effect on 2021.1.1 (listed in 民法典第一千二百六十条), plus rules the
# project owner confirmed repealed.
REPEALED: dict[str, str] = {
    **{name: "已被《民法典》取代，自 2021 年 1 月 1 日起废止" for name in (
        "婚姻法", "继承法", "民法通则", "收养法", "担保法", "合同法", "物权法", "侵权责任法", "民法总则")},
    "违反和解除劳动合同的经济补偿办法": "已废止，应适用现行有效的《劳动合同法》及相关规定",
}


def _core(name: str) -> str:
    text = re.sub(r"\s*English.*$", "", name)
    text = re.sub(r"[（(][^)）]*[)）]", "", text)
    return text.replace("《", "").replace("》", "").strip().removeprefix("中华人民共和国")


def _plain(text: str) -> str:
    return re.sub(r"[^\w]", "", text or "")


@dataclass
class Citation:
    law: str
    article: str
    start: int
    end: int
    anchor: int
    status: str = ""
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def extract_citations(text: str) -> list[Citation]:
    """Every 《law》article citation, including 《law》第A条、第B条 lists and trailing 第N款."""
    found: list[Citation] = []
    for match in LAW_CITE.finditer(text):
        law = match.group(1)
        start, end = match.start("first"), match.end("first")
        while True:
            anchor = end + len(re.match(SUBITEM, text[end:]).group(0))
            found.append(Citation(law, text[start:end], start, end, anchor))
            more = CONTINUATION.match(text, anchor)
            if not more:
                break
            start, end = more.start("art"), more.end("art")
    return found


def _quote_pieces(quote: str) -> list[str]:
    pieces = re.split(r"…+|\.{3,}|⋯+", quote)
    return [p for p in (_plain(piece) for piece in pieces) if len(p) >= MIN_QUOTE_PIECE]


def _quote_matches(quote_text: str, content: str) -> bool:
    pieces = _quote_pieces(quote_text)
    plain = _plain(content)
    return all(piece in plain for piece in pieces) if pieces else True


def _classify(citation: Citation, engine: Any, evidence: list[dict[str, Any]], text: str) -> None:
    core = _core(citation.law)
    if core in REPEALED:
        citation.status, citation.detail = "repealed", REPEALED[core]
        return
    result: dict[str, Any] = {}
    if engine is not None:
        try:
            result = json.loads(engine.law_exact_search(citation.law, citation.article))
        except Exception:
            result = {}
    data = result.get("data") if result.get("success") else None
    if data:
        resolved = _core(data["law_name"])
        # The engine matches abbreviations loosely ("民事诉讼法解释" -> 民事诉讼法); only accept the same law.
        if core == resolved or (len(core) >= 4 and (core in resolved or resolved in core)):
            citation.data = data
            citation.status = "ok" if data.get("effectiveness", "现行有效") == "现行有效" else "not_in_force"
            if citation.status == "not_in_force":
                citation.detail = f"该法时效性为“{data.get('effectiveness')}”"
            elif (quote := QUOTE_AFTER.match(text[citation.anchor : citation.anchor + 160])) and not _quote_matches(
                quote.group(1), data["content"]
            ):
                citation.status, citation.detail = "quote_mismatch", "引文与法规原文不一致，请以原文为准"
            return
    message = result.get("message", "")
    if engine is not None and message.startswith("未找到法条"):
        # The law resolved but this article number does not exist in it.
        resolved_name = message.removeprefix("未找到法条:").strip().rsplit(" ", 1)[0]
        if _core(resolved_name) == core or (len(core) >= 4 and core in _core(resolved_name)):
            citation.status, citation.detail = "no_article", f"《{resolved_name}》中没有这一条"
            return
    wanted = normalize_article_number(citation.article)
    for item in evidence:
        if _core(str(item.get("law_name", ""))) == core and normalize_article_number(str(item.get("article_number", ""))) == wanted:
            citation.data = item
            citation.status, citation.detail = "evidence_only", "该条来自检索来源，未经本地法规库核对"
            return
    citation.status = "no_law"
    citation.detail = "法规库中没有这部法规，未能核验"


def _law_line(citation: Citation) -> str:
    data = citation.data
    name = data.get("law_name") or citation.law
    parts = [data.get("effectiveness") or ("" if citation.status != "evidence_only" else "时效待核实")]
    if data.get("publish_date"):
        parts.append(f"{data['publish_date']} 公布")
    if data.get("implement_date"):
        parts.append(f"{data['implement_date']} 施行")
    meta = "；".join(p for p in parts if p)
    return f"《{name}》{citation.article}" + (f"（{meta}）" if meta else "")


@dataclass
class VerifyReport:
    answer: str
    section: str
    checked: int = 0
    problems: int = 0
    citations: list[Citation] = field(default_factory=list)


def verify_answer(
    answer: str,
    evidence: list[dict[str, Any]] | None = None,
    engine: Any = None,
    known_case_numbers: set[str] | None = None,
) -> VerifyReport:
    evidence = evidence or []
    citations = extract_citations(answer)
    memo: dict[tuple[str, str], Citation] = {}
    for citation in citations:
        key = (_core(citation.law), normalize_article_number(citation.article) or citation.article)
        if key not in memo:
            _classify(citation, engine, evidence, answer)
            memo[key] = citation
        else:  # same article cited again: the lookup result is reused, a quote is still checked here
            first = memo[key]
            citation.status, citation.detail, citation.data = first.status, first.detail, first.data
            if citation.status == "ok" and (quote := QUOTE_AFTER.match(answer[citation.anchor : citation.anchor + 160])) \
                    and not _quote_matches(quote.group(1), citation.data["content"]):
                citation.status, citation.detail = "quote_mismatch", "引文与法规原文不一致，请以原文为准"

    edits: list[tuple[int, int, str]] = []
    for citation in citations:
        if citation.status == "no_article":
            edits.append((citation.start, citation.end, f"【{citation.detail}，已删除】"))
        elif citation.status in {"repealed", "no_law", "not_in_force", "quote_mismatch"}:
            edits.append((citation.anchor, citation.anchor, f"［{citation.detail}］"))

    case_flags: list[str] = []
    if known_case_numbers is None:
        known_case_numbers = _known_case_numbers(evidence)
    known = {_plain(number) for number in known_case_numbers}
    for match in CASE_NUMBER.finditer(answer):
        if _plain(match.group(0)) not in known:
            edits.append((match.end(), match.end(), "［该案号不在检索到的案例中，未能核验］"))
            case_flags.append(match.group(0))

    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        answer = answer[:start] + replacement + answer[end:]

    return VerifyReport(
        answer=answer,
        section=_section(memo, case_flags, engine),
        checked=len(memo),
        problems=sum(1 for c in memo.values() if c.status not in {"ok", "evidence_only"}) + len(case_flags),
        citations=citations,
    )


def _known_case_numbers(evidence: list[dict[str, Any]]) -> set[str]:
    known = {str(item.get("case_number", "")) for item in evidence if item.get("case_number")}
    try:
        from .casebase import store
        from .config import CASE_CATALOG_PATH

        known |= {r["case_number"] for r in store.records(CASE_CATALOG_PATH) if r.get("case_number")}
    except Exception:
        pass
    return known


def _section(memo: dict[tuple[str, str], Citation], case_flags: list[str], engine: Any) -> str:
    if not memo and not case_flags:
        return ""
    verified = [c for c in memo.values() if c.status in {"ok", "evidence_only"}]
    flagged = [c for c in memo.values() if c.status not in {"ok", "evidence_only"}]
    lines = ["### 引用核验"]
    if engine is None and memo:
        lines.append("本地法规库不可用，本次回答中的法条引用没有经过核对，请自行查证。")
    if verified:
        lines.append("以下法条已与法规库核对，存在且条号正确（括号内为法规的时效信息，使用前请再确认是否有新的修改）：")
        lines += [f"- {_law_line(c)}" + ("　（仅来自检索来源）" if c.status == "evidence_only" else "") for c in verified]
    if flagged or case_flags:
        lines.append("以下内容未通过核验，已在正文中删除或标注，请不要据此判断：")
        lines += [f"- 《{c.law}》{c.article}：{c.detail}" for c in flagged]
        lines += [f"- 案号 {number}：不在检索到的案例中" for number in case_flags]
    return "\n".join(lines)

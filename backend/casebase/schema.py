"""Case record structure, normalisation and validation.

A case record is one authoritative published case (guiding case, typical case, arbitration/court
roundup entry, ...). Loading is tolerant (old hand-made records still work); the importer applies
the strict checks in ``validate_record``.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlparse

from ..retrieval.lexicon import CONCEPTS

SOURCE_TYPES = (
    "最高人民法院指导性案例",
    "最高人民法院典型案例",
    "最高人民法院人社部联合发布典型案例",
    "人力资源社会保障部典型案例",
    "高级人民法院典型案例",
    "中基层人民法院典型案例",
    "劳动人事争议仲裁委典型案例",
    "人民法院案例库入库案例",
    "公报案例",
    "已公开裁判文书",
    "其他官方发布案例",
)
AUTHORITATIVE_HOSTS = (
    "court.gov.cn", "chinacourt.org", "spp.gov.cn", "mohrss.gov.cn", "moj.gov.cn", "12333.gov.cn",
    "acftu.org", "gov.cn",
)
# How far a case can be trusted as a citation, shown to the user next to every case.
REVIEWED_LABEL = "已核验"
UNREVIEWED_LABEL = "来源待核验"
EXTERNAL_LABEL = "外部来源（未经本库核验）"
TIER_ORDER = (REVIEWED_LABEL, UNREVIEWED_LABEL, EXTERNAL_LABEL)
FIELD_LABELS = {
    "标题": "title", "案号": "case_number", "审理法院": "court", "法院": "court", "裁判日期": "decision_date",
    "案由": "cause", "案件类型": "case_type", "来源类型": "source_type", "发布机构": "issuer",
    "发布日期": "publish_date", "来源网址": "source_url", "地区": "region", "适用法条": "statutes",
    "关键词": "keywords", "已核验": "reviewed", "核验人": "reviewed_by", "来源说明": "provenance_note",
}
SECTION_LABELS = {
    "简要案情": "summary", "基本案情": "summary", "案情": "summary", "争议焦点": "dispute_focus",
    "裁判要旨": "holding", "裁判理由": "holding", "裁判结果": "result", "典型意义": "significance",
}
TEXT_FIELDS = ("title", "case_number", "court", "cause", "case_type", "source_type", "issuer", "region",
               "summary", "dispute_focus", "holding", "result", "significance", "provenance_note", "reviewed_by")
LIST_FIELDS = ("keywords", "statutes")
DATE = re.compile(r"^\d{4}[.\-]\d{1,2}[.\-]\d{1,2}$")
NUM = r"[一二三四五六七八九十百千零〇两0-9]+"
CITATION = re.compile(rf"《([^》]{{2,60}})》\s*(第{NUM}条(?:之{NUM})?)")
PIPE_CITATION = re.compile(rf"^([^|｜]{{2,60}})[|｜]\s*(第{NUM}条(?:之{NUM})?)$")
PRIVACY = (
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
)
MIN_SUMMARY_CHARS = 40
MAX_FIELD_CHARS = 20000
# Procedural topics appear in nearly every case, so they say nothing about what the dispute is about.
PROCEDURAL_CONCEPTS = frozenset({"lawsuit", "arbitration", "evidence", "legal_aid"})


def split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = re.split(r"[、,，;；\n]+", str(value or ""))
    return list(dict.fromkeys(clean_value(part) for part in parts if clean_value(part)))


# Spreadsheet-style "nothing here" markers; stored verbatim they look like real data and defeat
# duplicate detection (a case number of "—" never equals an empty one).
PLACEHOLDERS = frozenset({"—", "－", "-", "--", "——", "/", "／", "无", "暂无", "不详", "未知", "n/a", "na", "null", "none", "空", "未填写", "待补充"})


def clean_value(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in PLACEHOLDERS else text


def clean_law_name(name: str) -> str:
    """Normalise angle brackets so 《》, <>, 〈〉 all name the same law."""
    text = name.strip().replace("<", "《").replace(">", "》").replace("〈", "《").replace("〉", "》")
    return text


def parse_statutes(value: Any) -> list[tuple[str, str]]:
    """Accept 《劳动合同法》第四十七条, 劳动合同法|第四十七条, or a list of either."""
    refs: list[tuple[str, str]] = []
    items = value if isinstance(value, list) else re.split(r"[;；\n]+", str(value or ""))
    for item in items:
        text = clean_law_name(str(item).strip())
        if not text:
            continue
        found = CITATION.findall(text)
        if found:
            refs.extend(found)
            continue
        piped = PIPE_CITATION.match(text)
        if piped:
            refs.append((clean_law_name(piped.group(1)), piped.group(2)))
    return list(dict.fromkeys(refs))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "是", "yes", "y"}


def content_hash_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def case_id(record: dict[str, Any]) -> str:
    basis = (record.get("case_number") or record.get("title") or "") + "|" + (record.get("issuer") or "")
    return hashlib.sha1(basis.strip().encode("utf-8")).hexdigest()[:12]


def content_hash(record: dict[str, Any]) -> str:
    material = "\n".join(str(record.get(name, "")) for name in (*TEXT_FIELDS, "source_url", "decision_date", "publish_date"))
    material += "\n" + "|".join(f"{a}{b}" for a, b in parse_statutes(record.get("statutes", [])))
    material += "\n" + "|".join(split_list(record.get("keywords", [])))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def scenarios_of(text: str) -> list[str]:
    """Lexicon concepts whose colloquial or formal wording appears in the case text."""
    found = []
    for concept in CONCEPTS:
        if concept.search(text) or any(term in text for term in concept.formal):
            found.append(concept.key)
    return found


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {name: clean_value(raw.get(name, ""))[:MAX_FIELD_CHARS] for name in TEXT_FIELDS}
    record["source_url"] = clean_value(raw.get("source_url") or raw.get("url") or "")
    for name in ("decision_date", "publish_date"):
        record[name] = clean_value(raw.get(name, "")).replace("-", ".")
    record["summary"] = record["summary"] or clean_value(raw.get("content", ""))
    record["keywords"] = split_list(raw.get("keywords", []))
    record["statutes"] = [f"{law}|{article}" for law, article in parse_statutes(raw.get("statutes", []))]
    record["reviewed"] = _truthy(raw.get("reviewed", False))
    record["id"] = str(raw.get("id") or "").strip() or case_id(record)
    # Scenario tags come from the legal reasoning fields; incidental words in the facts ("租赁车辆")
    # would otherwise tag a labour-relation case as a tenancy case. Facts are only a last resort.
    reasoning = " ".join([record["title"], record["cause"], record["dispute_focus"], record["holding"],
                          record["significance"], " ".join(record["keywords"])])
    manual = split_list(raw.get("scenarios_manual", []))
    derived = scenarios_of(reasoning)
    if not set(derived) - PROCEDURAL_CONCEPTS:
        derived = list(dict.fromkeys(derived + scenarios_of(record["summary"])))
    record["scenarios"] = manual or derived
    record["content_hash"] = content_hash(record)
    return record


def host_ok(url: str) -> bool:
    """True for an http(s) URL on a government or court domain."""
    return _host_ok(url)


def _host_ok(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return urlparse(url).scheme in {"http", "https"} and any(host == h or host.endswith("." + h) for h in AUTHORITATIVE_HOSTS)


def validate_record(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Strict import-time checks. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    for name, label in (("title", "标题"), ("issuer", "发布机构"), ("source_type", "来源类型"), ("summary", "简要案情")):
        if not record.get(name):
            errors.append(f"缺少必填项：{label}")
    if record.get("source_type") and record["source_type"] not in SOURCE_TYPES:
        errors.append(f"来源类型必须是以下之一：{'、'.join(SOURCE_TYPES)}")
    if record.get("summary") and len(record["summary"]) < MIN_SUMMARY_CHARS:
        errors.append(f"简要案情少于 {MIN_SUMMARY_CHARS} 字，信息不足以检索和引用")
    for name in ("decision_date", "publish_date"):
        if record.get(name) and not DATE.match(record[name]):
            errors.append(f"{name} 格式应为 2024.05.20")
    url = record.get("source_url", "")
    if record.get("reviewed"):
        if not url or not _host_ok(url):
            errors.append("已核验的案例必须填写政府/法院官方域名的来源网址")
        if not record.get("reviewed_by"):
            errors.append("已核验的案例必须填写核验人")
    else:
        if url and not _host_ok(url):
            errors.append(f"来源网址 {url!r} 不是政府或法院官方域名；没有官方网址时请留空并在来源说明里写明出处")
        if not url and not record.get("provenance_note"):
            errors.append("没有来源网址时必须填写来源说明")
    blob = " ".join(str(record.get(name, "")) for name in TEXT_FIELDS)
    for label, pattern in PRIVACY:
        if pattern.search(blob):
            errors.append(f"文本中含有疑似{label}，请先脱敏")
    if not record.get("holding"):
        warnings.append("缺少裁判要旨，检索结果将只展示案情")
    if not record.get("statutes"):
        warnings.append("没有填写适用法条，无法按法条关联检索")
    return errors, warnings

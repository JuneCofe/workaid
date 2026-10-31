"""Reusable wording for asserting rights: what to say to the employer, the labour inspectorate, the
arbitration commission and legal aid, with the statute behind each point.

Templates are collected and reviewed by people (see data/script_import/README.md); this module only
stores, validates, selects and fills them. Filling is plain substitution: facts the user has given
(dates, pay, arrears) go in, a name or employer goes in only when the user asked to be remembered,
and anything unknown stays as a visible blank. Amounts come from `calc`, never from a model.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .casebase.schema import clean_value, content_hash_of, host_ok, parse_statutes, split_list

AUDIENCES = ("用人单位/HR", "劳动保障监察", "劳动人事争议仲裁委", "法律援助中心", "法院", "工会", "调解组织", "其他")
CHANNELS = ("当面沟通", "电话", "书面/微信/邮件", "线上申请", "现场递交")
SOURCE_KINDS = ("官方发布资料整理", "官方新闻稿整理", "律师审核提供", "系统起草（待审核）")
SCENARIOS = ("wage_arrears", "no_contract", "illegal_dismissal", "overtime", "work_injury", "social_insurance", "general")
PLACEHOLDERS = ("姓名", "单位名称", "常住地址", "入职日期", "离职日期", "月薪", "欠薪月数", "欠薪金额", "所在地", "今天日期")
SENSITIVE_PLACEHOLDERS = {"姓名": "姓名", "单位名称": "单位名称", "常住地址": "常住地址"}
_PLACEHOLDER = re.compile(r"\{([^{}]{1,12})\}")
TEXT_FIELDS = ("title", "scenario_note", "audience", "channel", "body", "source_kind", "issuer", "source_url",
               "source_note", "publish_date", "reviewed_by")
MIN_BODY_CHARS = 30
_LOCK = threading.Lock()
_CACHE: dict[Path, tuple[int, list[dict[str, Any]]]] = {}


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {name: clean_value(raw.get(name, ""))[:6000] for name in TEXT_FIELDS}
    record["publish_date"] = record["publish_date"].replace("-", ".")
    record["scenarios"] = [s for s in split_list(raw.get("scenarios", [])) if s]
    record["tips"] = split_list(raw.get("tips", []))
    record["warnings"] = split_list(raw.get("warnings", []))
    record["statutes"] = [f"{law}|{article}" for law, article in parse_statutes(raw.get("statutes", []))]
    record["reviewed"] = str(raw.get("reviewed", "")).strip().lower() in {"true", "1", "是", "yes", "y"}
    basis = record["title"] + "|" + record["audience"] + "|" + record["body"][:80]
    record["id"] = clean_value(raw.get("id", "")) or content_hash_of(basis)[:12]
    record["content_hash"] = content_hash_of(json.dumps({k: record[k] for k in sorted(record) if k != "content_hash"}, ensure_ascii=False))
    return record


def placeholders_in(body: str) -> list[str]:
    return list(dict.fromkeys(_PLACEHOLDER.findall(body)))


def validate_record(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for name, label in (("title", "标题"), ("audience", "沟通对象"), ("channel", "沟通渠道"), ("body", "话术正文"), ("source_kind", "来源类型")):
        if not record.get(name):
            errors.append(f"缺少必填项：{label}")
    if record["audience"] and record["audience"] not in AUDIENCES:
        errors.append(f"沟通对象必须是：{'、'.join(AUDIENCES)}")
    if record["channel"] and record["channel"] not in CHANNELS:
        errors.append(f"沟通渠道必须是：{'、'.join(CHANNELS)}")
    if record["source_kind"] and record["source_kind"] not in SOURCE_KINDS:
        errors.append(f"来源类型必须是：{'、'.join(SOURCE_KINDS)}")
    if not record["scenarios"]:
        errors.append("必须填写适用场景")
    bad = [s for s in record["scenarios"] if s not in SCENARIOS]
    if bad:
        errors.append(f"适用场景无效：{'、'.join(bad)}（可选：{'、'.join(SCENARIOS)}）")
    if record["body"] and len(record["body"]) < MIN_BODY_CHARS:
        errors.append(f"话术正文少于 {MIN_BODY_CHARS} 字")
    unknown = [p for p in placeholders_in(record["body"]) if p not in PLACEHOLDERS]
    if unknown:
        errors.append(f"正文含不支持的占位符：{'、'.join('{' + p + '}' for p in unknown)}（支持：{'、'.join(PLACEHOLDERS)}）")
    if record["publish_date"] and not re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", record["publish_date"]):
        errors.append("发布日期格式应为 2024.05.20")
    url = record["source_url"]
    if url and not host_ok(url):
        errors.append(f"来源网址 {url!r} 不是政府或法院官方域名；没有官方网址时请留空并写来源说明")
    if record["source_kind"] in ("官方发布资料整理", "官方新闻稿整理") and not (url or record["source_note"]):
        errors.append("来源于官方资料时必须填写来源网址或来源说明")
    if record["reviewed"]:
        if not record["reviewed_by"]:
            errors.append("已审核的话术必须填写审核人")
        if record["source_kind"] == "系统起草（待审核）":
            errors.append("来源类型为“系统起草（待审核）”的话术不能直接标记为已审核；审核后请改为“律师审核提供”并填写审核人")
    if not record["statutes"]:
        warnings.append("没有填写对应法条，用户无法知道这句话的法律依据")
    if not record["warnings"]:
        warnings.append("没有填写“不要这样做”的提醒")
    return errors, warnings


def load(path: Path) -> list[dict[str, Any]]:
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return []
    with _LOCK:
        cached = _CACHE.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                records.append(normalize_record(raw))
        _CACHE[path] = (mtime, records)
        return records


def clear_cache() -> None:
    _CACHE.clear()


def select(path: Path, scenarios: list[str], limit: int = 2, only_reviewed: bool = False) -> list[dict[str, Any]]:
    """Templates for the given scenarios, most specific first, one per audience before repeats."""
    wanted = [s for s in dict.fromkeys(scenarios) if s]
    scored = []
    for record in load(path):
        if only_reviewed and not record["reviewed"]:
            continue
        rank = next((i for i, s in enumerate(wanted) if s in record["scenarios"]), None)
        if rank is None:
            continue
        # Reviewed first, then the order the audiences are listed in (employer before agencies), so the
        # result does not depend on how the file happens to be sorted.
        scored.append((rank, 0 if record["reviewed"] else 1, AUDIENCES.index(record["audience"]), record["audience"], record))
    scored.sort(key=lambda row: row[:3])
    chosen, audiences = [], set()
    for *_, audience, record in scored:
        if audience in audiences:
            continue
        audiences.add(audience)
        chosen.append(record)
        if len(chosen) >= limit:
            break
    return chosen


def _month(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", value)
    if not match:
        return value
    year, month, day = int(match.group(1)), int(match.group(2)), match.group(3)
    return f"{year}年{month}月" + (f"{int(day)}日" if day else "")


def fill_values(known: dict[str, Any], explicit: dict[str, str], today: str) -> dict[str, str]:
    """Placeholder values from the case file and from items the user asked to be remembered."""
    values: dict[str, str] = {"今天日期": today}
    if known.get("entry_date"):
        values["入职日期"] = _month(str(known["entry_date"]))
    if known.get("exit_date") and known["exit_date"] != "未离职":
        values["离职日期"] = _month(str(known["exit_date"]))
    if known.get("monthly_pay"):
        values["月薪"] = f"{int(known['monthly_pay'])}元"
    if known.get("arrears_months"):
        values["欠薪月数"] = str(known["arrears_months"])
        if known.get("monthly_pay"):
            values["欠薪金额"] = f"{int(known['arrears_months']) * int(known['monthly_pay'])}元"
    if known.get("region"):
        values["所在地"] = str(known["region"])
    for placeholder, key in SENSITIVE_PLACEHOLDERS.items():
        if explicit.get(key):
            values[placeholder] = explicit[key]
    return values


def render(record: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    unfilled = []

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        if name not in unfilled:
            unfilled.append(name)
        return f"【请填写：{name}】"

    return {
        "id": record["id"], "title": record["title"], "audience": record["audience"], "channel": record["channel"],
        "text": _PLACEHOLDER.sub(substitute, record["body"]), "unfilled": unfilled,
        "tips": record["tips"], "warnings": record["warnings"], "statutes": record["statutes"],
        "issuer": record["issuer"], "source_url": record["source_url"], "publish_date": record["publish_date"],
        "review_status": "已审核" if record["reviewed"] else "草稿，未经律师审核",
    }


def render_section(rendered: list[dict[str, Any]]) -> str:
    """Markdown block for the answer; every script shows who it is for, its basis and its review status."""
    if not rendered:
        return ""
    lines = ["### 可以直接使用的话术"]
    for item in rendered:
        lines.append("")
        lines.append(f"**{item['title']}**（对象：{item['audience']}；方式：{item['channel']}；{item['review_status']}）")
        lines += [f"> {line}" for line in item["text"].splitlines() if line.strip()]
        if item["unfilled"]:
            lines.append("还需要您补充：" + "、".join(item["unfilled"]))
        if item["statutes"]:
            lines.append("依据：" + "、".join(f"《{s.split('|')[0]}》{s.split('|')[1]}" for s in item["statutes"]))
        lines += [f"- 提示：{tip}" for tip in item["tips"]]
        lines += [f"- 不要这样做：{warning}" for warning in item["warnings"]]
        source = "，".join(p for p in (item["issuer"], item["publish_date"], item["source_url"]) if p)
        if source:
            lines.append(f"来源：{source}")
        if item["review_status"] != "已审核":
            lines.append("*这份话术是草稿，未经律师审核，发送前请自行核对事实，必要时先咨询律师。*")
    return "\n".join(lines)

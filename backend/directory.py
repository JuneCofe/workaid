"""Directory of the bodies a worker can turn to, and the step-by-step route built from it.

Entries describe where to go, what they handle, what to bring, and which time limits apply. National
hotlines and the kinds of local body (inspectorate, arbitration commission, legal aid centre) ship as
a seed; city- and district-level entries with addresses and phone numbers are added by people and
carry their official source. Nothing here is fetched at runtime, so an out-of-date entry is a data
problem fixed by re-importing, and every entry shows whether it has been verified.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .casebase.schema import clean_value, content_hash_of, host_ok, split_list

KINDS = ("热线", "劳动保障监察", "劳动人事争议仲裁", "法律援助", "人民法院", "工会", "妇联", "检察机关", "调解组织", "信访", "其他")
LEVELS = ("national", "province", "city", "district")
SCENARIOS = ("wage_arrears", "no_contract", "illegal_dismissal", "overtime", "work_injury", "social_insurance", "general")
TEXT_FIELDS = ("name", "kind", "level", "region", "what", "phone", "address", "hours", "online", "conditions", "notes",
               "source_url", "source_note", "retrieved_at", "reviewed_by")
_LOCK = threading.Lock()
_CACHE: dict[Path, tuple[int, list[dict[str, Any]]]] = {}


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {name: clean_value(raw.get(name, ""))[:1500] for name in TEXT_FIELDS}
    record["level"] = record["level"] or ("city" if record["region"] else "national")
    record["handles"] = [s for s in split_list(raw.get("handles", [])) if s]
    record["materials"] = split_list(raw.get("materials", []))
    record["statutes"] = split_list(raw.get("statutes", []))
    record["reviewed"] = str(raw.get("reviewed", "")).strip().lower() in {"true", "1", "是", "yes", "y"}
    basis = f"{record['name']}|{record['kind']}|{record['region']}"
    record["id"] = clean_value(raw.get("id", "")) or content_hash_of(basis)[:12]
    record["content_hash"] = content_hash_of(json.dumps({k: record[k] for k in sorted(record) if k != "content_hash"}, ensure_ascii=False))
    return record


def validate_record(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for name, label in (("name", "名称"), ("kind", "类别"), ("what", "受理事项说明")):
        if not record.get(name):
            errors.append(f"缺少必填项：{label}")
    if record["kind"] and record["kind"] not in KINDS:
        errors.append(f"类别必须是：{'、'.join(KINDS)}")
    if record["level"] not in LEVELS:
        errors.append(f"层级必须是：{'、'.join(LEVELS)}")
    if record["level"] != "national" and not record["region"]:
        errors.append("省市区级条目必须填写所在地区")
    bad = [s for s in record["handles"] if s not in SCENARIOS]
    if not record["handles"] or bad:
        errors.append(f"受理场景必须填写且只能是：{'、'.join(SCENARIOS)}" + (f"（无效：{'、'.join(bad)}）" if bad else ""))
    if record["phone"] and not re.fullmatch(r"[\d\-－—+()（）/、,，;； 转分机或]{3,60}", record["phone"]):
        errors.append("电话格式不正确，只能包含数字、连字符、顿号等")
    if record["source_url"] and not host_ok(record["source_url"]):
        errors.append(f"来源网址 {record['source_url']!r} 不是政府或法院官方域名")
    if record["reviewed"] and not (record["source_url"] and record["reviewed_by"]):
        errors.append("已核验的条目必须填写官方来源网址和核验人")
    if not record["source_url"] and not record["source_note"]:
        errors.append("必须填写来源网址或来源说明")
    if record["level"] != "national" and not (record["phone"] or record["address"] or record["online"]):
        warnings.append("没有电话、地址或线上入口，用户无法联系到这个机构")
    if not record["materials"]:
        warnings.append("没有填写所需材料")
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


def _region_matches(entry_region: str, user_region: str) -> bool:
    return bool(entry_region and user_region and (entry_region in user_region or user_region in entry_region))


def lookup(path: Path, scenario: str, region: str = "", kinds: tuple[str, ...] = (), phones: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Entries for a scenario: the user's own region first, then national ones.

    With `kinds` and/or `phones`, an entry must be of one of those kinds, or be a hotline with one of
    those numbers, so a complaints step lists the inspectorate and 12333 rather than every hotline. The
    number only counts for hotlines: a commission that happens to publish 12333 is not a hotline.
    """
    local, national = [], []
    for record in load(path):
        if scenario not in record["handles"] and "general" not in record["handles"]:
            continue
        if (kinds or phones) and record["kind"] not in kinds and not (record["kind"] == "热线" and record["phone"] in phones):
            continue
        if record["level"] == "national":
            national.append(record)
        elif _region_matches(record["region"], region):
            local.append(record)
    return local + national


def describe(record: dict[str, Any]) -> str:
    """One line: who, how to reach them, what to bring."""
    parts = [record["name"]]
    reach = "，".join(p for p in (
        f"电话 {record['phone']}" if record["phone"] else "", f"地址 {record['address']}" if record["address"] else "",
        f"办公时间 {record['hours']}" if record["hours"] else "", f"线上 {record['online']}" if record["online"] else "") if p)
    if reach:
        parts.append(f"（{reach}）")
    if record["materials"]:
        parts.append("；需带：" + "、".join(record["materials"]))
    if record["conditions"]:
        parts.append("；条件：" + record["conditions"])
    if not record["reviewed"]:
        parts.append("；【联系方式未经核验，请以官方公布为准】")
    return "".join(parts)


# --------------------------------------------------------------------------------------- the route

def _step(number: int, title: str, detail: str, entries: list[dict[str, Any]], basis: list[str] | None = None,
          if_none: str = "") -> dict[str, Any]:
    """`if_none` is shown when the step should name a body but the directory has none to offer."""
    lines = [describe(record) for record in entries[:3]]
    return {
        "step": str(number), "title": title,
        "detail": detail + (if_none if not lines else " 可联系：" + "；".join(lines)),
        "institutions": [{"id": r["id"], "name": r["name"], "kind": r["kind"], "phone": r["phone"], "reviewed": r["reviewed"]} for r in entries[:3]],
        "basis": basis or [],
    }


EVIDENCE_BY_SCENARIO = {
    "wage_arrears": "工资条、银行流水、考勤记录、催要工资的聊天记录或录音、劳动合同",
    "no_contract": "工牌/工作证、工资发放记录、考勤、工作安排的聊天记录、同事证明、招聘信息",
    "illegal_dismissal": "解除或辞退通知（没有书面的就保留聊天记录和录音）、劳动合同、工资流水、社保记录",
    "overtime": "考勤/打卡记录、加班审批或工作安排的聊天记录、工资条（看加班费有没有发）",
    "work_injury": "受伤经过的证明、医院病历和诊断证明、医疗费票据、工作证件、同事证明",
    "social_insurance": "社保缴费查询记录（对比工资发放）、劳动合同、工资流水",
    "general": "劳动合同、工资流水、考勤、工作聊天记录、解除或处理决定",
}


def build_path(path: Path, scenario: str, region: str = "", deadline_note: str = "", only_reviewed: bool = False) -> list[dict[str, Any]]:
    """Ordered steps for a labour dispute, each with the bodies to contact for the user's region."""
    scenario = scenario if scenario in SCENARIOS else "general"

    def find(*kinds: str, phones: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        entries = lookup(path, scenario, region, kinds, phones)
        return [e for e in entries if e["reviewed"]] if only_reviewed else entries

    none = (
        f" （暂未收录{region or '您所在地区'}的{'已核验' if only_reviewed else ''}机构信息，"
        "请通过当地人力资源和社会保障、法院或司法行政部门的官方渠道查询。）"
    )

    steps = [_step(1, "固定事实与证据", f"先保存这些材料的原件或截图：{EVIDENCE_BY_SCENARIO[scenario]}。按时间顺序整理一份简短的经过。", [])]
    if scenario == "work_injury":
        steps.append(_step(2, "申请工伤认定", "单位应在事故发生后 30 日内申请；单位不申请的，本人或亲属可在一年内自己申请。" + deadline_note,
                           find("劳动保障监察", phones=("12333",)), ["《工伤保险条例》第十七条"], none))
        steps.append(_step(3, "劳动能力鉴定与待遇", "治疗稳定后申请劳动能力鉴定，再按伤残等级主张医疗、伤残和停工留薪期待遇。", [], ["《工伤保险条例》第二十一条"]))
    else:
        steps.append(_step(2, "书面主张，留下记录", "用微信、邮件或书面函件向单位明确提出诉求并限定答复时间，保留发送和回复记录，不要只在电话里说。", []))
        if scenario in ("wage_arrears", "no_contract", "social_insurance", "overtime"):
            steps.append(_step(3, "向劳动保障监察部门投诉", "劳动者认为单位侵犯劳动保障权益的，可以向用人单位用工所在地的劳动保障行政部门投诉，一般不收费。" + deadline_note,
                               find("劳动保障监察", phones=("12333",)), ["《劳动保障监察条例》第九条", "《劳动保障监察条例》第十三条"], none))
    number = len(steps) + 1
    steps.append(_step(number, "申请劳动仲裁", "劳动争议要先仲裁后才能起诉；仲裁不收费，时效一般为一年，由劳动合同履行地或用人单位所在地的仲裁委管辖。" + deadline_note,
                       find("劳动人事争议仲裁", "调解组织"), ["《劳动争议调解仲裁法》第二十一条", "《劳动争议调解仲裁法》第二十七条", "《劳动争议调解仲裁法》第五十三条"], none))
    steps.append(_step(number + 1, "不服裁决再起诉", "对一般劳动争议的仲裁裁决不服，可在收到裁决书之日起十五日内向人民法院起诉；期满不起诉裁决即生效。",
                       find("人民法院", phones=("12368",)), ["《劳动争议调解仲裁法》第五十条"], none))
    steps.append(_step(number + 2, "经济困难可申请法律援助", "请求支付劳动报酬、请求社会保险待遇、工伤事故人身损害赔偿等，因经济困难没有委托代理人的，可以向法律援助机构申请。",
                       find("法律援助", phones=("12348",)), ["《法律援助法》第三十一条"], none))
    return steps

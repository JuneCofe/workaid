"""Case file ("案情档案") and follow-up questions.

Per conversation the agent keeps a structured record of the dispute: the case type, the facts that
decide the outcome (dates, pay, contract and insurance status, evidence...), and which of them are
still missing. Facts are filled automatically from what the user said and from the long-term profile;
at most three missing, high-value questions are asked per turn, and the same question is not repeated
more than twice. Figures that depend on these facts come from `calc`, never from a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from . import calc, privacy
from .profile import PROCEDURAL, extract_facts
from .retrieval.query import match_concepts

MAX_QUESTIONS = 3
MAX_ASKS = 2
ONGOING = calc.ONGOING


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    question: str
    kind: str  # date | amount | number | choice | multi | text
    why: str
    options: tuple[str, ...] = ()


FIELDS: dict[str, Field] = {f.key: f for f in (
    Field("region", "工作所在地", "您在哪个城市工作？", "text", "决定向哪里的仲裁委、监察部门和法院求助，以及适用的地方规定。"),
    Field("entry_date", "入职时间", "您是什么时候入职的？（精确到年月即可）", "date", "决定二倍工资的月数、经济补偿的年限和各类期限。"),
    Field("exit_date", "离职/被辞退时间", "您的劳动关系是什么时候结束的？还没结束请填“未离职”。", "date", "仲裁时效从这里起算，也决定补偿年限。"),
    Field("monthly_pay", "月薪", "您的月工资大约是多少？（税前，含固定补贴）", "amount", "用来估算欠薪、二倍工资和经济补偿的金额。"),
    Field("has_contract", "书面劳动合同", "单位有没有和您签书面劳动合同？", "choice", "没签合同可能涉及二倍工资。",
          ("没有签书面劳动合同", "签了书面劳动合同", "不确定")),
    Field("has_insurance", "社会保险", "单位有没有给您缴社保？", "choice", "未缴社保可以投诉，也是解除合同要求补偿的理由。",
          ("单位没有缴社保", "单位缴了社保", "不确定")),
    Field("arrears_months", "拖欠月数", "公司一共拖欠了您几个月的工资？", "number", "用来估算欠薪金额。"),
    Field("evidence", "已有证据", "您手上有哪些证据？", "multi", "证据决定能主张什么，也决定先补哪些材料。",
          ("聊天记录", "考勤记录", "工资凭证", "录音录像", "劳动合同", "工作证件", "证人", "医疗凭证", "解除通知", "都没有")),
    Field("dismissal_form", "解除方式", "劳动关系是怎么结束的？", "choice", "区分违法解除、合同到期、被迫辞职等，适用的条文和补偿不同。",
          ("公司单方辞退/开除", "合同到期公司不续签", "公司有过错我被迫辞职", "协商一致解除", "试用期被辞退", "我自己辞职")),
    Field("overtime_hours", "每周加班小时数", "您平均每周加班多少小时？", "number", "用来判断加班费的大致范围。"),
    Field("injury_date", "受伤时间", "您是什么时候受伤的？（精确到年月日更好）", "date", "工伤认定有申请期限，需要从受伤当天起算。"),
    Field("injury_recognition", "工伤认定进展", "工伤认定办到哪一步了？", "choice", "决定下一步是申请认定、等待结果还是主张待遇。",
          ("还没有申请认定", "已申请，等结果", "已认定为工伤", "单位不给申请/不认定")),
)}


@dataclass(frozen=True)
class CaseType:
    key: str
    label: str
    required: tuple[str, ...]
    formal: str


CASE_TYPES: dict[str, CaseType] = {t.key: t for t in (
    CaseType("wage_arrears", "拖欠工资", ("arrears_months", "monthly_pay", "entry_date", "region", "has_contract", "evidence"),
             "拖欠劳动报酬 未及时足额支付劳动报酬 加付赔偿金"),
    CaseType("no_contract", "未签劳动合同", ("entry_date", "monthly_pay", "has_contract", "exit_date", "region", "evidence"),
             "未订立书面劳动合同 二倍工资 自用工之日起一个月内订立书面劳动合同"),
    CaseType("illegal_dismissal", "解除劳动合同/辞退", ("dismissal_form", "entry_date", "exit_date", "monthly_pay", "region", "has_contract", "evidence"),
             "违法解除劳动合同 赔偿金 经济补偿 用人单位单方解除"),
    CaseType("overtime", "加班", ("monthly_pay", "overtime_hours", "region", "evidence", "entry_date"),
             "延长工作时间 加班工资 休息日 法定休假日"),
    CaseType("work_injury", "工伤", ("injury_date", "injury_recognition", "region", "has_insurance", "entry_date"),
             "工伤认定 工伤保险待遇 劳动能力鉴定 停工留薪期"),
    CaseType("social_insurance", "社会保险", ("has_insurance", "entry_date", "region", "monthly_pay"),
             "用人单位未依法缴纳社会保险费 补缴社会保险费 社会保险登记"),
    CaseType("general", "劳动争议", ("region", "entry_date", "has_contract"), ""),
)}
CONCEPT_TO_TYPE = {
    # "migrant_wages" (工地, 包工头, 农民工) is deliberately not mapped: those words say who the worker is, not what
    # went wrong. A worker on a site who was hurt is an injury case; unpaid wages are caught by "wage_arrears".
    "wage_arrears": "wage_arrears", "wage_deduction": "wage_arrears",
    "no_contract": "no_contract", "illegal_dismissal": "illegal_dismissal", "compensation": "illegal_dismissal",
    "contract_end": "illegal_dismissal", "resignation": "illegal_dismissal", "probation": "illegal_dismissal",
    "pregnancy": "illegal_dismissal", "overtime": "overtime", "work_injury": "work_injury", "social_insurance": "social_insurance",
}
# When a message touches several disputes, the first one listed here becomes the case type.
TYPE_PRIORITY = ("work_injury", "illegal_dismissal", "no_contract", "overtime", "social_insurance", "wage_arrears")

_NUM = r"(\d+|[一二两三四五六七八九十]+)"
_ARREARS = (
    re.compile(rf"(?:拖欠|欠|没发|未发|不发)[^，。；\d]{{0,6}}?{_NUM}\s*个?月(?:的)?(?:工资|薪)"),
    re.compile(rf"{_NUM}\s*个?月(?:的)?(?:工资|薪水)(?:都)?(?:没|未|不)(?:发|给|结)"),
    # "欠了四个月", "拖欠工资两个月": the amount of time does not have to touch the word 工资.
    re.compile(rf"(?:拖欠|欠)了?\s*{_NUM}\s*个月"),
    re.compile(rf"(?:拖欠|欠)[^，。；\d]{{0,4}}?(?:工资|薪水|薪资)[^，。；\d]{{0,4}}?{_NUM}\s*个月"),
)
_DISMISSAL = (
    (re.compile(r"被迫|逼我(?:辞职|离职)|不得不(?:辞职|离职)"), "公司有过错我被迫辞职"),
    (re.compile(r"试用期[^。；]{0,24}(?:辞退|不合格|开除|解除|走人|让我走)"), "试用期被辞退"),
    (re.compile(r"到期|期满|不续(?:签|约)|合同终止"), "合同到期公司不续签"),
    (re.compile(r"协商一致|协议解除|协商解除|协商离职"), "协商一致解除"),
    (re.compile(r"辞退|开除|解雇|裁员|劝退|炒了|被裁|让我走人"), "公司单方辞退/开除"),
    (re.compile(r"我(?:主动)?(?:提出)?(?:辞职|离职)了?|自己辞职"), "我自己辞职"),
)
_OVERTIME = (
    re.compile(r"每(?:周|星期)[^，。\d]{0,4}加班(?:大概|大约|约)?\s*(\d+)\s*(?:个)?小时"),
    re.compile(r"每天加班(?:大概|大约|约)?\s*(\d+)\s*(?:个)?小时"),
)
_INJURY_WORDS = ("受伤", "事故", "出事", "摔", "砸", "烫", "压断", "切断", "压伤", "夹伤", "割伤", "撞伤", "扎伤", "工伤")
_INJURY_RECOGNITION = (
    (re.compile(r"没(?:有)?认定|不给(?:我)?(?:申请|认定)|不(?:肯|愿意)?申请工伤|拒绝申请"), "单位不给申请/不认定"),
    (re.compile(r"已(?:经)?认定(?:为)?工伤|认定工伤了|认定为工伤|工伤认定(?:书|决定)"), "已认定为工伤"),
    (re.compile(r"申请(?:了)?工伤认定|工伤认定申请|已申请"), "已申请，等结果"),
)
_DATE = re.compile(r"(\d{4})\s*(?:年|[./-])\s*(\d{1,2})\s*(?:月份?|[./-]\s*(\d{1,2})|月)?(?:\s*(\d{1,2})\s*[日号])?")


def _to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if token == "十":
        return 10
    if len(token) == 1:
        return digits.get(token)
    if len(token) == 2 and token[0] == "十":
        return 10 + digits.get(token[1], 0)
    if len(token) == 2 and token[1] == "十":
        return digits.get(token[0], 0) * 10
    return None


def normalize_date(text: str) -> str | None:
    match = _DATE.search(text or "")
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    day = match.group(3) or match.group(4)
    value = f"{year}-{month:02d}" + (f"-{int(day):02d}" if day else "")
    return value if calc.parse_date(value) else None


def validate_answer(field: Field, raw: Any) -> tuple[Any, str]:
    """(normalised value, error). An empty error means the answer was accepted."""
    if field.kind == "date":
        if str(raw).strip() == ONGOING and field.key == "exit_date":
            return ONGOING, ""
        value = normalize_date(str(raw))
        return (value, "") if value else (None, f"{field.label}需要写成 2024-03 或 2024年3月15日 这样的日期")
    if field.kind == "amount":
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(万|千|k|K)?\s*(?:元|块)?\s*", str(raw))
        if not match:
            return None, f"{field.label}请填数字，例如 8000"
        amount = float(match.group(1)) * {"万": 10000, "千": 1000, "k": 1000, "K": 1000}.get(match.group(2) or "", 1)
        return (int(amount), "") if 500 <= amount <= 1_000_000 else (None, f"{field.label}数额看起来不合理，请检查")
    if field.kind == "number":
        match = re.fullmatch(r"\s*(\d{1,3})\s*", str(raw))
        return (int(match.group(1)), "") if match else (None, f"{field.label}请填整数")
    if field.kind == "choice":
        return (raw, "") if raw in field.options else (None, f"{field.label}请从选项中选择")
    if field.kind == "multi":
        chosen = raw if isinstance(raw, list) else [raw]
        return (list(dict.fromkeys(chosen)), "") if chosen and all(c in field.options for c in chosen) else (None, f"{field.label}请从选项中选择")
    return privacy.redact_text(str(raw).strip())[:40] or None, ""


def _dialogue_fields(text: str) -> dict[str, Any]:
    """Case fields stated in the user's words. Extraction is rule-based; nothing is inferred."""
    found: dict[str, Any] = {}
    facts = extract_facts(text)
    by_key: dict[str, list[str]] = {}
    for fact in facts:
        by_key.setdefault(fact.key, []).append(fact.value)
    if "所在地" in by_key:
        found["region"] = by_key["所在地"][0]
    if "入职时间" in by_key:
        found["entry_date"] = by_key["入职时间"][-1]
    if "离职时间" in by_key:
        found["exit_date"] = by_key["离职时间"][-1]
    if "月薪" in by_key:
        found["monthly_pay"] = int(by_key["月薪"][-1].rstrip("元"))
    if "劳动合同" in by_key:
        found["has_contract"] = by_key["劳动合同"][-1].replace("无书面劳动合同", "没有签书面劳动合同").replace("有书面劳动合同", "签了书面劳动合同")
    if "社保" in by_key:
        found["has_insurance"] = by_key["社保"][-1].replace("单位未缴社保", "单位没有缴社保").replace("单位已缴社保", "单位缴了社保")
    if "已有证据" in by_key:
        found["evidence"] = list(dict.fromkeys(by_key["已有证据"]))
    for pattern in _ARREARS:
        if (match := pattern.search(text)) and (months := _to_int(match.group(1))) and 0 < months <= 60:
            found["arrears_months"] = months
            break
    for pattern, label in _DISMISSAL:
        if pattern.search(text):
            found["dismissal_form"] = label
            break
    for pattern in _OVERTIME:
        if match := pattern.search(text):
            hours = int(match.group(1))
            found["overtime_hours"] = hours * 5 if "每天" in match.group(0) else hours
            break
    for match in _DATE.finditer(text):
        around = text[max(0, match.start() - 12) : match.end() + 16]
        if any(word in around for word in _INJURY_WORDS) and (value := normalize_date(match.group(0))):
            found["injury_date"] = value
            break
    for pattern, label in _INJURY_RECOGNITION:
        if pattern.search(text):
            found["injury_recognition"] = label
            break
    return found


_FORCED_RESIGNATION = re.compile(r"被迫|逼我(?:辞职|离职)|不得不(?:辞职|离职)")
# Concepts that only say a worker is leaving or asking about money, not what the dispute is about.
WEAK_CONCEPTS = frozenset({"resignation", "compensation"})


def detect_types(text: str) -> list[str]:
    """Case types the text points to, most specific first.

    Strong types come first, ordered by TYPE_PRIORITY rather than by where a word appears ("工地上摔伤了，
    老板还欠着工资" is an injury case whatever word comes earlier). A type that rests only on "辞职" or
    "补偿" comes after them: "离职了，公司还欠我工资" is an arrears case, not a dismissal.
    """
    strong: set[str] = set()
    weak: set[str] = set()
    for concept in match_concepts(text):
        if concept.key in PROCEDURAL:
            continue
        case_type = CONCEPT_TO_TYPE.get(concept.key)
        if case_type:
            (weak if concept.key in WEAK_CONCEPTS else strong).add(case_type)
    if _FORCED_RESIGNATION.search(text):
        strong.add("illegal_dismissal")  # a forced resignation is a dismissal dispute
    return [t for t in TYPE_PRIORITY if t in strong] + [t for t in TYPE_PRIORITY if t in weak and t not in strong]


def new_case() -> dict[str, Any]:
    return {"case_type": "", "topics": [], "fields": {}, "asked": {}}


def _setting(case: dict[str, Any], key: str, value: Any, source: str) -> None:
    current = case["fields"].get(key)
    if current and current["source"] == "form" and source != "form":
        return  # an answer given on the questionnaire is never overwritten by a guess from chat text
    case["fields"][key] = {"value": value, "source": source}


def update_case(case: dict[str, Any], user_messages: list[str], profile_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Merge what the user has said (all their messages in this conversation) into the case file."""
    case = {**new_case(), **case, "fields": dict(case.get("fields", {}))}
    text = "\n".join(user_messages)
    latest_types = detect_types(user_messages[-1]) if user_messages else []
    all_types = detect_types(text)
    case["topics"] = list(dict.fromkeys(case.get("topics", []) + all_types))
    if latest_types:
        case["case_type"] = latest_types[0]
    elif not case.get("case_type") and all_types:
        case["case_type"] = all_types[0]
    for key, value in _dialogue_fields(text).items():
        _setting(case, key, value, "dialogue")
    for item in profile_items or []:
        if item["key"] == "所在地" and "region" not in case["fields"] and not item.get("sensitive"):
            case["fields"]["region"] = {"value": item["value"], "source": "profile"}
    return case


def apply_answers(case: dict[str, Any], answers: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge questionnaire answers; returns (case, {field: error}) and applies only valid ones."""
    case = {**case, "fields": dict(case.get("fields", {}))}
    errors: dict[str, str] = {}
    for key, raw in answers.items():
        field = FIELDS.get(key)
        if field is None:
            errors[key] = "未知的字段"
            continue
        value, error = validate_answer(field, raw)
        if error or value in (None, ""):
            errors[key] = error or "不能为空"
            continue
        case["fields"][key] = {"value": value, "source": "form"}
    return case, errors


def type_of(case: dict[str, Any]) -> CaseType | None:
    return CASE_TYPES.get(case.get("case_type", "")) if case.get("case_type") else None


def known(case: dict[str, Any]) -> dict[str, Any]:
    return {key: entry["value"] for key, entry in case["fields"].items() if entry["value"] not in (None, "", [])}


def completeness(case: dict[str, Any]) -> tuple[int, int]:
    case_type = type_of(case)
    if case_type is None:
        return 0, 0
    values = known(case)
    return sum(1 for key in case_type.required if key in values), len(case_type.required)


def next_questions(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Up to three unanswered required questions, never the same one more than twice."""
    case = {**case, "asked": dict(case.get("asked", {}))}
    case_type = type_of(case)
    if case_type is None:
        return [], case
    values = known(case)
    questions = []
    for key in case_type.required:
        if key in values or case["asked"].get(key, 0) >= MAX_ASKS or len(questions) >= MAX_QUESTIONS:
            continue
        field = FIELDS[key]
        questions.append({"key": key, "label": field.label, "question": field.question, "kind": field.kind,
                          "options": list(field.options), "why": field.why})
        case["asked"][key] = case["asked"].get(key, 0) + 1
    return questions, case


def calculations(case: dict[str, Any], today: date | None = None) -> list[calc.Result]:
    """Figures that follow from the known facts for this case type."""
    today = today or date.today()
    values = known(case)
    case_type = case.get("case_type")
    topics = set(case.get("topics", []) + ([case_type] if case_type else []))
    entry, exit_value, pay = values.get("entry_date"), values.get("exit_date"), values.get("monthly_pay")
    results: list[calc.Result | None] = []
    if "wage_arrears" in topics:
        results.append(calc.wage_arrears_remedy(values.get("arrears_months"), pay))
        results.append(calc.arbitration_limit(exit_value, today, arrears_ongoing=exit_value in (None, ONGOING)))
    if "no_contract" in topics or values.get("has_contract") == "没有签书面劳动合同":
        results.append(calc.double_wage(entry, exit_value, pay, today))
    if "illegal_dismissal" in topics:
        unlawful = values.get("dismissal_form") in ("公司单方辞退/开除", "试用期被辞退", "公司有过错我被迫辞职")
        results.append(calc.severance(entry, exit_value, pay, unlawful=unlawful))
        if exit_value and exit_value != ONGOING:
            results.append(calc.arbitration_limit(exit_value, today))
    if "work_injury" in topics:
        results.append(calc.injury_deadlines(values.get("injury_date"), today))
    seen, unique = set(), []
    for result in results:
        if result and result.key not in seen:
            seen.add(result.key)
            unique.append(result)
    return unique


def search_terms(case: dict[str, Any]) -> str:
    """Statute-style phrases for the case, appended to retrieval so short follow-ups keep their subject."""
    phrases = [CASE_TYPES[t].formal for t in [case.get("case_type", "")] + case.get("topics", []) if t in CASE_TYPES and CASE_TYPES[t].formal]
    values = known(case)
    if values.get("has_contract") == "没有签书面劳动合同":
        phrases.append("未订立书面劳动合同 二倍工资")
    if values.get("has_insurance") == "单位没有缴社保":
        phrases.append("用人单位未依法缴纳社会保险费")
    form = values.get("dismissal_form")
    if form == "合同到期公司不续签":
        phrases.append("劳动合同期满终止 经济补偿")
    elif form == "公司有过错我被迫辞职":
        phrases.append("劳动者被迫解除劳动合同 经济补偿")
    return " ".join(dict.fromkeys(" ".join(phrases).split()))


def public_view(case: dict[str, Any], questions: list[dict[str, Any]], results: list[calc.Result]) -> dict[str, Any]:
    case_type = type_of(case)
    done, total = completeness(case)
    return {
        "case_type": case.get("case_type", ""),
        "case_type_label": case_type.label if case_type else "",
        "topics": [CASE_TYPES[t].label for t in case.get("topics", []) if t in CASE_TYPES],
        "fields": [
            {"key": key, "label": FIELDS[key].label, "value": entry["value"], "source": entry["source"]}
            for key, entry in case["fields"].items() if key in FIELDS
        ],
        "missing": [FIELDS[key].label for key in (case_type.required if case_type else ()) if key not in known(case)],
        "completeness": {"known": done, "required": total},
        "followups": questions,
        "calculations": [result.as_dict() for result in results],
    }


def render_section(case: dict[str, Any], questions: list[dict[str, Any]], results: list[calc.Result]) -> str:
    """Markdown for the answer: what is known, computed figures with their basis, and what to supply next."""
    case_type = type_of(case)
    if case_type is None:
        return ""
    lines: list[str] = []
    facts = [f"{FIELDS[k].label}：{'、'.join(v) if isinstance(v, list) else v}" for k, v in known(case).items() if k in FIELDS]
    done, total = completeness(case)
    lines.append(f"### 案情梳理（{case_type.label}，信息完整度 {done}/{total}）")
    lines.append("；".join(facts) if facts else "目前还没有可用的关键信息。")
    for result in results:
        lines.append(f"\n### {result.title}\n{result.summary}")
        lines += [f"- {item}" for item in result.details]
        if result.basis:
            lines.append("依据：" + "、".join(result.basis))
        lines += [f"- 注意：{item}" for item in result.caveats]
    if questions:
        lines.append("\n### 为了给出更准确的建议，请补充")
        for question in questions:
            options = f"（可选：{'、'.join(question['options'])}）" if question["options"] else ""
            lines.append(f"- {question['question']}{options}　*{question['why']}*")
    return "\n".join(lines)


def render_markdown(case: dict[str, Any], results: list[calc.Result]) -> str:
    """A hand-over summary the user can give a lawyer or a legal aid centre."""
    case_type = type_of(case)
    lines = ["# 案情摘要", "", f"案件类型：{case_type.label if case_type else '未确定'}", "", "## 已知事实"]
    for key, entry in case["fields"].items():
        if key in FIELDS and entry["value"] not in (None, "", []):
            value = "、".join(entry["value"]) if isinstance(entry["value"], list) else entry["value"]
            lines.append(f"- {FIELDS[key].label}：{value}（{'本人填写' if entry['source'] == 'form' else '对话中提到' if entry['source'] == 'dialogue' else '来自长期记忆'}）")
    missing = [FIELDS[k].label for k in (case_type.required if case_type else ()) if k not in known(case)]
    lines += ["", "## 还缺少的信息", "、".join(missing) if missing else "（必填信息已齐全）"]
    for result in results:
        lines += ["", f"## {result.title}", result.summary, *[f"- {item}" for item in result.details]]
        lines += ["依据：" + "、".join(result.basis)] if result.basis else []
    lines += ["", "> 本摘要由 WorkAid 根据对话整理，仅供沟通使用，不构成法律意见；金额和期限为估算，请以律师或主管部门的意见为准。"]
    return "\n".join(lines)

"""Build the long-term profile from conversations instead of interviewing the user.

After every turn, facts that matter for a labour dispute (where the person works, the job, pay, contract
and insurance status, evidence in hand, steps already taken, special circumstances, answer preferences)
are extracted from the *redacted* text and merged into `profile_items`; long conversations are
compressed into a short summary that keeps conclusions and key facts. Names, employers and addresses
are stored only when the user explicitly asked to be remembered.

Extraction is rule-based so it works offline; with a model key a second pass can add facts, and its
output is validated against the same whitelist.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from . import privacy
from .privacy import REGIONS
from .retrieval.lexicon import CONCEPTS
from .retrieval.query import match_concepts

KEEP_RECENT = int(os.getenv("CONTEXT_KEEP_RECENT", "10"))
COMPRESS_SLACK = 6
SUMMARY_CHARS = 700
MIN_MONTHLY_PAY = 500
MAX_MONTHLY_PAY = 1_000_000
ALLOWED_KEYS = frozenset({
    "所在地", "行业岗位", "入职时间", "离职时间", "工作时长", "月薪", "劳动合同", "社保", "用工形式", "回答偏好",
    "特殊情形", "争议主题", "已有证据", "已采取行动",
})
PROCEDURAL = {"lawsuit", "arbitration", "evidence", "legal_aid"}
_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_REGION_RE = re.compile("(" + "|".join(sorted(set(REGIONS), key=len, reverse=True)) + ")")

_INDUSTRY = (
    (r"外卖|骑手|送餐|跑腿", "外卖骑手"), (r"快递|分拣|派件", "快递物流"), (r"网约车|滴滴|货拉拉|代驾|货车司机|卡车司机|司机", "司机"),
    (r"工地|建筑|包工头|农民工|瓦工|钢筋|木工|施工", "建筑工地"), (r"工厂|流水线|车间|操作工|普工|制造", "制造业工人"),
    (r"服务员|餐饮|饭店|后厨|厨师|酒店", "餐饮酒店"), (r"销售|业务员|客服|电销", "销售客服"), (r"程序员|开发|互联网|算法|测试工程师|运营", "互联网"),
    (r"保安|保洁|物业|家政|月嫂|保姆", "保安保洁家政"), (r"老师|教师|培训|教育机构", "教育培训"), (r"护士|医生|医院|药店", "医疗"),
    (r"会计|财务|出纳", "财务"), (r"文员|行政|前台|人事|HR", "行政文职"), (r"主播|直播|带货|网红", "直播"),
)
_FORM = (
    (r"外卖|骑手|滴滴|网约车|主播|带货|众包|平台", "新业态"), (r"劳务派遣|派遣工|派遣公司", "劳务派遣"), (r"实习生?|实习期", "实习"),
    (r"兼职|临时工|钟点工|零工|小时工", "兼职临时"), (r"退休返聘|返聘|超龄|已退休", "超龄/返聘"),
)
_SPECIAL = (
    (r"怀孕|孕期|孕妇|怀了孩子|待产", "孕期"), (r"哺乳期|喂奶|产后", "哺乳期"), (r"产假", "产假"), (r"未成年|童工|不满十八", "未成年"),
    (r"工伤|职业病|伤残|停工留薪", "工伤/职业病"), (r"患病|生病|住院|病假|医疗期|得了.{0,2}病", "患病/医疗期"), (r"农民工|外来务工", "农民工"),
    (r"残疾|残联", "残疾人"),
)
_EVIDENCE = (
    (r"聊天记录|微信记录|微信聊天|钉钉|短信", "聊天记录"), (r"考勤|打卡", "考勤记录"), (r"工资条|工资单|银行流水|转账记录", "工资凭证"),
    (r"录音|录像", "录音录像"),
    (r"(?<![没未无])有(?:书面)?劳动合同|(?:劳动)?合同(?:原件|复印件)", "劳动合同"),
    (r"工牌|工作证|工服|入职登记", "工作证件"),
    (r"监控", "监控"), (r"证人|同事(?:能)?证明", "证人"), (r"病历|诊断证明|医疗费|发票", "医疗凭证"), (r"解除通知|辞退通知|离职证明", "解除通知"),
)
_ACTION = (
    (r"(?:已经|已)?(?:申请|提起|去)了?(?:劳动)?仲裁|仲裁委", "已申请仲裁"), (r"(?:投诉|举报)了?|12333|劳动监察|监察大队", "已投诉举报"),
    (r"报了?警|报警|派出所", "已报警"), (r"(?:已经|已)?起诉|上了?法庭|开庭|立案", "已起诉"), (r"协商|谈过|沟通过|找过(?:老板|公司|领导|HR)", "已协商"),
    (r"调解", "已调解"),
)
_PREFERENCE = (
    (r"简单点|简短|别太长|一句话|长话短说|概括", "简洁"), (r"详细(?:点|说|一点)|展开|具体点|说清楚", "详细"),
    (r"通俗|大白话|看不懂|听不懂|白话", "通俗"),
)
_CONTRACT_NO = re.compile(
    r"没(?:有)?(?:跟我|和我|给我)?签(?:订)?(?:过)?(?:书面)?(?:劳动)?合同|未签(?:订)?(?:书面)?劳动合同|不(?:肯|给|跟|和)?(?:我)?签(?:订)?(?:书面)?(?:劳动)?合同"
    r"|[没未无]有?(?:书面)?(?:的)?劳动合同|(?:书面)?劳动合同(?:都)?(?:没有|没签|未签)"
)
_CONTRACT_YES = re.compile(r"签(?:订)?了(?:书面)?(?:劳动)?合同|(?<![没未无])有(?:书面)?劳动合同|已(?:经)?签(?:订)?(?:劳动)?合同")
_INSURANCE_NO = re.compile(r"没(?:有)?(?:给我)?(?:交|缴纳?|买)(?:过)?社保|未(?:缴|交)(?:纳)?社保|不(?:给我)?(?:交|缴纳?)社保|没有社保")
_INSURANCE_YES = re.compile(r"(?:交|缴纳?)了社保|有社保|社保(?:一直)?都?在交")
_PAY = re.compile(
    r"(?:(?:月薪|月工资|底薪|基本工资|月收入|薪水)(?:是|为|大概|大约|约|有|才|只有)?|(?:工资|薪资)(?:是|为|一个月|每个月|每月)|每个?月(?:工资|拿|挣)?(?:大概|大约|约)?)"
    r"\s*(?P<num>\d+(?:\.\d+)?|[一二两三四五六七八九十]+(?:[万千百][一二三四五六七八九]?)?)\s*(?P<unit>万|千|k|K|块|元)?"
)
_DURATION = re.compile(
    r"(?:干了|工作了|上班|做了|在职|入职了?|待了|呆了|在(?:这家)?公司(?:干|做|工作)了?)\s*(?:大概|差不多|将近|快|整整|近|约|有)?\s*"
    r"(?P<num>\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)\s*(?P<unit>个月|年|周|天)"
)
_DATE = re.compile(r"(?P<y>\d{4})\s*(?:年|[./-])\s*(?P<m>\d{1,2})\s*(?:月份?|[./-]\s*(?P<d>\d{1,2})|(?:月|[./-]))?(?:\s*(?P<d2>\d{1,2})\s*[日号])?")
_ENTRY_WORDS = ("入职", "进厂", "进公司", "开始上班", "开始工作", "来公司", "到岗")
_EXIT_WORDS = ("离职", "辞退", "开除", "解雇", "被裁", "裁员", "劝退", "终止", "走人", "辞职", "解除")


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    confidence: float = 0.7


def _number(token: str) -> float | None:
    token = token.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    if token == "半":
        return 0.5
    match = re.fullmatch(r"([一二两三四五六七八九十]+)(万|千|百)?([一二三四五六七八九])?", token)
    if not match:
        return None
    head, unit, tail = match.groups()
    base = 10 if head == "十" else sum(_CN[ch] for ch in head) if len(head) == 1 else None
    if base is None:
        return None
    scale = {"万": 10000, "千": 1000, "百": 100, None: 1}[unit]
    value = base * scale
    if tail:
        value += _CN[tail] * scale // 10
    return float(value)


def _pay(text: str) -> int | None:
    for match in _PAY.finditer(text):
        number = _number(match.group("num"))
        if number is None:
            continue
        unit = match.group("unit") or ""
        amount = number * {"万": 10000, "千": 1000, "k": 1000, "K": 1000}.get(unit, 1)
        if re.fullmatch(r"[一二两三四五六七八九十]+", match.group("num")) and not unit and amount < MIN_MONTHLY_PAY:
            continue
        if MIN_MONTHLY_PAY <= amount <= MAX_MONTHLY_PAY:
            return int(amount)
    return None


def _first(patterns: tuple[tuple[str, str], ...], text: str) -> list[str]:
    return [label for pattern, label in patterns if re.search(pattern, text)]


def _dates(text: str) -> list[Fact]:
    facts: list[Fact] = []
    for match in _DATE.finditer(text):
        year, month = int(match.group("y")), int(match.group("m"))
        if not (1990 <= year <= 2100 and 1 <= month <= 12):
            continue
        day = match.group("d") or match.group("d2")
        value = f"{year}-{month:02d}" + (f"-{int(day):02d}" if day and 1 <= int(day) <= 31 else "")
        after = text[match.end() : match.end() + 8]
        before = text[max(0, match.start() - 6) : match.start()]
        # The word right after a date ("2024年3月被辞退") wins; the text before it is only a fallback,
        # otherwise the previous date's "入职" would be attributed to this one.
        for window in (after, before):
            if any(word in window for word in _ENTRY_WORDS):
                facts.append(Fact("入职时间", value, 0.8))
                break
            if any(word in window for word in _EXIT_WORDS):
                facts.append(Fact("离职时间", value, 0.8))
                break
    return facts


def extract_facts(text: str) -> list[Fact]:
    """Facts stated in `text` (already redacted). Pure and offline; nothing here is inferred beyond wording."""
    facts: list[Fact] = []
    if (region := _REGION_RE.search(text)):
        facts.append(Fact("所在地", region.group(1), 0.6))
    if (industry := _first(_INDUSTRY, text)):
        facts.append(Fact("行业岗位", industry[0], 0.6))
    if (form := _first(_FORM, text)):
        facts.append(Fact("用工形式", form[0], 0.6))
    if (pay := _pay(text)) is not None:
        facts.append(Fact("月薪", f"{pay}元", 0.75))
    if (duration := _DURATION.search(text)) and (number := _number(duration.group("num"))) is not None:
        facts.append(Fact("工作时长", f"约{number:g}{duration.group('unit')}", 0.65))
    facts += _dates(text)
    if _CONTRACT_NO.search(text):
        facts.append(Fact("劳动合同", "无书面劳动合同", 0.8))
    elif _CONTRACT_YES.search(text):
        facts.append(Fact("劳动合同", "有书面劳动合同", 0.7))
    if _INSURANCE_NO.search(text):
        facts.append(Fact("社保", "单位未缴社保", 0.8))
    elif _INSURANCE_YES.search(text):
        facts.append(Fact("社保", "单位已缴社保", 0.7))
    for label in _first(_SPECIAL, text):
        facts.append(Fact("特殊情形", label, 0.7))
    for label in _first(_EVIDENCE, text):
        facts.append(Fact("已有证据", label, 0.65))
    for label in _first(_ACTION, text):
        facts.append(Fact("已采取行动", label, 0.65))
    for label in _first(_PREFERENCE, text):
        facts.append(Fact("回答偏好", label, 0.7))
    for concept in match_concepts(text):
        if concept.key not in PROCEDURAL:
            facts.append(Fact("争议主题", concept.label, 0.6))
    return list(dict.fromkeys(facts))


def _llm_facts(text: str) -> list[Fact]:
    """Optional second pass by a chat model; off without a key. Output is whitelisted and re-redacted."""
    if os.getenv("PROFILE_LLM", "1") == "0" or not os.getenv("OPENAI_API_KEY"):
        return []
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from . import llm as llm_module

        model = llm_module.chat_model(model=os.getenv("PROFILE_MODEL") or None, temperature=0, timeout=8, max_retries=0)
        if model is None:
            return []
        reply = model.invoke([
            SystemMessage(content=(
                "从用户的法律咨询中提取稳定、可复用的背景事实，只提取用户明确说过的内容，不要推测。只输出 JSON："
                '{"facts":[{"key":"…","value":"…"}]}。key 只能是：' + "、".join(sorted(ALLOWED_KEYS))
                + "。不要输出姓名、电话、证件号、住址、单位名称。"
            )),
            HumanMessage(content=text[:4000]),
        ])
        raw = str(reply.content).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        payload = json.loads(raw)
    except Exception:
        return []
    facts = []
    for item in payload.get("facts", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict) and item.get("key") in ALLOWED_KEYS and str(item.get("value", "")).strip():
            facts.append(Fact(item["key"], privacy.redact_text(str(item["value"]).strip())[:80], 0.5))
    return facts[:12]


def update_from_text(store: Any, user_id: str, text: str, source: str = "conversation", use_model: bool = True) -> int:
    """Merge facts found in one redacted user message into the long-term profile.

    The rules run instantly. The model pass is slow (seconds), so the chat endpoint calls this with
    use_model=False and schedules update_with_model after the reply has been sent.
    """
    facts = extract_facts(text) + (_llm_facts(text) if use_model else [])
    items = [{"key": f.key, "value": f.value, "confidence": f.confidence, "source": source} for f in facts]
    return store.upsert_profile_items(user_id, items) if items else 0


def update_with_model(store: Any, user_id: str, text: str, source: str = "conversation") -> int:
    """Background pass: facts only a model finds. Never raises, so a failing model cannot break a request."""
    try:
        items = [{"key": f.key, "value": f.value, "confidence": f.confidence, "source": source} for f in _llm_facts(text)]
        return store.upsert_profile_items(user_id, items) if items else 0
    except Exception:
        return 0


_EXPLICIT_KEYS = {"姓名": "姓名", "单位": "单位名称", "地址": "常住地址"}


def remember_entities(store: Any, user_id: str, entities: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Store names, employers and addresses the user explicitly asked to be remembered."""
    items = [{"key": _EXPLICIT_KEYS[kind], "value": value, "sensitive": True, "explicit": True,
              "confidence": 1.0, "source": "用户要求记住"} for kind, value in entities if kind in _EXPLICIT_KEYS]
    if items:
        store.upsert_profile_items(user_id, items)
    return [{"key": item["key"], "value": item["value"]} for item in items]


# ------------------------------------------------------------------------ conversation compression

def _cited(text: str) -> list[str]:
    from .verify import extract_citations

    return [f"《{c.law}》{c.article}" for c in extract_citations(text)]


def summarize(messages: list[dict[str, Any]]) -> str:
    """Compact summary of a conversation: known facts, topics, and what the user was already told."""
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    facts = extract_facts(user_text)
    grouped: dict[str, list[str]] = {}
    for fact in facts:
        grouped.setdefault(fact.key, [])
        if fact.value not in grouped[fact.key]:
            grouped[fact.key].append(fact.value)
    known = "；".join(f"{key}：{'、'.join(values)}" for key, values in grouped.items() if key != "回答偏好")
    cited = list(dict.fromkeys(_cited(assistant_text)))[:8]
    cases = len(re.findall(r"^- .+（.+）：", assistant_text, flags=re.M))
    parts = []
    if known:
        parts.append(f"用户已说明的情况：{known}")
    if cited:
        parts.append("此前已向用户引用：" + "、".join(cited))
    if cases:
        parts.append(f"此前已列出类似案例约{cases}个")
    return "。".join(parts)[:SUMMARY_CHARS]


def _llm_summary(messages: list[dict[str, Any]]) -> str:
    if os.getenv("PROFILE_LLM", "1") == "0" or not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from . import llm as llm_module

        model = llm_module.chat_model(model=os.getenv("PROFILE_MODEL") or None, temperature=0, timeout=10, max_retries=0)
        if model is None:
            return ""
        body = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in messages)[:6000]
        reply = model.invoke([
            SystemMessage(content="把法律咨询对话压缩成 300 字以内的摘要，保留：用户的关键事实、已给出的结论、已引用的法条名称和条号。"
                                  "不要添加对话中没有的内容，不要出现姓名、电话、证件号、住址。"),
            HumanMessage(content=body),
        ])
        return privacy.redact_text(str(reply.content).strip())[:SUMMARY_CHARS]
    except Exception:
        return ""


def compress_if_needed(store: Any, user_id: str, conversation_id: str, keep_recent: int = KEEP_RECENT) -> bool:
    """Fold older messages into a summary once the recent window has grown by COMPRESS_SLACK messages."""
    conversation = store.get_conversation(conversation_id, user_id)
    if conversation is None:
        return False
    messages = conversation["messages"]
    covered_now = (store.get_summary(conversation_id) or {}).get("covered", 0)
    if len(messages) - covered_now <= keep_recent + COMPRESS_SLACK:
        return False
    covered = len(messages) - keep_recent
    older = messages[:covered]
    summary = _llm_summary(older) or summarize(older)
    if not summary:
        return False
    store.save_summary(conversation_id, summary, covered)
    return True


def render(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        grouped.setdefault(item["key"], []).append(item["value"])
    return grouped

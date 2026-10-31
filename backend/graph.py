from __future__ import annotations

import json
import os
import operator
import re
from datetime import date
from typing import Any, Literal, TypedDict
from typing_extensions import Annotated

from . import intake
from . import rag as rag_module
from .cases import search_cases
from .casebase.schema import EXTERNAL_LABEL, REVIEWED_LABEL, TIER_ORDER
from . import config as app_config
from . import llm as llm_module
from . import directory, scriptbase
from .config import MAX_HISTORY_MESSAGES
from .memory import MemoryStore
from .rag import search_statutes
from .retrieval.query import match_concepts
from .verify import verify_answer

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency is installed by pyproject
    ChatOpenAI = None
    HumanMessage = SystemMessage = None

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Overwrite, Send
except ImportError:  # pragma: no cover - local lightweight fallback for source inspection
    MemorySaver = None
    Overwrite = None
    Send = None
    END = START = StateGraph = None


Intent = Literal["statute", "case", "aid", "mixed", "general"]


class WorkAidState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    message: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    intent: Intent
    worker_outputs: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    verification: dict[str, Any]
    case: dict[str, Any]
    case_view: dict[str, Any] | None
    case_terms: str
    intake_section: str
    stream: bool

def classify_intent(message: str) -> Intent:
    text = message.lower()
    statute = ("法条", "法律", "条例", "条文", "依据", "规定", "第几条", "民法典", "劳动法")
    case = ("案例", "判例", "裁判", "判决", "类似案件", "类案", "法院", "案号")
    aid = ("怎么办", "如何处理", "起诉", "仲裁", "报警", "维权", "流程", "证据", "援助", "下一步")
    flags = [any(word in text for word in words) for words in (statute, case, aid)]
    if sum(flags) > 1:
        return "mixed"
    if flags[0]:
        return "statute"
    if flags[1]:
        return "case"
    if flags[2]:
        return "aid"
    return "general"


def _llm() -> Any | None:
    # Generation time is roughly proportional to length; an uncapped answer took 25-45 s.
    return llm_module.chat_model(
        max_tokens=int(os.getenv("ANSWER_MAX_TOKENS", "1400")), timeout=float(os.getenv("ANSWER_TIMEOUT", "75"))
    )


def _route(state: WorkAidState) -> dict[str, Any]:
    intent = classify_intent(state["message"])
    # The keyword rules are decisive when they find something; the model is only asked about messages
    # they cannot place, which saves a 1-4 s round trip on most turns.
    llm = _llm() if intent == "general" else None
    if llm and HumanMessage and SystemMessage:
        try:
            result = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "判断中文法律咨询应路由到哪个专长。只输出一个 JSON 对象，"
                            '字段 intent 必须是 statute、case、aid、mixed、general 之一。'
                            "statute=检索法规条文，case=找类似裁判案例，aid=维权/求助路径，"
                            "mixed=明显需要多类专家，general=其他法律解释。"
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "history": state.get("history", [])[-8:],
                                "message": state["message"],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            payload = json.loads(str(result.content))
            candidate = payload.get("intent")
            if candidate in {"statute", "case", "aid", "mixed", "general"} and (
                intent == "general" or candidate != "general"
            ):
                intent = candidate
        except Exception:
            pass
    return {
        "intent": intent,
        "worker_outputs": Overwrite([]) if Overwrite else [],
        "evidence": Overwrite([]) if Overwrite else [],
    }


def _search_text(state: WorkAidState) -> str:
    """The question as a search query. Short follow-ups ("那赔多少") carry no searchable facts, so the
    prior user turn is prepended; when the message names no legal topic itself, the case file's
    statute-style phrases are appended so the search keeps the subject of the dispute."""
    message = state["message"]
    if len(message.strip()) < 12:
        previous = [item["content"] for item in state.get("history", []) if item.get("role") == "user"]
        if previous:
            message = f"{previous[-1]} {message}"
    terms = state.get("case_terms", "")
    if terms and not match_concepts(state["message"]):
        message = f"{message} {terms}"
    return message


def _has_legal_signal(message: str, has_case: bool = False) -> bool:
    """Greetings and small talk must not trigger statute search, which may reach a paid remote.
    A follow-up inside an identified dispute ("那赔多少") does count as a legal question."""
    text = message.strip()
    if has_case and len(text) >= 2 and classify_intent(text) == "general" and not _SMALL_TALK.fullmatch(text):
        return True
    return classify_intent(text) != "general" or bool(match_concepts(text)) or len(text) >= 12


_SMALL_TALK = re.compile(r"你好|您好|hi|hello|嗨|谢谢|多谢|好的|嗯+|哦+|再见|拜拜|ok|OK|在吗|在不在")


def _targets(intent: str, message: str, has_case: bool = False) -> list[str]:
    if intent == "general":
        return ["statute", "aid"] if _has_legal_signal(message, has_case) else ["aid"]
    return {
        "statute": ["statute"],
        "case": ["case"],
        "aid": ["statute", "aid"],
        "mixed": ["statute", "case", "aid"],
    }.get(intent, ["aid"])


def _statute_worker(state: WorkAidState) -> dict[str, Any]:
    # Local corpus first; the remote (Huayu Yuandian) is only consulted inside search_statutes on a miss.
    items = search_statutes(_search_text(state))
    remote = next((item.get("source_name") for item in items if item.get("source_name")), "")
    output = {
        "worker": "statute",
        "label": "法条检索 Worker",
        "summary": (
            f"本地知识库未命中，由{remote}提供兜底结果。"
            if items and remote
            else "检索到本地法规条文。"
            if items
            else "本地法规库与华宇元典均未命中，需要补充法律名称、案情和地区，或咨询专业律师。"
        ),
        "items": items,
    }
    return {"worker_outputs": [output], "evidence": items}


def _case_worker(state: WorkAidState) -> dict[str, Any]:
    items = search_cases(_search_text(state))
    remote = next((item.get("source_name") for item in items if item.get("source_name")), "")
    output = {
        "worker": "case",
        "label": "类似案例 Worker",
        "summary": (
            f"本地案例目录未命中，由{remote}提供兜底结果。"
            if items and remote
            else "找到本地可比案例。"
            if items
            else "本地案例目录与华宇元典均未命中。"
        ),
        "items": items,
    }
    return {"worker_outputs": [output], "evidence": items}


def _aid_worker(state: WorkAidState) -> dict[str, Any]:
    case = state.get("case") or {}
    scenario = case.get("case_type") or ""
    if scenario and scenario != "general":
        known = intake.known(case)
        today = date.today()
        deadline = ""
        if known.get("exit_date") and known["exit_date"] != intake.ONGOING:
            result = intake.calc.arbitration_limit(known["exit_date"], today)
            deadline = f" {result.summary}" if result else ""
        elif known.get("injury_date"):
            result = intake.calc.injury_deadlines(known["injury_date"], today)
            deadline = f" {result.summary}" if result else ""
        only_reviewed = os.getenv("DIRECTORY_ONLY_REVIEWED", "0") == "1"
        steps = directory.build_path(app_config.INSTITUTIONS_PATH, scenario, str(known.get("region", "")), deadline, only_reviewed)
        label = intake.CASE_TYPES[scenario].label
        output = {"worker": "aid", "label": "法律援助路径 Worker",
                  "summary": f"已按“{label}”整理行动路径" + (f"（{known['region']}）" if known.get("region") else "；告诉我所在城市可以列出当地机构") + "。",
                  "items": steps}
        return {"worker_outputs": [output]}
    output = {
        "worker": "aid",
        "label": "法律援助路径 Worker",
        "summary": "已根据当前描述整理行动路径。",
        "items": [
            {"step": "1", "title": "固定事实与证据", "detail": "按时间线保存合同、聊天、付款、通知和现场材料原件。"},
            {"step": "2", "title": "确认管辖与时限", "detail": "核对地域管辖、仲裁条款及诉讼时效，避免错过期限。"},
            {"step": "3", "title": "选择救济渠道", "detail": "根据争议类型评估协商、投诉、调解、仲裁或诉讼，并保留提交凭证。"},
            {"step": "4", "title": "寻求专业帮助", "detail": "重大财产、刑事风险或人身安全问题应尽快联系当地律师或法律援助中心。"},
        ],
    }
    return {"worker_outputs": [output]}


NO_RESULT_MARK = "### 未检索到相关内容"
_SEARCHED = (("statute", "条文"), ("case", "案例"))


def no_result_notice(outputs: dict[str, dict[str, Any]]) -> str:
    """Last-resort text when every retrieval tier (local, remote) came back empty for a searched kind."""
    missing = [f"相关{name}" for key, name in _SEARCHED if key in outputs and not outputs[key].get("items")]
    if not missing:
        return ""
    what = "、".join(missing)
    return (
        f"{NO_RESULT_MARK}\n本地资料库和远程法律检索都没有找到{what}，上面的内容不是对{what}的引用，"
        "请不要据此判断您的权利或结果。建议您尽快咨询专业律师；经济困难的，可拨打 12348 法律服务热线，"
        "或到当地法律援助中心申请法律援助。"
    )


def _force_unreviewed_warning(items: list[dict[str, Any]]) -> bool:
    return any(item.get("review_status") != REVIEWED_LABEL for item in items)


def _statute_line(record: dict[str, Any]) -> str:
    """One statute as shown to the user, with how current it is."""
    status = [
        record.get("effectiveness") or "时效待核实",
        f"{record['publish_date']} 公布" if record.get("publish_date") else "",
        f"{record['implement_date']} 施行" if record.get("implement_date") else "",
    ]
    quality = "；注意：本条文本在语料中被截断，请以官方原文为准" if record.get("quality") == "truncated" else ""
    return (
        f"- 《{record.get('law_name', '未注明法律')}》{record.get('article_number', '')}"
        f"（{'，'.join(part for part in status if part)}）：{record.get('content', '')[:240]}{quality} "
        f"{record.get('url', '')}"
    ).rstrip()


def _case_lines(records: list[dict[str, Any]]) -> list[str]:
    """Cases grouped by how far their source can be trusted; unverified ones carry a visible warning."""
    lines: list[str] = []
    for tier in TIER_ORDER:
        group = [r for r in records if (r.get("review_status") or EXTERNAL_LABEL) == tier]
        if not group:
            continue
        lines.append(f"**{tier}**" + ("" if tier == REVIEWED_LABEL else "（出处未经本库对照官方原文核对，仅供参考，请以官方原文为准）"))
        for record in group[:3]:
            meta = "，".join(
                part for part in (
                    record.get("court") or record.get("issuer"),
                    record.get("case_number"),
                    record.get("decision_date") or record.get("publish_date"),
                ) if part
            )
            lines.append(
                f"- {record.get('title', '未命名案例')}{f'（{meta}）' if meta else ''}："
                f"{(record.get('holding') or record.get('summary', ''))[:240]} {record.get('url', '')}".rstrip()
            )
    return lines


def _verified(answer: str, state: WorkAidState) -> tuple[str, dict[str, Any]]:
    """Check every statute article and case number in the answer; never let verification break a reply."""
    try:
        engine = rag_module._load_engine()
        report = verify_answer(answer, state.get("evidence", []), engine)
    except Exception:
        return answer, {"checked": 0, "problems": 0, "error": True}
    text = report.answer + (f"\n\n{report.section}" if report.section else "")
    return text, {"checked": report.checked, "problems": report.problems}


def _intake(state: WorkAidState) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Case-file section for this turn: known facts, computed figures, and at most three questions.

    Explicit statute or case lookups are answered as asked, without a questionnaire.
    """
    case = state.get("case")
    if not case or state.get("intent") in ("statute", "case") or not intake.type_of(case):
        return "", case, None
    questions, updated = intake.next_questions(case)
    results = intake.calculations(updated)
    return (
        intake.render_section(updated, questions, results),
        updated,
        intake.public_view(updated, questions, results),
    )


def _generate(llm: Any, messages: list[Any], stream: bool) -> str:
    """Run the model. When the graph is being streamed, tokens are pushed to the client as they arrive
    through LangGraph's custom stream; otherwise (and for models without .stream) it is one call."""
    if stream and hasattr(llm, "stream"):
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            writer = None
        if writer is not None:
            parts: list[str] = []
            for chunk in llm.stream(messages):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if piece:
                    parts.append(piece)
                    writer({"delta": piece})
            return "".join(parts)
    return str(llm.invoke(messages).content)


def _synthesize(state: WorkAidState) -> dict[str, Any]:
    section, case, view = _intake(state)
    result = _synthesize_text({**state, "intake_section": section})
    if case is not None:
        result["case"] = case
    result["case_view"] = view
    return result


def _synthesize_text(state: WorkAidState) -> dict[str, Any]:
    outputs = {item["worker"]: item for item in state.get("worker_outputs", [])}
    notice = no_result_notice(outputs)
    llm = _llm()
    if llm and HumanMessage and SystemMessage:
        evidence = json.dumps(state.get("evidence", []), ensure_ascii=False)[:18000]
        cases_unreviewed = _force_unreviewed_warning(outputs.get("case", {}).get("items", []))
        prompt = (
            "你是 WorkAid 的 Leader。结合最近的对话历史、用户问题、画像和 Worker 的结构化结果，"
            "用中文给出谨慎、可核验的法律信息。不要编造条文、案例或链接；明确说明未知和需核实之处。"
            "回答包含：结论摘要、依据/案例、行动路径、风险提示。"
            "篇幅要求：全文不超过 700 字；法条只引用最关键的 5 条以内，不要罗列所有相关条文；同一法条不要重复引用。"
            "引用法条时写出法规全名和条号，并注明该法规的时效性及公布、施行日期（见证据字段）；"
            "引用条文原文时用引号并与证据逐字一致，不确定就概括而不要引用原文；案号只能使用证据里出现的。"
            "金额、期限、月数一律不要自己计算：系统会在回答后附上由程序计算的“案情梳理”，你只需说明依据并提示用户核对，"
            "也不要重复向用户追问系统已列出的问题。"
            + (
                "案例按 review_status 分开列出，“已核验”在前；非“已核验”的案例必须注明“出处未经核验，仅供参考”。"
                if cases_unreviewed
                else ""
            )
            + (
                "某类检索结果为空时，必须明确说明未检索到，不得凭记忆列举条文或案例，并建议咨询专业律师。"
                if notice
                else ""
            )
            + "用户问题："
            f"对话历史：{json.dumps(state.get('history', [])[-12:], ensure_ascii=False)}"
            f"\n当前问题：{state['message']}\n用户画像：{json.dumps(state.get('profile', {}), ensure_ascii=False)}"
            f"\nWorker结果：{json.dumps(outputs, ensure_ascii=False)}\n证据：{evidence}"
        )
        try:
            answer = _generate(
                llm,
                [SystemMessage(content="你是严谨的法律信息助手。"), HumanMessage(content=prompt)],
                bool(state.get("stream")),
            )
            if notice and NO_RESULT_MARK not in answer:
                answer = f"{answer}\n\n{notice}"
            if state.get("intake_section"):
                answer = f"{answer}\n\n{state['intake_section']}"
            text, verification = _verified(answer, state)
            return {"answer": text, "verification": verification}
        except Exception:
            pass
    sections = ["## Leader 综合意见", "以下内容基于当前问题和本地可用资料，不能替代律师的个案意见。"]
    for key in ("statute", "case", "aid"):
        item = outputs.get(key)
        if not item:
            continue
        sections.append(f"\n### {item['label']}\n{item['summary']}")
        if key == "statute":
            sections.extend(_statute_line(record) for record in item.get("items", [])[:3])
        elif key == "case":
            sections.extend(_case_lines(item.get("items", [])))
        else:
            sections.extend(f"- {record.get('title')}: {record.get('detail')}" for record in item.get("items", [])[:3])
    if notice:
        sections.append(f"\n{notice}")
    if state.get("intake_section"):
        sections.append(f"\n{state['intake_section']}")
    sections.append("\n### 核验提醒\n法规效力、管辖、时效和证据完整性需要结合具体事实再次核实。")
    text, verification = _verified("\n".join(sections), state)
    return {"answer": text, "verification": verification}


def _dispatch(state: WorkAidState) -> list[Any] | str:
    if Send is None:
        return "synthesize"
    targets = _targets(state.get("intent", "general"), state["message"], bool(state.get("case_terms")))
    return [Send(target, state) for target in targets]


def _build_graph() -> Any:
    if StateGraph is None:
        return None
    graph = StateGraph(WorkAidState)
    graph.add_node("route", _route)
    graph.add_node("statute", _statute_worker)
    graph.add_node("case", _case_worker)
    graph.add_node("aid", _aid_worker)
    graph.add_node("synthesize", _synthesize)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _dispatch)
    for node in ("statute", "case", "aid"):
        graph.add_edge(node, "synthesize")
    graph.add_edge("synthesize", END)
    checkpointer = MemorySaver() if MemorySaver else None
    return graph.compile(checkpointer=checkpointer)


class LeaderGraph:
    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.graph = _build_graph()

    def _scripts(self, user_id: str, state: WorkAidState, with_identity: bool = True) -> list[dict[str, Any]]:
        """Assertion scripts for this dispute, filled with the user's facts. Done after the graph runs so
        that a name or employer the user asked to remember never enters graph state or a model prompt.
        `with_identity=False` leaves those blanks, which is the version that is allowed to be stored."""
        case = state.get("case") or {}
        if state.get("case_view") is None or not case.get("case_type"):
            return []
        wanted = [case["case_type"], *case.get("topics", [])]
        picked = scriptbase.select(app_config.SCRIPTS_PATH, wanted, 2, os.getenv("SCRIPTS_ONLY_REVIEWED", "0") == "1")
        if not picked:
            return []
        explicit = {i["key"]: i["value"] for i in self.memory.list_profile_items(user_id) if i["sensitive"]} if with_identity else {}
        values = scriptbase.fill_values(intake.known(case), explicit, date.today().strftime("%Y年%m月%d日").replace("年0", "年").replace("月0", "月"))
        return [scriptbase.render(record, values) for record in picked]

    def _prepare(self, user_id: str, conversation_id: str, message: str) -> WorkAidState:
        """Everything the graph needs before it starts: recent history (or its summary), the case file
        brought up to date with this turn, and the profile as a model may see it."""
        conversation = self.memory.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise KeyError("会话不存在")
        messages = conversation["messages"]
        summary = self.memory.get_summary(conversation_id)
        if summary and summary["covered"] > 0:
            # Older turns are represented by their compressed summary; only the recent ones are verbatim.
            recent = messages[summary["covered"] : -1]
            history = [{"role": "system", "content": f"此前对话摘要：{summary['summary']}"}]
        else:
            recent = messages[-(MAX_HISTORY_MESSAGES + 1) : -1]
            history = []
        history += [{"role": item["role"], "content": item["content"]} for item in recent]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        case = intake.update_case(
            self.memory.get_case_file(conversation_id) or intake.new_case(),
            user_messages,
            self.memory.list_profile_items(user_id),
        )
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": message,
            "history": history,
            "profile": self.memory.profile_for_prompt(user_id),
            "case": case,
            "case_terms": intake.search_terms(case),
        }

    def run(self, *, user_id: str, conversation_id: str, message: str) -> dict[str, Any]:
        state = self._prepare(user_id, conversation_id, message)
        if self.graph is None:
            state.update(_route(state))
            targets = _targets(state["intent"], message, bool(state["case_terms"]))
            if "statute" in targets:
                delta = _statute_worker(state)
                state["worker_outputs"].extend(delta["worker_outputs"])
                state["evidence"].extend(delta["evidence"])
            if "case" in targets:
                delta = _case_worker(state)
                state["worker_outputs"].extend(delta["worker_outputs"])
                state["evidence"].extend(delta["evidence"])
            if "aid" in targets:
                state["worker_outputs"].extend(_aid_worker(state)["worker_outputs"])
            state.update(_synthesize(state))
        else:
            result = self.graph.invoke(
                state,
                config={"configurable": {"thread_id": conversation_id}},
            )
            state.update(result)
        return self._finish(user_id, conversation_id, state)

    def _finish(self, user_id: str, conversation_id: str, state: WorkAidState) -> dict[str, Any]:
        """Persist the case file, add the filled scripts, and shape the reply."""
        if state.get("case") is not None:
            self.memory.save_case_file(conversation_id, state["case"])
        answer = state.get("answer", "")
        stored_answer = answer
        rendered = self._scripts(user_id, state)
        if rendered:
            answer = answer + "\n\n" + scriptbase.render_section(rendered)
            # What is persisted keeps the identity blanks: a remembered name must not land in chat history.
            stored_answer = stored_answer + "\n\n" + scriptbase.render_section(self._scripts(user_id, state, with_identity=False))
        return {
            "intent": state.get("intent", "general"),
            "scripts": rendered,
            "answer": answer,
            "answer_stored": stored_answer,
            "workers": list(state.get("worker_outputs", [])),
            "evidence": state.get("evidence", []),
            "verification": state.get("verification", {"checked": 0, "problems": 0}),
            "case_file": state.get("case_view"),
        }

    def stream(self, *, user_id: str, conversation_id: str, message: str):
        """Run the graph and yield progress as it happens, as {"event": ..., "data": ...} dicts.

        Events: stage (what the agent is doing), route (the intent), worker (one specialist finished,
        with its evidence), delta (a piece of the answer text), and finally one "final" with the same
        payload `run` returns. The final answer replaces the streamed text because citation
        verification may delete or annotate parts of it.
        """
        state = self._prepare(user_id, conversation_id, message)
        state["stream"] = True
        yield {"event": "stage", "data": {"name": "understand"}}
        if self.graph is None:
            result = self.run(user_id=user_id, conversation_id=conversation_id, message=message)
            yield {"event": "final", "data": result}
            return

        outputs: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        expected: set[str] = set()
        done: set[str] = set()
        merged: dict[str, Any] = {}
        for mode, payload in self.graph.stream(
            state,
            config={"configurable": {"thread_id": conversation_id}},
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                if "delta" in payload:
                    yield {"event": "delta", "data": {"text": payload["delta"]}}
                continue
            for node, update in payload.items():
                if update is None:
                    continue
                if node == "route":
                    merged["intent"] = update["intent"]
                    expected = set(_targets(update["intent"], message, bool(state.get("case_terms"))))
                    yield {"event": "route", "data": {"intent": update["intent"], "workers": sorted(expected)}}
                    yield {"event": "stage", "data": {"name": "retrieve"}}
                elif node in ("statute", "case", "aid"):
                    new_outputs = update.get("worker_outputs", [])
                    new_evidence = update.get("evidence", [])
                    outputs.extend(new_outputs)
                    evidence.extend(new_evidence)
                    done.add(node)
                    yield {"event": "worker", "data": {"outputs": new_outputs, "evidence": new_evidence}}
                    if expected and expected <= done:
                        yield {"event": "stage", "data": {"name": "compose"}}
                elif node == "synthesize":
                    merged.update(update)
        yield {"event": "stage", "data": {"name": "verify"}}
        final_state: WorkAidState = {**state, **merged, "worker_outputs": outputs, "evidence": evidence}
        yield {"event": "final", "data": self._finish(user_id, conversation_id, final_state)}

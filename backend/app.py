from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from . import casebase
from . import directory, intake
from . import privacy
from . import profile as profile_module
from .cases import case_sources
from .config import CASE_CATALOG_PATH, DATABASE_PATH, INSTITUTIONS_PATH, ROOT
from .graph import LeaderGraph
from .memory import MemoryStore
from .rag import engine_status, search_statutes_traced, statute_sources
from .retrieval import vector
from .schemas import CaseAnswersRequest, CaseSearchRequest, ChatRequest, ProfileRequest
from .warmup import Warmup
from . import yuandian_mcp

logger = logging.getLogger("workaid")


def create_app(
    *,
    database_path: Path | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    store = MemoryStore(database_path or DATABASE_PATH)
    leader = LeaderGraph(store)
    warmup = Warmup()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        warmup.start()
        yield

    app = FastAPI(
        title="WorkAid",
        description="Intent-routed legal assistance with shared LangGraph context",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.memory = store
    app.state.leader = leader
    app.state.warmup = warmup

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        status = engine_status()
        return {
            "ok": True,
            "app": "WorkAid",
            "langgraph": leader.graph is not None,
            "statute_rag": status,
            "vector_index": vector.status(),
            "case_catalog": casebase.stats(CASE_CATALOG_PATH),
            "warmup": warmup.snapshot(),
            "yuandian_mcp": yuandian_mcp.status(),
        }

    @app.post("/api/cases/search")
    def case_search(
        request: CaseSearchRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """Case search over this instance's catalog only, for use as another instance's CASE_SEARCH_URL.

        Disabled (404) until CASE_SERVICE_API_KEY is set; callers send `Authorization: Bearer <key>`.
        """
        expected = os.getenv("CASE_SERVICE_API_KEY", "").strip()
        if not expected:
            raise HTTPException(status_code=404, detail="Not found")
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {"items": casebase.search(request.query, request.limit, path=CASE_CATALOG_PATH)}

    def _begin_turn(request: ChatRequest) -> dict[str, Any]:
        """Start a turn: the only place raw user text exists. Everything after this point (storage,
        retrieval, remote search, language models, profile extraction) sees the redacted text."""
        analysis = privacy.analyze(request.message.strip())
        message = analysis.text

        conversation_id = request.conversation_id
        if conversation_id is None:
            created = store.create_conversation(request.user_id, message.replace("\n", " ")[:60] or "新对话")
            conversation_id = created["id"]
        elif store.get_conversation(conversation_id, request.user_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        remembered: list[dict[str, str]] = []
        if analysis.remember and analysis.entities:
            remembered = profile_module.remember_entities(store, request.user_id, analysis.entities)
        store.add_message(conversation_id, request.user_id, "user", message)
        profile_module.update_from_text(store, request.user_id, message, conversation_id, use_model=False)
        return {
            "conversation_id": conversation_id,
            "message": message,
            "privacy": {"redacted": analysis.counts, "remembered": [item["key"] for item in remembered]},
        }

    def _end_turn(request: ChatRequest, turn: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Persist the reply (with identity blanks left empty) and shape the response."""
        store.add_message(
            turn["conversation_id"],
            request.user_id,
            "assistant",
            result.get("answer_stored", result["answer"]),
            {
                "intent": result["intent"],
                "workers": result["workers"],
                "evidence": result["evidence"],
                "verification": result.get("verification"),
                "case_file": result.get("case_file"),
            },
        )
        reply = {k: v for k, v in result.items() if k != "answer_stored"}
        return {"conversation_id": turn["conversation_id"], **reply, "privacy": turn["privacy"]}

    def _after_reply(request: ChatRequest, turn: dict[str, Any]) -> None:
        """Model passes (extra profile facts, folding old turns into a summary) take seconds and only help
        later turns, so they run after the reply has been sent."""
        profile_module.update_with_model(store, request.user_id, turn["message"], turn["conversation_id"])
        profile_module.compress_if_needed(store, request.user_id, turn["conversation_id"])

    @app.post("/api/chat")
    def chat(request: ChatRequest, background: BackgroundTasks) -> dict[str, Any]:
        turn = _begin_turn(request)
        try:
            result = leader.run(user_id=request.user_id, conversation_id=turn["conversation_id"], message=turn["message"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        response = _end_turn(request, turn, result)
        background.add_task(_after_reply, request, turn)
        return response

    def _sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Server-sent events: progress while the agent works, the answer as it is written, then the
        complete verified reply. An unknown conversation is a normal 404 before the stream starts."""
        turn = _begin_turn(request)

        def events():
            yield _sse("meta", {"conversation_id": turn["conversation_id"], "privacy": turn["privacy"]})
            final = None
            try:
                for item in leader.stream(
                    user_id=request.user_id, conversation_id=turn["conversation_id"], message=turn["message"]
                ):
                    if item["event"] == "final":
                        final = item["data"]
                    else:
                        yield _sse(item["event"], item["data"])
                if final is not None:
                    yield _sse("final", _end_turn(request, turn, final))
            except Exception:
                logger.exception("streaming chat failed")
                yield _sse("error", {"message": "生成回答时出错，请稍后重试。"})
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(_after_reply, request, turn),
        )

    @app.get("/api/conversations")
    def conversations(user_id: str = Query(min_length=1, max_length=120)) -> list[dict[str, Any]]:
        return store.list_conversations(user_id)

    @app.get("/api/conversations/{conversation_id}")
    def conversation(
        conversation_id: str, user_id: str = Query(min_length=1, max_length=120)
    ) -> dict[str, Any]:
        result = store.get_conversation(conversation_id, user_id)
        if result is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return result

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str, user_id: str = Query(min_length=1, max_length=120)
    ) -> dict[str, bool]:
        if not store.delete_conversation(conversation_id, user_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": True}

    def _case_for(conversation_id: str, user_id: str) -> dict[str, Any]:
        if store.get_conversation(conversation_id, user_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return store.get_case_file(conversation_id) or intake.new_case()

    def _case_payload(case: dict[str, Any]) -> dict[str, Any]:
        questions = intake.next_questions({**case, "asked": {}})[0]
        return intake.public_view(case, questions, intake.calculations(case))

    @app.get("/api/conversations/{conversation_id}/case")
    def get_case(conversation_id: str, user_id: str = Query(min_length=1, max_length=120)) -> dict[str, Any]:
        return _case_payload(_case_for(conversation_id, user_id))

    @app.put("/api/conversations/{conversation_id}/case")
    def answer_case(conversation_id: str, request: CaseAnswersRequest) -> dict[str, Any]:
        """Questionnaire answers: each is validated; valid ones are saved, the rest come back as errors."""
        case = _case_for(conversation_id, request.user_id)
        case, errors = intake.apply_answers(case, request.answers)
        if len(errors) < len(request.answers):
            store.save_case_file(conversation_id, case)
        return {"case": _case_payload(case), "errors": errors}

    @app.get("/api/conversations/{conversation_id}/case/export")
    def export_case(conversation_id: str, user_id: str = Query(min_length=1, max_length=120)) -> PlainTextResponse:
        case = _case_for(conversation_id, user_id)
        return PlainTextResponse(
            intake.render_markdown(case, intake.calculations(case)),
            headers={"Content-Disposition": "attachment; filename=case-summary.md"},
        )

    @app.get("/api/conversations/{conversation_id}/scripts")
    def case_scripts(conversation_id: str, user_id: str = Query(min_length=1, max_length=120)) -> dict[str, Any]:
        case = _case_for(conversation_id, user_id)
        state = {"case": case, "case_view": {}}
        return {"scripts": leader._scripts(user_id, state)}

    @app.get("/api/institutions")
    def institutions(
        scenario: str = Query(default="general"), region: str = Query(default="", max_length=40)
    ) -> dict[str, Any]:
        entries = directory.lookup(INSTITUTIONS_PATH, scenario, region)
        return {"institutions": [{**e, "description": directory.describe(e)} for e in entries]}

    @app.get("/api/case-fields")
    def case_fields(case_type: str = Query(default="")) -> dict[str, Any]:
        selected = intake.CASE_TYPES.get(case_type)
        keys = selected.required if selected else tuple(intake.FIELDS)
        return {
            "case_types": {key: item.label for key, item in intake.CASE_TYPES.items()},
            "fields": [
                {"key": k, "label": intake.FIELDS[k].label, "question": intake.FIELDS[k].question,
                 "kind": intake.FIELDS[k].kind, "options": list(intake.FIELDS[k].options), "why": intake.FIELDS[k].why}
                for k in keys
            ],
        }

    @app.get("/api/profile")
    def profile(user_id: str = Query(min_length=1, max_length=120)) -> dict[str, Any]:
        return {**store.profile(user_id), "items": store.list_profile_items(user_id)}

    @app.put("/api/profile")
    def save_profile(request: ProfileRequest) -> dict[str, Any]:
        return store.save_profile(request.user_id, request.facts, request.preferences)

    @app.delete("/api/profile/items")
    def delete_profile_item(
        user_id: str = Query(min_length=1, max_length=120),
        key: str = Query(min_length=1, max_length=40),
        value: str = Query(min_length=1, max_length=200),
    ) -> dict[str, bool]:
        if not store.delete_profile_item(user_id, key, value):
            raise HTTPException(status_code=404, detail="没有这条记忆")
        return {"deleted": True}

    @app.delete("/api/user-data")
    def delete_user_data(user_id: str = Query(min_length=1, max_length=120)) -> dict[str, Any]:
        """Erase every conversation, summary and profile item stored for this user."""
        return {"deleted": store.delete_user_data(user_id)}

    @app.get("/api/sources")
    def sources() -> dict[str, Any]:
        return {
            "statutes": statute_sources(),
            "cases": case_sources(),
            "remote": yuandian_mcp.sources(),
            "mcp_status": yuandian_mcp.status(),
            "statute_status": engine_status(),
        }

    @app.get("/api/statutes/search")
    def statute_search(q: str = Query(min_length=1, max_length=1000)) -> list[dict[str, Any]]:
        from .rag import search_statutes

        return search_statutes(q)

    @app.get("/api/statutes/trace")
    def statute_trace(q: str = Query(min_length=1, max_length=1000)) -> dict[str, Any]:
        items, trace = search_statutes_traced(q, remote=False)
        return {"items": items, "trace": trace}

    static_dir = frontend_dir or (ROOT / "frontend" / "dist")
    if (static_dir / "index.html").exists():
        static_root = static_dir.resolve()
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = (static_root / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(static_root):
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app

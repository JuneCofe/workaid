from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx


_OFFICIAL_HOST = "open.chineselaw.com"
_LAW_URL = "https://open.chineselaw.com/mcp/law/stream"
_CASE_URL = "https://open.chineselaw.com/mcp/case/stream"
_TOOLS = {
    "statute": "yuandian_law_vector_search",
    "case": "yuandian_case_vector_search",
}
_PROTOCOL_VERSION = "2025-06-18"
_ACCEPT = "application/json, text/event-stream"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _endpoint(kind: str) -> str:
    if kind == "statute":
        return _setting("YUANDIAN_LAW_MCP_URL", _LAW_URL)
    if kind == "case":
        return _setting("YUANDIAN_CASE_MCP_URL", _CASE_URL)
    raise ValueError(f"Unknown Yuandian MCP service: {kind}")


def _tool(kind: str) -> str:
    env_name = "YUANDIAN_LAW_MCP_TOOL" if kind == "statute" else "YUANDIAN_CASE_MCP_TOOL"
    return _setting(env_name, _TOOLS[kind])


def _valid_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return (
        parsed.scheme == "https"
        and parsed.hostname == _OFFICIAL_HOST
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def status() -> dict[str, Any]:
    configured = bool(_setting("YUANDIAN_MCP_API_KEY"))
    services = {}
    for kind in ("statute", "case"):
        endpoint = _endpoint(kind)
        valid = _valid_endpoint(endpoint)
        services[kind] = {
            "configured": configured and valid,
            "endpoint": endpoint if valid else "",
            "tool": _tool(kind),
            "transport": "Streamable HTTP",
        }
    return {
        "provider": "华宇元典 MCP",
        "configured": configured and all(item["configured"] for item in services.values()),
        "services": services,
    }


def _decode_response(response: httpx.Response) -> dict[str, Any] | None:
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise ValueError("Yuandian MCP response exceeded the size limit")
    text = response.text.strip()
    if not text:
        return None
    if "text/event-stream" not in response.headers.get("content-type", ""):
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid Yuandian MCP JSON-RPC response")
        return payload

    messages = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        if isinstance(payload, dict):
            messages.append(payload)
    if not messages:
        raise ValueError("Yuandian MCP returned an empty event stream")
    return messages[-1]


def _post(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    session_id: str = "",
    notification: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": _ACCEPT,
        "Content-Type": "application/json",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    response = httpx.post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=35,
        follow_redirects=False,
    )
    response.raise_for_status()
    new_session = response.headers.get("Mcp-Session-Id", session_id)
    if notification and response.status_code == 202 and not response.content:
        return None, new_session
    return _decode_response(response), new_session


def _rpc_result(
    message: dict[str, Any] | None, request_id: int
) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("id") != request_id:
        raise ValueError("Unexpected Yuandian MCP JSON-RPC response")
    if "error" in message:
        error = message["error"]
        detail = error.get("message", "MCP request failed") if isinstance(error, dict) else str(error)
        raise ValueError(f"Yuandian MCP error: {detail}")
    result = message.get("result")
    if not isinstance(result, dict):
        raise ValueError("Yuandian MCP response has no result")
    return result


def _call_tool(kind: str, arguments: dict[str, Any]) -> Any:
    endpoint = _endpoint(kind)
    api_key = _setting("YUANDIAN_MCP_API_KEY")
    if not api_key:
        return None
    if not _valid_endpoint(endpoint):
        raise ValueError("Yuandian MCP endpoint must use the official HTTPS host")

    session_id = ""
    initialized, session_id = _post(
        endpoint,
        api_key,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "WorkAid", "version": "0.1.0"},
            },
        },
    )
    _rpc_result(initialized, 1)
    _, session_id = _post(
        endpoint,
        api_key,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=session_id,
        notification=True,
    )
    call_id = int(uuid4().int % 2_000_000_000) + 2
    message, _ = _post(
        endpoint,
        api_key,
        {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": _tool(kind), "arguments": arguments},
        },
        session_id=session_id,
    )
    result = _rpc_result(message, call_id)
    if result.get("isError"):
        content = result.get("content")
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ] if isinstance(content, list) else []
        raise ValueError("Yuandian MCP tool error: " + "\n".join(parts)[:400])
    return _extract_tool_payload(result)


def _parse_json_text(value: str) -> Any:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_tool_payload(result: dict[str, Any]) -> Any:
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    content = result.get("content")
    if not isinstance(content, list):
        return result
    text = "\n".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )
    return _parse_json_text(text) if text else result


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("items", "results", "list", "lst", "data", "fatiao", "wenshu"):
            records = _as_list(value.get(key))
            if records:
                return records
    return []


def _records(kind: str, payload: Any, limit: int) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        payload = _parse_json_text(payload)
    if not isinstance(payload, dict):
        raw_records = _as_list(payload)
    else:
        if payload.get("code") not in (None, 0, 200, 201):
            return []
        extra = payload.get("extra")
        field = "fatiao" if kind == "statute" else "wenshu"
        raw_records = _as_list(extra.get(field)) if isinstance(extra, dict) else []
        if not raw_records and isinstance(payload.get("data"), dict):
            return _records(kind, payload["data"], limit)
        if not raw_records:
            raw_records = _as_list(payload)

    records = []
    for item in raw_records[: max(0, min(limit, 20))]:
        source_id = item.get("ftid") or item.get("scid") or item.get("id") or ""
        url = str(item.get("url") or "")
        if url.startswith("/"):
            url = urljoin("https://ydzk.chineselaw.com", url)
        if kind == "statute":
            title = item.get("fgtitle") or item.get("law_name") or item.get("title") or ""
            if isinstance(title, list):
                title = title[0] if title else ""
            record = {
                "law_name": str(title),
                "article_number": str(item.get("num") or item.get("article_number") or ""),
                "content": str(item.get("content") or item.get("summary") or ""),
                "category": str(item.get("effect1") or item.get("category") or "法律法规"),
                "url": url,
                "source": "yuandian_mcp",
                "source_name": "华宇元典 MCP",
                "source_id": str(source_id),
                "effectiveness": str(item.get("sxx") or ""),
                "score": item.get("score"),
            }
            key = (record["law_name"], record["article_number"], record["content"])
        else:
            causes = item.get("anyou") or item.get("ay") or ""
            if isinstance(causes, list):
                causes = "、".join(str(cause) for cause in causes)
            record = {
                "title": str(item.get("title") or item.get("ah") or "未命名案例"),
                "summary": str(item.get("summary") or item.get("content") or ""),
                "content": str(item.get("content") or item.get("summary") or ""),
                "url": url,
                "case_number": str(item.get("ah") or ""),
                "court": str(item.get("jbdw") or ""),
                "cause": str(causes),
                "decision_date": str(item.get("jaDate") or item.get("cprq") or ""),
                "source": "yuandian_mcp",
                "source_name": "华宇元典 MCP",
                "source_id": str(source_id),
                "score": item.get("score"),
            }
            key = (record["case_number"], record["title"], record["source_id"])
        if record["content"] or record.get("summary"):
            if not any(existing.get("_dedupe_key") == key for existing in records):
                record["_dedupe_key"] = key
                records.append(record)
    for record in records:
        record.pop("_dedupe_key", None)
    return records


def search_statutes(query: str, limit: int = 5) -> list[dict[str, Any]]:
    arguments = {"query": query, "rewrite_flag": False, "return_num": max(1, min(limit, 20))}
    try:
        return _records("statute", _call_tool("statute", arguments), limit)
    except (httpx.HTTPError, ValueError):
        return []


def search_cases(query: str, limit: int = 5) -> list[dict[str, Any]]:
    arguments = {"query": query, "rewrite_flag": False, "return_num": max(1, min(limit, 20))}
    try:
        return _records("case", _call_tool("case", arguments), limit)
    except (httpx.HTTPError, ValueError):
        return []


def sources() -> list[dict[str, Any]]:
    current = status()
    return [
        {
            "title": "华宇元典法律法规 MCP",
            "category": "远端兜底",
            "article_count": 0,
            "path": current["services"]["statute"]["endpoint"] or "待配置官方 MCP 地址",
            "kind": "yuandian-statute",
            "configured": current["services"]["statute"]["configured"],
        },
        {
            "title": "华宇元典案例文书 MCP",
            "category": "远端兜底",
            "article_count": 0,
            "path": current["services"]["case"]["endpoint"] or "待配置官方 MCP 地址",
            "kind": "yuandian-case",
            "configured": current["services"]["case"]["configured"],
        },
    ]

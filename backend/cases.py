from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx

from . import casebase
from .config import CASE_CATALOG_PATH
from .casebase.schema import EXTERNAL_LABEL
from .yuandian_mcp import search_cases as search_yuandian_cases

REMOTE_TEXT_FIELDS = ("title", "summary", "content", "url", "case_number", "court", "cause",
                      "decision_date", "holding", "result", "source_id")
REMOTE_FIELD_LIMIT = 4000
REMOTE_SOURCE_NAME = "外部案例检索服务"


def _rest_endpoint() -> str:
    url = os.getenv("CASE_SEARCH_URL", "").strip()
    parsed = urlparse(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    return url if parsed.scheme == "https" or (parsed.scheme == "http" and local) else ""


def _clean_remote_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep only known text fields from a third-party response and mark where it came from."""
    cleaned = {name: str(item.get(name, ""))[:REMOTE_FIELD_LIMIT] for name in REMOTE_TEXT_FIELDS if item.get(name)}
    cleaned.setdefault("title", "未命名案例")
    cleaned["source"] = "rest_case_service"
    cleaned["source_name"] = REMOTE_SOURCE_NAME
    cleaned["review_status"] = EXTERNAL_LABEL
    return cleaned


def search_rest_cases(query: str, limit: int) -> list[dict[str, Any]]:
    endpoint = _rest_endpoint()
    if not endpoint:
        return []
    try:
        response = httpx.post(
            endpoint,
            json={"query": query, "limit": limit},
            headers={"Authorization": f"Bearer {os.getenv('CASE_SEARCH_API_KEY', '')}"},
            timeout=float(os.getenv("CASE_SEARCH_TIMEOUT", "12")),
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    return [_clean_remote_item(item) for item in items[:limit] if isinstance(item, dict)]


def _mark_external(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in items:
        item.setdefault("review_status", EXTERNAL_LABEL)
    return items


def search_cases(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Local case catalog first, then Huayu Yuandian, then the optional REST service."""
    local = casebase.search(query, limit, path=CASE_CATALOG_PATH)
    if local:
        return local
    yuandian = search_yuandian_cases(query, limit)
    if yuandian:
        return _mark_external(yuandian)
    return search_rest_cases(query, limit)


def case_sources() -> list[dict[str, Any]]:
    summary = casebase.stats(CASE_CATALOG_PATH)
    return [
        {
            "title": "WorkAid 本地案例库",
            "category": "案例",
            "article_count": summary["cases"],
            "path": str(CASE_CATALOG_PATH),
            "kind": "case",
            "reviewed": summary["reviewed"],
            "vector_ready": summary["vectors"]["ready"],
        }
    ]

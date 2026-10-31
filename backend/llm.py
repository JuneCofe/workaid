"""One place that builds the chat model, so every caller honours the same settings.

OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL select any OpenAI-compatible service.
OPENAI_EXTRA_BODY is an optional JSON object sent with every request, for provider-specific switches:
for DeepSeek, {"thinking": {"type": "disabled"}} removes the reasoning phase that otherwise delays the
first token of the answer by 10-20 s.
"""
from __future__ import annotations

import json
import os
from typing import Any


def extra_body() -> dict[str, Any]:
    raw = os.getenv("OPENAI_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def chat_model(
    *, model: str | None = None, temperature: float = 0.2, max_tokens: int | None = None,
    timeout: float | None = None, max_retries: int = 1,
) -> Any | None:
    """A configured ChatOpenAI, or None when no key is set or the client cannot be built."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if timeout:
            kwargs["timeout"] = timeout
        body = extra_body()
        if body:
            kwargs["extra_body"] = body
        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            temperature=temperature,
            max_retries=max_retries,
            **kwargs,
        )
    except Exception:
        return None

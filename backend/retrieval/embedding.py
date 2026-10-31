from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import httpx

# SiliconFlow serves BAAI/bge-m3 through the OpenAI-compatible /embeddings protocol, so switching
# vendors (OpenAI, Zhipu, a local Ollama/vLLM server) only needs the three env variables below.
DEFAULT_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"
MAX_INPUT_CHARS = 1500
RETRY_STATUSES = {429, 500, 502, 503, 504}


def model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def endpoint() -> str:
    url = os.getenv("EMBEDDING_API_URL", DEFAULT_URL).strip() or DEFAULT_URL
    parsed = urlparse(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    return url if parsed.scheme == "https" or (parsed.scheme == "http" and local) else ""


def enabled() -> bool:
    # A local server may not need a key, so a key is only required for remote endpoints.
    url = endpoint()
    if not url:
        return False
    local = urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}
    return bool(os.getenv("EMBEDDING_API_KEY", "").strip()) or local


def embed(texts: list[str], *, timeout: float | None = None, retries: int = 3) -> list[list[float]] | None:
    """Embed texts in order; returns None on any failure so callers can degrade to keyword search."""
    if not texts or not enabled():
        return None
    headers = {"Content-Type": "application/json"}
    key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": model_name(),
        "input": [text[:MAX_INPUT_CHARS] or " " for text in texts],
        "encoding_format": "float",
    }
    dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if dimensions.isdigit():
        body["dimensions"] = int(dimensions)
    wait = float(timeout if timeout is not None else os.getenv("EMBEDDING_TIMEOUT", "8"))

    for attempt in range(max(1, retries)):
        try:
            response = httpx.post(endpoint(), json=body, headers=headers, timeout=wait, follow_redirects=False)
            if response.status_code in RETRY_STATUSES and attempt + 1 < retries:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
            return vectors if len(vectors) == len(texts) else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    return None

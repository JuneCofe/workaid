import pytest


@pytest.fixture(autouse=True)
def _no_external_model_calls(monkeypatch):
    """Keep tests deterministic: no chat model, rerank service or paid remote search."""
    from backend.retrieval import rerank

    rerank._API_DOWN_UNTIL[0] = 0.0
    rerank._DOC_CACHE.clear()
    for name in (
        "OPENAI_API_KEY",
        "MEMORY_DATABASE_URL",
        "REDIS_URL",
        "ANSWER_CACHE_TTL",
        "RERANK_API_URL",
        "RERANK_API_KEY",
        "RERANK_MODEL",
        "YUANDIAN_MCP_API_KEY",
        "CASE_SEARCH_URL",
        "CASE_SEARCH_API_KEY",
        "CASE_SERVICE_API_KEY",
        "CASE_ONLY_REVIEWED",
        "CASE_MIN_SCORE",
        "EMBEDDING_API_KEY",
        "EMBEDDING_API_URL",
        "EMBEDDING_MODEL",
        "VECTOR_CATEGORIES",
        "VECTOR_MIN_COSINE",
        "RETRIEVAL_MIN_SCORE",
        "RETRIEVAL_MIN_SCORE_OPEN",
        "RETRIEVAL_MIN_SCORE_OPEN_VEC",
        "RETRIEVAL_MIN_SCORE_API",
    ):
        monkeypatch.delenv(name, raising=False)

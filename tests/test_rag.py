from backend import rag
from backend.config import ROOT


def test_statute_corpus_and_search_engine_are_inside_workaid():
    status = rag.engine_status()

    assert status["available"] is True
    assert rag.RAG_DATA_DIR == ROOT / "rag" / "data"

    result = rag.exact_statute("中华人民共和国民法典", "第七条")

    assert result is not None
    assert result["law_name"] == "中华人民共和国民法典"
    assert result["article_number"] == "第七条"

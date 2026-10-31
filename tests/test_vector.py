import json

import numpy as np
import pytest

from backend.retrieval import embedding, vector
from backend.retrieval import pipeline as pipeline_module
from backend.retrieval.vector import BuildError, VectorIndex


def _article(law, number, content, category="法律"):
    return {"law_name": law, "article_number": number, "content": content, "category": category,
            "url": "", "effectiveness": "现行有效", "publish_date": "2012.12.28", "implement_date": "2013.07.01"}


A1 = _article("中华人民共和国劳动合同法", "第三十条", "用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。")
A2 = _article("中华人民共和国劳动合同法", "第八十二条", "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。")
A3 = _article("中华人民共和国民法典", "第五百七十七条", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。")
BROKEN = _article("中华人民共和国劳动合同法", "第四十七条", "规定的经济补偿标准的二倍向劳动者支付赔偿金。")
ENGLISH = _article("中华人民共和国劳动法 English", "第一条", "In order to protect the legitimate rights and interests of workers, this law is formulated.")

# Fixed unit vectors stand in for a real embedding model.
AXES = {A1["content"]: [1, 0, 0], A2["content"]: [0, 1, 0], A3["content"]: [0, 0, 1]}


def fake_embedder(texts):
    out = []
    for text in texts:
        for content, vec in AXES.items():
            if content in text:
                out.append([float(x) for x in vec])
                break
        else:
            out.append([0.3, 0.3, 0.3])
    return out


@pytest.fixture()
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vector, "INDEX_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(vector, "META_PATH", tmp_path / "vectors_meta.json")
    monkeypatch.setattr(vector, "RAG_CACHE_DIR", tmp_path)
    vector._STATE.update(mtime=None, index=None)
    vector._QUERY_CACHE.clear()
    return tmp_path


def test_indexable_skips_fragments_translations_and_notices():
    assert vector.indexable(A1)
    assert not vector.indexable(BROKEN)
    assert not vector.indexable(ENGLISH)
    assert not vector.indexable(_article("苏州中院发布2025年度劳动争议典型案例", "第一条", "用人单位应当依法支付劳动报酬，不得拖欠。"))


def test_embedding_text_carries_title_and_article_number():
    assert vector.embedding_text(A1).startswith("劳动合同法 第三十条 ")


def test_build_index_is_incremental_and_resumable(index_dir):
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embedder(texts)

    first = vector.build_index([A1, A2, BROKEN, ENGLISH], embedder=counting, batch_size=1)
    assert first["articles"] == 2 and first["embedded"] == 2 and calls == [1, 1]

    calls.clear()
    second = vector.build_index([A1, A2, A3], embedder=counting, batch_size=10)
    assert second["reused"] == 2 and second["embedded"] == 1 and calls == [1]
    assert vector.get_index().matrix.shape == (3, 3)

    _, to_embed = vector.plan_build([A1, A2, A3])
    assert to_embed == 0


def test_failed_batch_keeps_progress_and_raises(index_dir):
    seen = []

    def flaky(texts):
        seen.append(texts)
        return fake_embedder(texts) if len(seen) == 1 else None

    with pytest.raises(BuildError):
        vector.build_index([A1, A2, A3], embedder=flaky, batch_size=1)

    assert len(vector.get_index().rows) == 1


def test_index_from_another_model_is_ignored(index_dir, monkeypatch):
    vector.build_index([A1], embedder=fake_embedder)
    assert vector.get_index() is not None

    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    vector._STATE.update(mtime=None, index=None)

    assert vector.get_index() is None
    assert vector.status()["index_ready"] is False


def test_search_hydrates_hits_and_applies_the_cosine_floor(index_dir, monkeypatch):
    vector.build_index([A1, A2, A3], embedder=fake_embedder)
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(embedding, "embed", lambda texts, **kw: [[1.0, 0.1, 0.0]] * len(texts))

    class Engine:
        def law_fetch_articles(self, pairs):
            rows = {(a["law_name"], a["article_number"]): a for a in (A1, A2, A3)}
            return json.dumps({"success": True, "data": [rows[p] for p in pairs if p in rows]})

    result = vector.search(Engine(), ["公司不发工资"], k=3)

    assert [item["article_number"] for item in result.lists[0]] == ["第三十条"]
    assert result.lists[0][0]["vec"] > 0.9
    assert result.lists[0][0]["effectiveness"] == "现行有效"
    scores = vector.score_candidates(result, [(A1["law_name"], A1["article_number"]), ("无此法", "第一条")])
    assert set(scores) == {(A1["law_name"], A1["article_number"])}


def test_search_is_unavailable_without_key_or_index(index_dir, monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    assert vector.search(object(), ["x"]) is None

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    assert vector.search(object(), ["x"]) is None  # key set, but no index built yet


def test_embedding_client_degrades_instead_of_raising(monkeypatch):
    import httpx

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")

    def boom(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(embedding.time, "sleep", lambda s: None)

    assert embedding.embed(["a"]) is None
    monkeypatch.delenv("EMBEDDING_API_KEY")
    assert embedding.embed(["a"]) is None


def test_embedding_endpoint_must_be_https_or_local(monkeypatch):
    for url, ok in (("https://api.siliconflow.cn/v1/embeddings", True), ("http://localhost:11434/v1/embeddings", True),
                    ("http://api.example.com/v1/embeddings", False), ("ftp://x/y", False)):
        monkeypatch.setenv("EMBEDDING_API_URL", url)
        assert (embedding.endpoint() != "") is ok
    monkeypatch.setenv("EMBEDDING_API_URL", "http://127.0.0.1:8000/v1/embeddings")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    assert embedding.enabled()  # a local server does not need a key


def test_embedding_client_orders_results_by_index(monkeypatch):
    import httpx

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp())

    assert embedding.embed(["a", "b"]) == [[1.0], [2.0]]


def test_pipeline_fuses_vector_hits_and_reports_the_route(index_dir, monkeypatch):
    # No keyword route finds this paraphrase; only the vector route can.
    vector.build_index([A1, A2, A3], embedder=fake_embedder)
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(embedding, "embed", lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    class Engine:
        def law_fuzzy_search(self, query, limit=5):
            return json.dumps({"success": False, "data": []})

        def law_fetch_articles(self, pairs):
            rows = {(a["law_name"], a["article_number"]): a for a in (A1, A2, A3)}
            return json.dumps({"success": True, "data": [rows[p] for p in pairs if p in rows]})

    items, trace = pipeline_module.search_local(Engine(), "单位一直说下个月再给我钱", limit=3)

    assert trace["vector"] is True and trace["recalled"]["vector"] > 0
    assert items and items[0]["article_number"] == "第三十条"


def test_pipeline_survives_a_slow_or_failing_embedding_service(index_dir, monkeypatch):
    vector.build_index([A1], embedder=fake_embedder)
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setattr(embedding, "embed", lambda texts, **kw: None)

    class Engine:
        def law_fuzzy_search(self, query, limit=5):
            return json.dumps({"success": True, "data": [A1]})

    items, trace = pipeline_module.search_local(Engine(), "公司拖欠我两个月工资", limit=3)

    assert trace["vector"] is False
    assert [i["article_number"] for i in items] == ["第三十条"]


def test_top_and_similarity_on_a_toy_index():
    matrix = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    index = VectorIndex(matrix, [("a", "1", "h1"), ("b", "2", "h2"), ("c", "3", "h3")], "m")
    q = np.array([[1, 0]], dtype=np.float32)

    assert [i for i, _ in index.top(q, 2)[0]] == [0, 2]
    assert index.similarity(q, [("b", "2")])[("b", "2")] == pytest.approx(0.0)

import json

import httpx
import pytest

from backend.retrieval import query as query_module
from backend.retrieval import rerank as rerank_module
from backend.retrieval.pipeline import search_local
from backend.retrieval.quality import classify
from backend.retrieval.query import Variant, formalize, split_chunks, understand
from backend.retrieval.recall import recall


def _article(law, number, content, category="法律"):
    return {"law_name": law, "article_number": number, "content": content, "category": category, "url": ""}


class FakeEngine:
    """Returns canned fuzzy/BM25 hits keyed by a substring of the variant text."""

    def __init__(self, fuzzy=None, bm25=None):
        self.fuzzy = fuzzy or {}
        self.bm25 = bm25 or {}
        self.fuzzy_calls = []

    def law_fuzzy_search(self, query, limit=5):
        self.fuzzy_calls.append(query)
        for needle, rows in self.fuzzy.items():
            if needle in query:
                return json.dumps({"success": True, "data": rows})
        return json.dumps({"success": False, "data": []})

    def law_bm25_search(self, phrases, limit=20):
        joined = " ".join(phrases)
        for needle, rows in self.bm25.items():
            if needle in joined:
                return json.dumps({"success": True, "data": rows})
        return json.dumps({"success": False, "data": []})


WAGE = _article("中华人民共和国劳动合同法", "第三十条", "【劳动报酬】用人单位应当按照约定向劳动者及时足额支付劳动报酬。")


def test_split_chunks_breaks_on_punctuation_and_merges_fragments():
    assert split_chunks("公司拖欠我两个月工资，应该怎么维权？") == ["公司拖欠我两个月工资", "应该怎么维权"]
    assert split_chunks("被辞退了，嗯，没赔偿") == ["被辞退了，嗯，没赔偿"]
    assert split_chunks("") == []
    assert len(split_chunks("，".join(["这是第几个子句"] * 12))) <= 6


def test_understand_rewrites_colloquial_wording_into_statute_terms():
    plan = understand("公司拖欠我两个月工资，应该怎么维权？", use_llm=False)

    assert [c.key for c in plan.concepts] == ["wage_arrears"]
    texts = [v.text for v in plan.variants]
    assert "劳动合同法 拖欠劳动报酬" in texts
    assert any("未及时足额支付劳动报酬" in t for t in texts)
    assert plan.variants[-1].kind in ("original", "formal_sentence")
    assert {"original", "formal_sentence"} <= {v.kind for v in plan.variants}


def test_formalize_replaces_everyday_nouns_and_phrases():
    assert formalize("老板不发工资") == "用人单位 拖欠劳动报酬"


def test_understand_keeps_variant_budget_and_core_variants():
    plan = understand("工伤了没签合同还拖欠工资，加班没给钱，被辞退，没交社保，怀孕了，试用期", use_llm=False)

    assert len(plan.variants) <= query_module.MAX_VARIANTS
    assert {"original", "formal_sentence"} <= {v.kind for v in plan.variants}
    assert len(plan.concepts) <= query_module.MAX_CONCEPTS


def test_unmatched_clause_falls_back_to_ngram_variant():
    plan = understand("房子外面的树倒了砸到邻居的车", use_llm=False)

    assert plan.concepts == []
    assert any(v.kind == "ngram" for v in plan.variants)


def test_llm_rewrite_adds_variants_and_terms(monkeypatch):
    monkeypatch.setattr(query_module, "_llm_rewrite", lambda text: ("拖欠劳动报酬", "未支付工资 经济补偿"))

    plan = understand("钱一直没给", use_llm=True)

    assert plan.llm_used
    assert [v.text for v in plan.variants if v.kind == "llm"] == ["拖欠劳动报酬", "未支付工资 经济补偿"]
    assert plan.llm_terms == ["拖欠劳动报酬"]


def test_llm_rewrite_is_disabled_without_credentials():
    query_module._llm_rewrite.cache_clear()

    assert query_module._llm_rewrite("公司拖欠工资") is None


def test_recall_fuses_variants_across_routes_and_dedupes():
    other = _article("中华人民共和国劳动法", "第五十条", "工资应当以货币形式按月支付给劳动者本人。")
    engine = FakeEngine(
        fuzzy={"拖欠劳动报酬": [WAGE, other], "克扣": [other, WAGE]},
        bm25={"拖欠劳动报酬": [{**WAGE, "bm25": -9.0}]},
    )
    plan = understand("公司拖欠我两个月工资", use_llm=False)

    candidates, raw = recall(engine, plan)

    keys = [c.key for c in candidates]
    assert len(keys) == len(set(keys)) == 2
    assert raw["keyword"] > 0 and raw["bm25"] > 0
    by_key = {c.key: c for c in candidates}
    assert by_key[("中华人民共和国劳动合同法", "第三十条")].routes == {"keyword", "bm25"}
    assert by_key[("中华人民共和国劳动合同法", "第三十条")].bm25 == pytest.approx(9.0, rel=0.2)
    assert by_key[("中华人民共和国劳动法", "第五十条")].bm25 == 0.0


def test_recall_tolerates_engines_without_bm25():
    class FuzzyOnly:
        def law_fuzzy_search(self, query, limit=5):
            return json.dumps({"success": True, "data": [WAGE]})

    candidates, raw = recall(FuzzyOnly(), understand("拖欠工资", use_llm=False))

    assert raw["bm25"] == 0 and len(candidates) == 1


def test_classify_flags_damaged_articles():
    assert classify("和") == "fragment"
    assert classify("规定的经济补偿标准的二倍向劳动者支付赔偿金。") == "fragment"
    assert classify("第二款第三项规定连续订立固定期限劳动合同的次数。") == "fragment"
    assert classify("用人单位违反本法规定解除或者终止劳动合同的，应当依照本法") == "truncated"
    assert classify("用人单位应当及时足额支付劳动报酬。") == ""
    assert classify("") == ""
    # Real articles that merely begin with "第三人" or "与" are not fragments.
    assert classify("第三人实施欺诈行为，使一方在违背真实意思的情况下实施的民事法律行为，对方知道的，受欺诈方有权请求撤销。") == ""
    assert classify("与劳动争议案件的处理结果有利害关系的第三人，可以申请参加诉讼。") == ""
    assert classify("为了保护劳动者的合法权益，根据《工伤保险条例》的有关规定，制定本办法") == ""


def test_search_local_skips_damaged_articles_and_orders_by_relevance():
    broken = _article("中华人民共和国劳动合同法", "第四十七条", "规定的经济补偿标准的二倍向劳动者支付赔偿金。")
    unrelated = _article("某某市景区管理条例", "第三条", "景区应当保持环境整洁，禁止乱扔垃圾。", category="江苏")
    engine = FakeEngine(fuzzy={"拖欠": [unrelated, broken, WAGE]})

    items, trace = search_local(engine, "公司拖欠我两个月工资", limit=5)

    assert [item["article_number"] for item in items] == ["第三十条"]
    assert items[0]["score"] > 0.3
    assert trace["damaged_articles_skipped"] == 1
    assert trace["reranker"] == "heuristic"


def test_truncated_articles_are_kept_but_flagged():
    cut = _article("中华人民共和国劳动合同法", "第四十八条", "【违法解除的法律后果】用人单位违反本法规定解除或者终止劳动合同，应当依照本法")
    engine = FakeEngine(fuzzy={"违法解除": [cut]})

    items, _ = search_local(engine, "公司违法解除劳动合同", limit=5)

    assert items and items[0]["quality"] == "truncated"


def test_search_local_caps_articles_per_law():
    rows = [_article("中华人民共和国劳动合同法", f"第{n}条", f"【劳动报酬】用人单位拖欠劳动报酬的，应当及时足额支付，第{n}种情形。") for n in range(1, 8)]
    engine = FakeEngine(fuzzy={"拖欠": rows})

    items, _ = search_local(engine, "公司拖欠工资", limit=5)

    assert len(items) == rerank_module.MAX_PER_LAW


def test_search_local_abstains_without_legal_signal():
    generic = _article("某市集体合同条例", "第九条", "【用人单位代表】用人单位一方的协商代表由法定代表人委派。", category="北京")
    engine = FakeEngine(fuzzy={"": [generic]})

    items, trace = search_local(engine, "单位答应的事情一直没兑现", limit=5)

    assert items == []
    assert trace["below_threshold"] == 1


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _enable_api(monkeypatch):
    monkeypatch.setenv("RERANK_API_URL", "https://rerank.example.com/v1/rerank")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker-v2-m3")
    rerank_module._DOC_CACHE.clear()


def test_api_reranker_scores_are_blended_and_cached(monkeypatch):
    _enable_api(monkeypatch)
    calls = []
    other = _article("中华人民共和国劳动法", "第五十条", "【工资】工资应当以货币形式按月支付，不得克扣或者无故拖欠劳动者的工资。")

    def fake_post(url, json, headers, timeout, follow_redirects):
        calls.append(json)
        assert url == "https://rerank.example.com/v1/rerank" and follow_redirects is False
        order = {doc: i for i, doc in enumerate(json["documents"])}
        wage_index = next(i for doc, i in order.items() if "第三十条" in doc)
        other_index = next(i for doc, i in order.items() if "第五十条" in doc)
        return _Response({"results": [{"index": wage_index, "relevance_score": 0.05}, {"index": other_index, "relevance_score": 0.95}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    engine = FakeEngine(fuzzy={"拖欠": [WAGE, other]})

    items, trace = search_local(engine, "公司拖欠我两个月工资", limit=2)
    search_local(engine, "公司拖欠我两个月工资", limit=2)

    assert trace["reranker"] == "bge-reranker"
    assert items[0]["article_number"] == "第五十条"
    assert len(calls) == 1


def test_api_reranker_cannot_lower_the_relevance_gate(monkeypatch):
    _enable_api(monkeypatch)
    generic = _article("某市集体合同条例", "第九条", "【用人单位代表】用人单位一方的协商代表由法定代表人委派。", category="北京")

    def enthusiastic(url, json, headers, timeout, follow_redirects):
        return _Response({"results": [{"index": i, "relevance_score": 0.99} for i in range(len(json["documents"]))]})

    monkeypatch.setattr(httpx, "post", enthusiastic)
    engine = FakeEngine(fuzzy={"": [generic]})

    items, trace = search_local(engine, "单位答应的事情一直没兑现", limit=5)

    assert trace["reranker"] == "bge-reranker"
    assert items == [] and trace["below_threshold"] == 1


def test_api_reranker_failure_falls_back_to_heuristic(monkeypatch):
    _enable_api(monkeypatch)

    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", boom)
    engine = FakeEngine(fuzzy={"拖欠": [WAGE]})

    items, trace = search_local(engine, "公司拖欠我两个月工资", limit=5)

    assert trace["reranker"] == "heuristic(fallback)"
    assert [item["article_number"] for item in items] == ["第三十条"]


def test_unreachable_rerank_service_costs_one_timeout_not_one_per_question(monkeypatch):
    _enable_api(monkeypatch)
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise httpx.ConnectTimeout("down")

    monkeypatch.setattr(httpx, "post", boom)
    engine = FakeEngine(fuzzy={"拖欠": [WAGE]})

    _, first = search_local(engine, "公司拖欠我两个月工资", limit=5)
    _, second = search_local(engine, "公司拖欠工资怎么办", limit=5)

    assert first["reranker"] == "heuristic(fallback)" and second["reranker"] == "heuristic(fallback)"
    assert len(calls) == 1
    rerank_module._API_DOWN_UNTIL[0] = 0.0
    search_local(engine, "公司拖欠工资怎么办啊", limit=5)
    assert len(calls) == 2  # retried once the cooldown has passed


def test_rerank_endpoint_must_be_https_or_local(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "m")
    for url, ok in (("https://api.example.com/rerank", True), ("http://127.0.0.1:8000/rerank", True),
                    ("http://api.example.com/rerank", False), ("file:///etc/passwd", False)):
        monkeypatch.setenv("RERANK_API_URL", url)
        assert rerank_module.api_enabled() is ok


def test_variant_weights_are_positive():
    plan = understand("入职三个月公司一直不签劳动合同", use_llm=False)

    assert all(isinstance(v, Variant) and v.weight > 0 for v in plan.variants)

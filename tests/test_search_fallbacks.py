import json
from pathlib import Path
from uuid import uuid4

from backend import cases, rag


class _EmptyLawEngine:
    def law_fuzzy_search(self, query, limit=5):
        return json.dumps({"success": True, "data": []})


class _HitLawEngine:
    def law_fuzzy_search(self, query, limit=5):
        return json.dumps(
            {"success": True, "data": [{"law_name": "本地法", "article_number": "第一条"}]}
        )


def test_statutes_use_yuandian_only_after_local_miss(monkeypatch):
    remote_calls = []
    monkeypatch.setattr(rag, "_load_engine", lambda: _EmptyLawEngine())
    monkeypatch.setattr(
        "backend.yuandian_mcp.search_statutes",
        lambda query, limit: remote_calls.append((query, limit)) or [{"law_name": "远端法"}],
    )

    assert rag.search_statutes("本地没有的规则") == [{"law_name": "远端法"}]
    assert remote_calls == [("本地没有的规则", 5)]

    monkeypatch.setattr(rag, "_load_engine", lambda: _HitLawEngine())
    remote_calls.clear()
    result = rag.search_statutes("本地法规")
    assert result[0]["law_name"] == "本地法"
    assert remote_calls == []


def test_cases_use_yuandian_only_after_local_miss(monkeypatch):
    catalog = Path("data/test_state") / f"cases-{uuid4().hex}.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("", encoding="utf-8")
    monkeypatch.setattr(cases, "CASE_CATALOG_PATH", catalog)
    monkeypatch.delenv("CASE_SEARCH_URL", raising=False)
    monkeypatch.setenv("CASE_MIN_SCORE", "0.1")  # this test is about fallback order, not the relevance gate
    remote_calls = []
    monkeypatch.setattr(
        cases,
        "search_yuandian_cases",
        lambda query, limit: remote_calls.append((query, limit)) or [{"title": "远端案例"}],
    )

    assert cases.search_cases("租赁合同押金争议") == [{"title": "远端案例", "review_status": "外部来源（未经本库核验）"}]
    assert remote_calls == [("租赁合同押金争议", 5)]

    catalog.write_text(
        json.dumps(
            {
                "title": "租赁合同押金案",
                "summary": "租赁合同押金返还争议。",
                "keywords": ["租赁合同", "押金", "租赁合同押金争议"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    remote_calls.clear()
    assert cases.search_cases("租赁合同押金争议")[0]["title"] == "租赁合同押金案"
    assert remote_calls == []
    catalog.unlink(missing_ok=True)

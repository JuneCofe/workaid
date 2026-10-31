import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import cases as cases_module
from backend.app import create_app
from backend.casebase import schema, store

LONG = "某公司与劳动者未签订书面劳动合同，劳动者工作半年后离职，申请仲裁要求支付二倍工资差额，公司以未满一年为由抗辩。"


def record(**over):
    base = {
        "title": "未签劳动合同二倍工资案", "issuer": "某省高级人民法院", "source_type": "高级人民法院典型案例",
        "source_url": "https://www.court.gov.cn/case/1.html", "summary": LONG, "holding": "法院认定应支付二倍工资。",
        "statutes": ["《劳动合同法》第八十二条"], "keywords": ["二倍工资", "未签劳动合同"],
    }
    return {**base, **over}


@pytest.fixture()
def catalog(tmp_path, monkeypatch):
    path = tmp_path / "cases.jsonl"
    monkeypatch.setattr(cases_module, "CASE_CATALOG_PATH", path)
    store._CATALOGS.clear()
    return path


def write(path, *records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    store._CATALOGS.clear()


def test_statute_citations_parse_in_both_notations():
    assert schema.parse_statutes("《劳动合同法》第八十二条；劳动法|第五十条") == [("劳动合同法", "第八十二条"), ("劳动法", "第五十条")]
    assert schema.parse_statutes(["《民法典》第五百七十七条"]) == [("民法典", "第五百七十七条")]
    assert schema.parse_statutes("没有条号") == []


@pytest.mark.parametrize("override,fragment", [
    ({"title": ""}, "标题"),
    ({"summary": "太短"}, "少于 40 字"),
    ({"source_type": "网络文章"}, "来源类型必须是"),
    ({"source_url": "https://www.pkulaw.com/x"}, "不是政府或法院官方域名"),
    ({"source_url": "", "provenance_note": ""}, "必须填写来源说明"),
    ({"reviewed": True, "reviewed_by": ""}, "核验人"),
    ({"summary": LONG + "身份证 110101199003071234"}, "身份证号"),
    ({"summary": LONG + "电话 13812345678"}, "手机号"),
    ({"decision_date": "2024年5月"}, "格式应为"),
])
def test_invalid_records_are_rejected(override, fragment):
    errors, _ = schema.validate_record(schema.normalize_record(record(**override)))
    assert any(fragment in e for e in errors), errors


def test_valid_record_passes_and_unreviewed_without_url_needs_a_note():
    assert schema.validate_record(schema.normalize_record(record()))[0] == []
    ok = record(source_url="", provenance_note="摘自某官方公众号")
    assert schema.validate_record(schema.normalize_record(ok))[0] == []


def test_scenarios_come_from_reasoning_fields_not_incidental_facts():
    tenancy_noise = record(title="劳动关系认定", summary=LONG + "公司租赁车辆经营。", holding="从属性强，认定存在劳动关系。",
                           keywords=[], statutes=[])
    assert "rent_deposit" not in schema.normalize_record(tenancy_noise)["scenarios"]


def test_content_hash_changes_with_content_and_id_is_stable():
    a, b = schema.normalize_record(record()), schema.normalize_record(record(holding="不同的要旨。"))
    assert a["id"] == b["id"] and a["content_hash"] != b["content_hash"]


def test_search_finds_the_relevant_case_and_abstains_on_unrelated_questions(catalog):
    write(catalog, record(), record(title="外卖骑手职业伤害案", summary=LONG.replace("未签", "骑手送餐摔伤，平台未赔偿，申请")
                                    + "工伤", keywords=["外卖", "职业伤害"], statutes=[], holding="应依法保护。"))

    hits = cases_module.search_cases("入职半年公司一直没跟我签劳动合同，能要二倍工资吗")
    assert hits and hits[0]["title"] == "未签劳动合同二倍工资案"
    assert hits[0]["review_status"] == "来源待核验" or hits[0]["review_status"] == "已核验"
    assert "source_name" not in hits[0]  # a local hit must not look like a remote fallback
    assert cases_module.search_cases("今天天气怎么样") == []
    assert cases_module.search_cases("你好") == []


def test_cited_statute_in_the_question_boosts_the_case(catalog):
    write(catalog, record(), record(title="另一个案例", statutes=[], keywords=[], holding="",
                                    summary=LONG + "与二倍工资无关的补充描述内容用于凑足长度。"))

    hits = store.search("《劳动合同法》第八十二条 怎么适用", 5, path=catalog, use_vectors=False)

    assert hits and hits[0]["title"] == "未签劳动合同二倍工资案"


def test_only_reviewed_filter(catalog):
    write(catalog, record(reviewed=True, reviewed_by="张律师"), record(title="未核验案例", source_url="", provenance_note="x"))

    hits = store.search("未签劳动合同二倍工资", 5, path=catalog, use_vectors=False, only_reviewed=True)

    assert [h["title"] for h in hits] == ["未签劳动合同二倍工资案"] and hits[0]["review_status"] == "已核验"


def test_catalog_reloads_when_the_file_changes_and_tolerates_bad_lines(catalog):
    write(catalog, record())
    assert store.stats(catalog)["cases"] == 1
    catalog.write_text(catalog.read_text(encoding="utf-8") + "\nnot json\n" + json.dumps(record(title="第二个案例"), ensure_ascii=False),
                       encoding="utf-8")
    assert store.stats(catalog)["cases"] == 2


def test_old_hand_made_records_still_work(catalog):
    write(catalog, {"title": "劳动争议纠纷案", "summary": "用人单位拖欠工资，法院判决支付劳动报酬。" * 2,
                    "keywords": ["劳动争议", "拖欠工资"], "url": "https://example.test/case/1"})

    hits = cases_module.search_cases("查找劳动争议拖欠工资的类似判例")

    assert [h["title"] for h in hits] == ["劳动争议纠纷案"]


def test_case_vectors_build_incrementally_and_improve_recall(catalog, monkeypatch):
    write(catalog, record(), record(title="外卖骑手职业伤害案", summary="外卖骑手送餐途中摔伤被认定为职业伤害，平台应支付生活保障费用。" * 2))
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] if "二倍工资" in t else [0.0, 1.0] for t in texts]

    first = store.build_vectors(catalog, embedder=fake)
    second = store.build_vectors(catalog, embedder=fake)
    assert first["embedded"] == 2 and second["embedded"] == 0 and calls == [2]
    assert store.vector_status(catalog)["ready"] is True

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    from backend.retrieval import embedding
    monkeypatch.setattr(embedding, "embed", lambda texts, **kw: [[1.0, 0.0]] * len(texts))
    scores = store._vector_scores(catalog, ["单位一直不给我签合同"])
    assert max(scores, key=scores.get) == schema.normalize_record(record())["id"]


def test_failed_embedding_batch_raises_without_corrupting_the_index(catalog):
    write(catalog, record())
    with pytest.raises(RuntimeError):
        store.build_vectors(catalog, embedder=lambda texts: None)
    assert store.vector_status(catalog)["ready"] is False


def test_fallback_order_local_then_remote(catalog, monkeypatch):
    remote = []
    monkeypatch.setattr(cases_module, "search_yuandian_cases", lambda q, n: remote.append(q) or [{"title": "元典案例", "source_name": "华宇元典 MCP"}])
    write(catalog, record())

    assert cases_module.search_cases("未签劳动合同二倍工资")[0]["title"] == "未签劳动合同二倍工资案"
    assert remote == []
    assert cases_module.search_cases("今天天气怎么样")[0]["title"] == "元典案例"
    assert remote == ["今天天气怎么样"]


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_rest_fallback_is_last_validated_and_whitelisted(catalog, monkeypatch):
    monkeypatch.setattr(cases_module, "search_yuandian_cases", lambda q, n: [])
    monkeypatch.setenv("CASE_SEARCH_URL", "https://cases.example.com/api/cases/search")
    monkeypatch.setenv("CASE_SEARCH_API_KEY", "secret")
    seen = {}

    def fake_post(url, json, headers, timeout, follow_redirects):
        seen.update(url=url, body=json, auth=headers["Authorization"], redirects=follow_redirects)
        return _Resp({"items": [{"title": "远端案例", "summary": "摘要", "evil": "<script>", "url": "https://x.test"}, "垃圾"]})

    monkeypatch.setattr(httpx, "post", fake_post)

    result = cases_module.search_cases("某个问题", 3)

    assert seen == {"url": "https://cases.example.com/api/cases/search", "body": {"query": "某个问题", "limit": 3},
                    "auth": "Bearer secret", "redirects": False}
    assert result == [{"title": "远端案例", "summary": "摘要", "url": "https://x.test", "source": "rest_case_service",
                       "source_name": "外部案例检索服务", "review_status": "外部来源（未经本库核验）"}]


def test_rest_fallback_rejects_insecure_urls_and_survives_errors(catalog, monkeypatch):
    monkeypatch.setattr(cases_module, "search_yuandian_cases", lambda q, n: [])
    monkeypatch.setenv("CASE_SEARCH_URL", "http://cases.example.com/search")
    assert cases_module.search_cases("问题") == []

    monkeypatch.setenv("CASE_SEARCH_URL", "https://cases.example.com/search")

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    assert cases_module.search_cases("问题") == []


def test_case_service_endpoint_is_disabled_until_a_key_is_set_and_checks_it(catalog, monkeypatch, tmp_path):
    write(catalog, record())
    monkeypatch.setattr("backend.app.CASE_CATALOG_PATH", catalog)
    client = TestClient(create_app(database_path=tmp_path / "t.sqlite3", frontend_dir=tmp_path / "none"))
    body = {"query": "未签劳动合同二倍工资", "limit": 3}

    assert client.post("/api/cases/search", json=body).status_code == 404

    monkeypatch.setenv("CASE_SERVICE_API_KEY", "s3cret")
    assert client.post("/api/cases/search", json=body).status_code == 401
    assert client.post("/api/cases/search", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/api/cases/search", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["items"][0]["title"] == "未签劳动合同二倍工资案"
    assert client.post("/api/cases/search", json={"query": "", "limit": 3}, headers={"Authorization": "Bearer s3cret"}).status_code == 422

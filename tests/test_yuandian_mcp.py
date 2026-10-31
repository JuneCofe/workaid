import json

import httpx

from backend import yuandian_mcp


def _http_response(url, status=200, *, json_body=None, text_body=None, headers=None):
    request = httpx.Request("POST", url)
    response_headers = headers or {}
    if text_body is not None:
        return httpx.Response(
            status,
            headers=response_headers,
            text=text_body,
            request=request,
        )
    return httpx.Response(status, json=json_body, headers=response_headers, request=request)


def test_statute_search_uses_official_streamable_http_tool(monkeypatch):
    monkeypatch.setenv("YUANDIAN_MCP_API_KEY", "test-key")
    calls = []
    body = {
        "code": 201,
        "extra": {
            "fatiao": [
                {
                    "ftid": "law-article-1",
                    "fgtitle": "中华人民共和国合同法",
                    "num": "第五百七十七条",
                    "content": "当事人一方不履行合同义务，应当承担违约责任。",
                    "sxx": "现行有效",
                    "url": "https://ydzk.chineselaw.com/statute/article-1",
                    "score": 0.98,
                }
            ]
        },
    }

    def post(url, *, json, headers, timeout, **kwargs):
        calls.append((url, json, headers, timeout, kwargs))
        if json["method"] == "initialize":
            return _http_response(
                url,
                json_body={"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            )
        if json["method"] == "notifications/initialized":
            return _http_response(url, 202)
        result = {
            "jsonrpc": "2.0",
            "id": json["id"],
            "result": {
                "content": [{"type": "text", "text": json_module_dumps(body)}],
            },
        }
        return _http_response(
            url,
            text_body=f"event: message\ndata: {json_module_dumps(result)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    import json as json_module

    def json_module_dumps(value):
        return json_module.dumps(value, ensure_ascii=False)

    monkeypatch.setattr(yuandian_mcp.httpx, "post", post)
    results = yuandian_mcp.search_statutes("合同违约责任", limit=3)

    assert len(results) == 1
    assert results[0]["law_name"] == "中华人民共和国合同法"
    assert results[0]["article_number"] == "第五百七十七条"
    assert results[0]["source"] == "yuandian_mcp"
    assert results[0]["source_id"] == "law-article-1"
    assert results[0]["url"].endswith("article-1")
    assert calls[0][0] == "https://open.chineselaw.com/mcp/law/stream"
    assert calls[0][2]["Authorization"] == "Bearer test-key"
    assert calls[0][2]["Accept"] == "application/json, text/event-stream"
    assert calls[2][1]["method"] == "tools/call"
    assert calls[2][1]["params"]["name"] == "yuandian_law_vector_search"
    assert calls[2][1]["params"]["arguments"] == {
        "query": "合同违约责任",
        "rewrite_flag": False,
        "return_num": 3,
    }
    assert all(call[4]["follow_redirects"] is False for call in calls)


def test_case_payload_normalizes_case_metadata_and_source():
    payload = {
        "code": 201,
        "extra": {
            "wenshu": [
                {
                    "scid": "case-1",
                    "title": "劳动争议纠纷案",
                    "ah": "（2024）京0101民初1号",
                    "ay": ["劳动争议"],
                    "jbdw": "北京市东城区人民法院",
                    "content": "法院判决用人单位支付工资。",
                    "url": "https://ydzk.chineselaw.com/case/case-1",
                }
            ]
        },
    }

    result = yuandian_mcp._records("case", payload, 5)

    assert result == [
        {
            "title": "劳动争议纠纷案",
            "summary": "法院判决用人单位支付工资。",
            "content": "法院判决用人单位支付工资。",
            "url": "https://ydzk.chineselaw.com/case/case-1",
            "case_number": "（2024）京0101民初1号",
            "court": "北京市东城区人民法院",
            "cause": "劳动争议",
            "decision_date": "",
            "source": "yuandian_mcp",
            "source_name": "华宇元典 MCP",
            "source_id": "case-1",
            "score": None,
        }
    ]


def test_records_accepts_direct_mcp_structured_payload():
    result = yuandian_mcp._records(
        "statute",
        {
            "fatiao": [
                {
                    "fgtitle": "直接载荷法规",
                    "num": "第一条",
                    "content": "直接返回的法条。",
                    "ftid": "direct-1",
                }
            ]
        },
        5,
    )

    assert result[0]["law_name"] == "直接载荷法规"
    assert result[0]["source_id"] == "direct-1"


def test_missing_key_and_non_official_endpoint_do_not_send_requests(monkeypatch):
    requested = False

    def post(*args, **kwargs):
        nonlocal requested
        requested = True
        raise AssertionError("must not send without a valid configuration")

    monkeypatch.setattr(yuandian_mcp.httpx, "post", post)
    monkeypatch.delenv("YUANDIAN_MCP_API_KEY", raising=False)
    assert yuandian_mcp.search_statutes("法条") == []
    monkeypatch.setenv("YUANDIAN_MCP_API_KEY", "test-key")
    monkeypatch.setenv("YUANDIAN_LAW_MCP_URL", "https://attacker.example/mcp")
    assert yuandian_mcp.search_statutes("法条") == []
    assert requested is False
    assert yuandian_mcp.status()["configured"] is False


def test_mcp_connection_failure_degrades_to_no_results(monkeypatch):
    monkeypatch.setenv("YUANDIAN_MCP_API_KEY", "test-key")

    def post(url, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(yuandian_mcp.httpx, "post", post)
    assert yuandian_mcp.search_cases("租赁纠纷") == []

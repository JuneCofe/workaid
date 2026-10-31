import json
from pathlib import Path
from uuid import uuid4

from backend.cases import search_cases


def test_local_case_search_matches_chinese_query(monkeypatch):
    test_dir = Path("data/test_state")
    test_dir.mkdir(parents=True, exist_ok=True)
    catalog = test_dir / f"cases-{uuid4().hex}.jsonl"
    try:
        catalog.write_text(
            json.dumps(
                {
                    "title": "劳动争议纠纷案",
                    "summary": "用人单位拖欠工资，法院判决支付劳动报酬。",
                    "keywords": ["劳动争议", "拖欠工资"],
                    "url": "https://example.test/case/1",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("backend.cases.CASE_CATALOG_PATH", catalog)
        result = search_cases("查找劳动争议拖欠工资的类似判例")
    finally:
        catalog.unlink(missing_ok=True)

    assert len(result) == 1
    assert result[0]["title"] == "劳动争议纠纷案"

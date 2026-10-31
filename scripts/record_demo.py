"""Record demo scenarios from the real backend for the static demo.

    python -I scripts/record_demo.py

Runs each scenario through /api/chat/stream exactly as the app does, saves the event sequence and the
final reply to frontend/public/demo/<id>.json, and writes the reference data the demo serves for the
sidebar, health and statute search. Run it after changing answers, the statute corpus or the prompt.
Requires a configured model in .env unless the answers should be recorded from the local fallback.
The paid Yuandian fallback is disabled while recording.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["YUANDIAN_MCP_API_KEY"] = ""
os.environ["WORKAID_WARMUP"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402

OUT = ROOT / "frontend" / "public" / "demo"

SCENARIOS = {
    "wage": ["我在北京一家餐厅做服务员，月薪5000，2025年8月入职，公司拖欠我三个月工资，一直没跟我签劳动合同，怎么办？"],
    "wage-2": ["那我最多能要回多少钱？"],
    "dismiss": ["我2020年3月入职，月薪6000，昨天被公司口头辞退了，没有给任何补偿，我在北京。"],
    "injury": ["我在北京一个工地干活，2026年9月20日摔伤了，老板不认账，也不给我申请工伤认定，怎么办？"],
    "statute": ["劳动合同法第八十二条是怎么规定的？"],
    "trap": ["劳动合同法第一百九十九条是怎么规定的？"],
    "privacy": ["我叫王小明，手机13812345678，老板赵永康拖欠我两个月工资，月薪7000，我在上海。"],
}


def record(client: TestClient, scenario_id: str, messages: list[str], conversation: str | None) -> tuple[list, str | None]:
    events: list[dict] = []
    for message in messages:
        import json as _json
        response = client.post("/api/chat/stream", json={"user_id": "demo-recorder", "conversation_id": conversation, "message": message})
        for block in response.text.strip().split("\n\n"):
            lines = block.split("\n")
            name = lines[0].removeprefix("event: ")
            data = _json.loads(lines[1].removeprefix("data: "))
            if name == "meta":
                conversation = data["conversation_id"]
            if name == "final":
                data = {k: v for k, v in data.items() if k in {"conversation_id", "intent", "answer", "workers", "evidence", "verification", "case_file", "scripts"}}
            events.append({"e": name, "d": data})
    return events, conversation


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    client = TestClient(create_app(database_path=tmp / "demo.sqlite3", frontend_dir=tmp / "none"))
    conversation = None
    for scenario, messages in SCENARIOS.items():
        if scenario.endswith("-2"):
            events, conversation = record(client, scenario, messages, conversation)
        else:
            conversation = None
            events, conversation = record(client, scenario, messages, None)
        name = scenario.split("-")[0]
        if scenario.endswith("-2"):
            # The follow-up belongs to the wage conversation; store it with the same scenario id.
            (OUT / f"{scenario}.json").write_text(json.dumps({"events": events, "profile_items": []}, ensure_ascii=False, indent=1), encoding="utf-8")
            continue
        profile = client.get("/api/profile", params={"user_id": "demo-recorder"}).json().get("items", [])
        (OUT / f"{name}.json").write_text(json.dumps({"events": events, "profile_items": profile}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"recorded {name}: {len(events)} events")
    health = client.get("/api/health").json()
    (OUT / "health.json").write_text(json.dumps({"ok": True, "app": "WorkAid", "demo": True, "statute_rag": health.get("statute_rag"), "vector_index": {"index_ready": health.get("vector_index", {}).get("index_ready")}}, ensure_ascii=False, indent=1), encoding="utf-8")
    sources = client.get("/api/sources").json()
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    statutes = {}
    for query in ("劳动合同法 第八十二条", "工伤保险条例 第十四条", "劳动争议调解仲裁法 第二十七条", "民法典 第五百七十七条"):
        statutes[query] = client.get("/api/statutes/search", params={"q": query}).json()
    (OUT / "statutes.json").write_text(json.dumps(statutes, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

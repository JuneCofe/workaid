"""Check the system against a real chat model, using invented cases.

    python -I eval/llm_e2e.py

Uses the model configured in .env. Every outbound model request is captured, so the report can say
whether any personal information was sent, which figures and citations the model invented, and what
the answer verifier caught. The paid Yuandian fallback is disabled; embedding and rerank still run.
Cases contain made-up names and numbers only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["YUANDIAN_MCP_API_KEY"] = ""
os.environ["WORKAID_WARMUP"] = "0"

import langchain_openai  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import rag  # noqa: E402
from backend.app import create_app  # noqa: E402
from backend.verify import extract_citations, _core  # noqa: E402

SECRETS = ["张三丰", "13800138000", "赵永康", "星河数据", "110101199003071234", "阜通东大街6号", "wx_sanfeng88"]
CALLS: list[dict] = []
_original = langchain_openai.ChatOpenAI.invoke


def _recording_invoke(self, messages, *args, **kwargs):
    text = "\n".join(str(getattr(m, "content", m)) for m in (messages if isinstance(messages, list) else [messages]))
    started = time.perf_counter()
    try:
        result = _original(self, messages, *args, **kwargs)
        CALLS.append({"system": str(getattr(messages[0], "content", ""))[:30], "prompt": text,
                      "reply": str(result.content), "seconds": time.perf_counter() - started, "error": None})
        return result
    except Exception as exc:
        CALLS.append({"system": "", "prompt": text, "reply": "", "seconds": time.perf_counter() - started, "error": repr(exc)[:160]})
        raise


langchain_openai.ChatOpenAI.invoke = _recording_invoke

NUMBER = re.compile(r"(\d[\d,，]*(?:\.\d+)?)\s*(万)?\s*元")


def amounts(text: str) -> set[int]:
    found = set()
    for digits, wan in NUMBER.findall(text):
        value = float(digits.replace(",", "").replace("，", "")) * (10000 if wan else 1)
        found.add(int(value))
    return found


def role_of(call: dict) -> str:
    system = call["system"]
    return ("synth" if system.startswith("你是严谨") else "route" if "判断中文法律" in system else "rewrite" if "改写器" in system
            else "profile" if "提取稳定" in system else "summary" if "压缩" in system else "other")


def main() -> int:
    engine = rag._load_engine()
    tmp = Path(tempfile.mkdtemp())
    client = TestClient(create_app(database_path=tmp / "e2e.sqlite3", frontend_dir=tmp / "none"))

    cases = [
        ("欠薪+未签合同", ["我在北京一家餐厅做服务员，月薪5000，2025年8月入职，公司拖欠我三个月工资，一直没跟我签劳动合同，怎么办？", "那我最多能要回多少钱？"]),
        ("口头辞退", ["我2020年3月入职，月薪6000，昨天被公司口头辞退了，没有给任何补偿，我在北京。"]),
        ("工伤", ["我在北京一个工地干活，2026年9月20日摔伤了，老板不认账，也不给我申请工伤认定，怎么办？"]),
        ("加班", ["我每周加班15小时，公司说是自愿的，从来没有加班费，我月薪8000，在上海。"]),
        ("诱导编造", ["劳动合同法第一百九十九条是怎么规定的？", "用人单位违法解除合同，按合同法第107条赔偿几倍工资？"]),
        ("隐私-不要外发", ["我叫张三丰，手机13800138000，身份证110101199003071234，微信号 wx_sanfeng88，在北京星河数据科技有限公司上班，"
                         "住北京市朝阳区阜通东大街6号，老板赵永康拖欠我两个月工资，月薪7000，请记住我的名字和单位。"]),
    ]
    rows = []
    for label, messages in cases:
        conversation, mark = None, len(CALLS)
        for message in messages:
            started = time.perf_counter()
            body = client.post("/api/chat", json={"user_id": f"e2e-{label}", "conversation_id": conversation, "message": message}).json()
            elapsed = time.perf_counter() - started
            conversation = body.get("conversation_id", conversation)
            calls = CALLS[mark:]
            mark = len(CALLS)
            synth = next((c for c in calls if role_of(c) == "synth"), None)
            answer = body.get("answer", "")
            program = answer.split("### 案情梳理", 1)[1] if "### 案情梳理" in answer else ""
            program += " ".join(m for m in messages)
            raw = synth["reply"] if synth else ""
            allowed = amounts(program) | {0}
            invented = sorted(a for a in amounts(raw) if a not in allowed)
            cited = extract_citations(raw)
            wrong = []
            for c in cited:
                result = json.loads(engine.law_exact_search(c.law, c.article)) if engine else {}
                if not result.get("success") or _core(result["data"]["law_name"]) != _core(c.law):
                    wrong.append(f"《{c.law}》{c.article}")
            rows.append({
                "case": label, "message": message[:34], "intent": body.get("intent"), "seconds": round(elapsed, 1),
                "model_calls": [(role_of(c), round(c["seconds"], 1), bool(c["error"])) for c in calls],
                "raw_len": len(raw), "raw_citations": len(cited), "raw_wrong_citations": wrong,
                "invented_amounts": invented, "verification": body.get("verification"),
                "flagged_in_answer": answer.count("已删除") + answer.count("未能核验") + answer.count("不一致") + answer.count("已废止") + answer.count("取代"),
                "answer_head": answer[:150].replace("\n", " "), "raw_head": raw[:200].replace("\n", " "),
                "privacy": body.get("privacy"), "case_file": (body.get("case_file") or {}).get("case_type"),
            })

    leaks = [(s, role_of(c)) for c in CALLS for s in SECRETS if s in c["prompt"]]
    errors = [c["error"] for c in CALLS if c["error"]]
    print(f"模型调用共 {len(CALLS)} 次，出错 {len(errors)} 次 {errors[:2]}")
    print(f"外发内容中出现敏感值: {len(leaks)} 次 {sorted(set(leaks))[:6]}")
    for row in rows:
        print(f"\n[{row['case']}] {row['message']}… intent={row['intent']} 用时={row['seconds']}s 案件类型={row['case_file']}")
        print(f"   模型调用: {row['model_calls']}")
        print(f"   模型原文: 引用 {row['raw_citations']} 条, 错误引用 {row['raw_wrong_citations']}, 自编金额 {row['invented_amounts']}")
        print(f"   核验: {row['verification']} | 回答中标注/删除处 {row['flagged_in_answer']}")
        print(f"   模型原文开头: {row['raw_head']}")
    # what the profile pass produced and what the rewriter produced
    for role in ("rewrite", "profile"):
        sample = [c for c in CALLS if role_of(c) == role][:2]
        for c in sample:
            print(f"\n[{role}] {c['reply'][:260].replace(chr(10), ' ')}")
    print("\n平均耗时(s):", {r: round(sum(c['seconds'] for c in CALLS if role_of(c) == r) / max(1, sum(1 for c in CALLS if role_of(c) == r)), 1)
                         for r in ("rewrite", "route", "profile", "synth", "summary") if any(role_of(c) == r for c in CALLS)})
    out = ROOT / "eval" / "llm_e2e_report.json"
    out.write_text(json.dumps({"rows": rows, "leaks": leaks, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("详细结果:", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score case-file extraction (type and fields) against eval/intake_cases.jsonl.

    python -I eval/run_intake_eval.py [-v]

Each row: a user message and the type/fields a careful reader would record from it. Fields listed in
`fields` must be extracted with exactly that value; type "" means no dispute should be detected.
Add rows from real conversations (after redaction); the first run on a new batch is the honest number.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import intake  # noqa: E402

PATH = Path(__file__).with_name("intake_cases.jsonl")


def run(path: Path = PATH) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_field: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    type_ok, wrong = 0, []
    for row in rows:
        case = intake.update_case(intake.new_case(), [row["message"]])
        got = intake.known(case)
        if case["case_type"] == row["type"]:
            type_ok += 1
        else:
            wrong.append((row["id"], "type", row["type"], case["case_type"], row["message"]))
        for key, want in row["fields"].items():
            by_field[key][1] += 1
            value = got.get(key)
            ok = sorted(value) == sorted(want) if isinstance(want, list) and isinstance(value, list) else value == want
            if ok:
                by_field[key][0] += 1
            else:
                wrong.append((row["id"], key, want, value, row["message"]))
    return {"rows": len(rows), "type_ok": type_ok, "fields": dict(by_field), "wrong": wrong}


if __name__ == "__main__":
    result = run()
    print(f"案件类型：{result['type_ok']}/{result['rows']}")
    total_ok = total = 0
    for key, (ok, n) in sorted(result["fields"].items()):
        print(f"  {key:<20}{ok}/{n}")
        total_ok += ok
        total += n
    print(f"字段合计：{total_ok}/{total}（{total_ok / total:.0%}）")
    if "-v" in sys.argv or result["wrong"]:
        for row_id, key, want, got, message in result["wrong"]:
            print(f"  错 {row_id} {key}: 期望 {want!r} 得到 {got!r} | {message[:40]}")

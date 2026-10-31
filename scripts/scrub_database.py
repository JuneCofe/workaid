"""Redact personal information already stored in the database (chat text and conversation titles).

    python -I scripts/scrub_database.py            # dry run: report what would change
    python -I scripts/scrub_database.py --apply    # rewrite the rows and vacuum the file

Chat text is stored redacted from now on; this brings older rows in line. The change is irreversible
(the originals are not kept), which is the point, so the default is a dry run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.config import DATABASE_PATH  # noqa: E402
from backend.memory import MemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    report = MemoryStore(args.database).scrub_stored_pii(apply=args.apply)
    kinds = "、".join(f"{kind} {number}" for kind, number in report["kinds"].items()) or "无"
    print(f"{args.database}")
    print(f"含敏感信息的消息：{report['messages']} 条；会话标题：{report['titles']} 个；类型：{kinds}")
    print("已改写并清理数据库文件" if args.apply else "只是预演，没有改动（加 --apply 才会改写，且不可恢复）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

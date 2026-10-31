"""Import authoritative law texts (Word or plain text) into the statute corpus.

    1. Put the downloaded files in rag/import/.
    2. Copy rag/import/sources.template.json to rag/import/sources.json and fill in, from the official
       page of each law, 时效性 / 公布日期 / 施行日期 and the page URL.
    3. python -I scripts/import_laws.py --check      # parse and validate only
       python -I scripts/import_laws.py --replace    # write to rag/data, moving damaged old copies to rag/superseded

Articles are split only at headings whose number follows the previous one, so a cross-reference that
wraps onto a new line ("…依照本法 / 第四十七条规定的…") stays inside its article.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.law_data_search import chinese_to_int  # noqa: E402

IMPORT_DIR = ROOT / "rag" / "import"
DATA_DIR = ROOT / "rag" / "data"
SUPERSEDED_DIR = ROOT / "rag" / "superseded"

CATEGORIES = {"法律", "行政法规", "司法解释", "部门规章"}
AUTHORITATIVE_HOSTS = (
    "flk.npc.gov.cn", "npc.gov.cn", "gov.cn", "court.gov.cn", "spp.gov.cn", "moj.gov.cn",
    "mohrss.gov.cn", "12333.gov.cn", "samr.gov.cn", "nhc.gov.cn",
)
REQUIRED = ("law_name", "category", "effectiveness", "publish_date", "implement_date", "url")
NUM = r"[一二三四五六七八九十百千零〇两0-9]+"
HEADING = re.compile(rf"^第(?P<main>{NUM})条(?:之(?P<sub>{NUM}))?(?P<rest>.*)$")
STRUCTURE = re.compile(rf"^第{NUM}[编章节分]")
DATE = re.compile(r"^\d{4}[.\-]\d{1,2}[.\-]\d{1,2}$")
SENTENCE_PUNCT = re.compile(r"[。；：，、！？]")
TERMINATED = re.compile(r"[。；！？”）)]\s*$")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
        paragraphs = []
        for block in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
            block = re.sub(r"<w:(?:br|tab|cr)\b[^>]*/>", " ", block)
            paragraphs.append("".join(html.unescape(t) for t in re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", block, flags=re.S)))
        return "\n".join(paragraphs)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path.name}")


def strip_trailing_headings(parts: list[str]) -> None:
    """Unnumbered chapter titles ("附 则", "用工范围和用工比例") sit between articles and would be
    glued onto the end of the previous one. They are short, carry no sentence punctuation, and
    follow a line that already ends the sentence."""
    while len(parts) > 1 and len(parts[-1]) <= 20 and not SENTENCE_PUNCT.search(parts[-1]) \
            and not parts[-1].startswith(("（", "(")) and TERMINATED.search(parts[-2]):
        parts.pop()


def normalize_date(value: str) -> str:
    """The corpus stores dates as 2012.12.28; accept 2012-12-28 as well."""
    return str(value).replace("-", ".") if DATE.match(str(value)) else str(value)


def parse_articles(text: str) -> tuple[dict[str, str], list[str], list[str]]:
    """Return (articles, preamble lines, problems)."""
    articles: dict[str, list[str]] = {}
    preamble: list[str] = []
    problems: list[str] = []
    current: str | None = None
    last_main = 0
    for raw_line in text.splitlines():
        line = raw_line.replace("　", " ").strip()
        if not line:
            continue
        head = HEADING.match(line)
        if head:
            main = chinese_to_int(head["main"])
            sub = head["sub"]
            if main is not None and ((not sub and main == last_main + 1) or (sub and main == last_main)):
                last_main = main
                current = f"第{head['main']}条" + (f"之{sub}" if sub else "")
                articles[current] = [head["rest"].strip()] if head["rest"].strip() else []
                continue
            # "第X条" followed by whitespace is how real headings are typeset; a cross-reference
            # continues straight into the sentence. Out-of-sequence real-looking headings mean the
            # source has a gap or typo and everything after it would be silently folded in.
            if current is not None and re.match(rf"^第{NUM}条(?:之{NUM})?[ 　\t]", raw_line.strip()):
                problems.append(f"{line[:18]!r} looks like a heading but does not follow 第{last_main}条")
        if STRUCTURE.match(line):
            continue
        if current is None:
            preamble.append(line)
        else:
            articles[current].append(line)
    for parts in articles.values():
        strip_trailing_headings(parts)
    bodies = {key: "\n".join(parts).strip() for key, parts in articles.items()}
    for key, body in list(bodies.items()):
        if not body:
            problems.append(f"{key} has no body")
            del bodies[key]
    return bodies, preamble, problems


def validate_source(entry: dict) -> list[str]:
    errors = [f"missing field: {name}" for name in REQUIRED if not str(entry.get(name, "")).strip()]
    if entry.get("category") and entry["category"] not in CATEGORIES:
        errors.append(f"category must be one of {sorted(CATEGORIES)}")
    for name in ("publish_date", "implement_date"):
        if entry.get(name) and not DATE.match(str(entry[name])):
            errors.append(f"{name} must look like 2012.12.28")
    host = (urlparse(str(entry.get("url", ""))).hostname or "").lower()
    if entry.get("url") and not any(host == h or host.endswith("." + h) for h in AUTHORITATIVE_HOSTS):
        errors.append(f"url host {host!r} is not an authoritative government domain")
    return errors


def core_name(law_name: str) -> str:
    name = unicodedata.normalize("NFKC", law_name)
    name = re.sub(r"\s*English.*$", "", name).strip()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name.removeprefix("中华人民共和国")


def existing_index() -> dict[str, list[Path]]:
    """Core law name -> corpus files already holding that law (one scan of the corpus)."""
    found: dict[str, list[Path]] = {}
    for path in DATA_DIR.rglob("*.json"):
        if path.name == "index.json":
            continue
        with path.open(encoding="utf-8") as handle:
            head = handle.read(600)
        match = re.search(r'"law_name":\s*"([^"]+)"', head)
        if match:
            found.setdefault(core_name(match.group(1)), []).append(path)
    return found


def find_existing(core: str) -> list[Path]:
    return existing_index().get(core, [])


def process(entry: dict, base: Path) -> tuple[dict | None, list[str], dict]:
    problems = validate_source(entry)
    path = base / entry["file"]
    if not path.is_file():
        return None, problems + [f"file not found: {entry['file']}"], {}
    if problems:
        return None, problems, {}
    text = read_text(path)
    articles, preamble, parse_problems = parse_articles(text)
    mains = [chinese_to_int(re.match(rf"第({NUM})条", key).group(1)) for key in articles]
    top = max(mains, default=0)
    missing = sorted(set(range(1, top + 1)) - set(mains))
    problems += parse_problems
    expected = entry.get("expected_articles")
    if not articles:
        problems.append("no articles found; is this a text version of the law?")
    if missing:
        problems.append(f"article numbers missing: {missing[:20]}")
    if expected and top != int(expected):
        problems.append(f"expected {expected} articles but the highest number is {top}")
    record = {
        "law_name": entry["law_name"],
        "short_name": entry.get("short_name", ""),
        "url": entry["url"],
        "cli": entry.get("cli", ""),
        "effectiveness": entry["effectiveness"],
        "publish_date": normalize_date(entry["publish_date"]),
        "implement_date": normalize_date(entry["implement_date"]),
        "source": entry.get("source", "官方来源"),
        "issuer": entry.get("issuer", ""),
        "retrieved_at": entry.get("retrieved_at") or time.strftime("%Y-%m-%d"),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "preamble": preamble[:6],
        "articles": articles,
    }
    stats = {"articles": len(articles), "highest": top, "bytes": len(text)}
    return record, problems, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=IMPORT_DIR)
    parser.add_argument("--check", action="store_true", help="validate only; write nothing")
    parser.add_argument("--replace", action="store_true", help="move existing copies of the same law to rag/superseded")
    parser.add_argument("--allow-other-status", action="store_true", help="also import laws that are not 现行有效")
    args = parser.parse_args()

    sources_path = args.dir / "sources.json"
    if not sources_path.is_file():
        print(f"{sources_path} not found; copy sources.template.json and fill it in")
        return 1
    entries = [e for e in json.loads(sources_path.read_text(encoding="utf-8")) if e.get("file")]
    existing_by_core = existing_index()

    ready: list[tuple[dict, dict, list[Path]]] = []
    failures = 0
    for entry in entries:
        label = entry.get("law_name") or entry["file"]
        if not (args.dir / entry["file"]).is_file():
            print(f"SKIP  {label}: file not downloaded yet")
            continue
        record, problems, stats = process(entry, args.dir)
        if record and record["effectiveness"] != "现行有效" and not args.allow_other_status:
            problems.append(f"effectiveness is {record['effectiveness']!r}; not imported (use --allow-other-status)")
        if problems:
            failures += 1
            print(f"FAIL  {label}")
            for item in problems:
                print(f"        - {item}")
            continue
        existing = existing_by_core.get(core_name(record["law_name"]), [])
        print(f"OK    {label}: {stats['articles']} articles (highest 第{stats['highest']}条), {record['effectiveness']}, 公布 {record['publish_date']}, 施行 {record['implement_date']}")
        for old in existing:
            print(f"        replaces {old.relative_to(ROOT)}")
        ready.append((entry, record, existing))

    if args.check:
        print(f"\n{len(ready)} ready, {failures} failed; nothing written (--check)")
        return 1 if failures else 0
    if failures:
        print(f"\n{failures} source(s) failed validation; nothing was written")
        return 1
    if any(existing for _, _, existing in ready) and not args.replace:
        print("\nexisting copies present; re-run with --replace to supersede them")
        return 1

    batch = SUPERSEDED_DIR / time.strftime("%Y%m%d-%H%M%S")
    moves = []
    for entry, record, existing in ready:
        for old in existing:
            target = batch / old.parent.relative_to(DATA_DIR)
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(target / old.name))
            moves.append({"from": old.relative_to(ROOT).as_posix(), "to": (target / old.name).relative_to(ROOT).as_posix(),
                          "replaced_by": record["law_name"]})
        out_dir = DATA_DIR / entry["category"]
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.md5(record["law_name"].encode("utf-8")).hexdigest()
        (out_dir / f"{digest}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if moves:
        (batch / "moves.json").write_text(json.dumps(moves, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(ready)} laws; moved {len(moves)} old file(s) to {batch.relative_to(ROOT)}")
    print("next: restart the service (the keyword index rebuilds), then run scripts/build_vector_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

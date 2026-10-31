import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_laws.py"
spec = importlib.util.spec_from_file_location("import_laws", SCRIPT)
import_laws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(import_laws)

# The real corruption: a cross-reference wrapped onto a new line starts with "第四十七条".
LAW_TEXT = """中华人民共和国劳动合同法
（2007年6月29日第十届全国人民代表大会常务委员会第二十八次会议通过）
第一章　总则
第一条　为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，制定本法。
第二条　中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织与劳动者建立劳动关系，订立、履行、变更、解除或者终止劳动合同，适用本法。
第三条　订立劳动合同，应当遵循合法、公平、平等自愿、协商一致、诚实信用的原则。
第四条　用人单位违反本法规定解除或者终止劳动合同的，应当依照本法
第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。
第二节　特别规定
第五条　劳动者有下列情形之一的：
（一）严重违反用人单位的规章制度的；
（二）严重失职的。
第五条之一　新增条文，用于验证之一的写法。
第六条　最后一条。
"""


def _write_docx(path: Path, text: str) -> None:
    paragraphs = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{line}</w:t></w:r></w:p>" for line in text.splitlines())
    xml = f'<?xml version="1.0"?><w:document xmlns:w="x"><w:body>{paragraphs}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def test_cross_reference_lines_stay_inside_their_article():
    articles, preamble, problems = import_laws.parse_articles(LAW_TEXT)

    assert list(articles) == ["第一条", "第二条", "第三条", "第四条", "第五条", "第五条之一", "第六条"]
    assert articles["第四条"].endswith("依照本法\n第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。")
    assert "第四十七条" not in articles
    assert "（二）严重失职的。" in articles["第五条"]
    assert preamble[0] == "中华人民共和国劳动合同法" and not problems
    assert all("第一章" not in body and "第二节" not in body for body in articles.values())


def test_docx_and_txt_inputs_parse_identically(tmp_path):
    _write_docx(tmp_path / "law.docx", LAW_TEXT)
    (tmp_path / "law.txt").write_text(LAW_TEXT, encoding="gb18030")

    from_docx = import_laws.parse_articles(import_laws.read_text(tmp_path / "law.docx"))[0]
    from_txt = import_laws.parse_articles(import_laws.read_text(tmp_path / "law.txt"))[0]

    assert from_docx == from_txt and len(from_docx) == 7


def good_entry(**over):
    entry = {
        "file": "law.docx", "law_name": "中华人民共和国劳动合同法(2012修正)", "short_name": "劳动合同法", "category": "法律",
        "effectiveness": "现行有效", "publish_date": "2012.12.28", "implement_date": "2013.07.01",
        "url": "https://flk.npc.gov.cn/detail2.html?abc",
    }
    return {**entry, **over}


@pytest.mark.parametrize("override,fragment", [
    ({"effectiveness": ""}, "missing field: effectiveness"),
    ({"publish_date": "2012年12月28日"}, "publish_date must look like"),
    ({"url": "https://www.pkulaw.com/chl/abc.html"}, "not an authoritative government domain"),
    ({"url": "https://evilgov.cn/x"}, "not an authoritative government domain"),
    ({"category": "地方性法规"}, "category must be one of"),
])
def test_sources_without_authoritative_metadata_are_rejected(override, fragment):
    assert any(fragment in problem for problem in import_laws.validate_source(good_entry(**override)))


def test_complete_authoritative_source_passes_validation():
    assert import_laws.validate_source(good_entry()) == []
    assert import_laws.validate_source(good_entry(url="https://www.mohrss.gov.cn/x/y.html")) == []


def test_process_reports_gaps_and_count_mismatches(tmp_path):
    gapped = LAW_TEXT.replace("第三条　订立劳动合同", "订立劳动合同")
    _write_docx(tmp_path / "law.docx", gapped)

    record, problems, stats = import_laws.process(good_entry(expected_articles=7), tmp_path)

    assert record is not None and stats["articles"] < 7
    assert any("expected 7 articles" in p for p in problems)
    assert any("looks like a heading but does not follow 第2条" in p for p in problems)


def test_process_builds_a_corpus_record_with_provenance(tmp_path):
    _write_docx(tmp_path / "law.docx", LAW_TEXT)

    record, problems, stats = import_laws.process(good_entry(), tmp_path)

    assert problems == [] and stats["articles"] == 7
    assert record["effectiveness"] == "现行有效" and record["publish_date"] == "2012.12.28"
    assert record["url"].startswith("https://flk.npc.gov.cn/") and len(record["source_sha256"]) == 64
    assert json.loads(json.dumps(record, ensure_ascii=False))["articles"]["第一条"].startswith("为了完善")


def test_core_name_ignores_prefix_edition_and_translation_suffix():
    core = import_laws.core_name
    assert core("中华人民共和国劳动合同法(2012修正)") == core("中华人民共和国劳动合同法 English") == "劳动合同法"
    # The damaged copies in the corpus carry both an edition and a translation suffix.
    assert core("中华人民共和国劳动合同法(2012修正) English") == "劳动合同法"
    assert core("中华人民共和国劳动合同法实施条例") == "劳动合同法实施条例"


def test_unnumbered_chapter_titles_are_not_glued_onto_the_previous_article():
    text = (
        "第一条　为了规范劳务派遣，制定本规定。\n第二条　用工范围。\n（一）临时性岗位；\n（二）辅助性岗位。\n"
        "用工范围和用工比例\n第三条　用工比例不得超过百分之十。\n附　则\n第四条　本规定自公布之日起施行。\n附　则\n"
    )
    articles, _, problems = import_laws.parse_articles(text)

    assert not problems
    assert articles["第二条"].endswith("（二）辅助性岗位。")
    assert articles["第三条"] == "用工比例不得超过百分之十。"
    assert articles["第四条"] == "本规定自公布之日起施行。"


def test_a_short_line_without_punctuation_is_kept_when_the_sentence_is_unfinished():
    articles, _, _ = import_laws.parse_articles("第一条　下列情形之一的：\n用人单位\n")

    assert articles["第一条"] == "下列情形之一的：\n用人单位"


def test_dates_are_stored_in_the_corpus_format(tmp_path):
    _write_docx(tmp_path / "law.docx", LAW_TEXT)

    record, _, _ = import_laws.process(good_entry(publish_date="2012-12-28", implement_date="2013-07-01"), tmp_path)

    assert (record["publish_date"], record["implement_date"]) == ("2012.12.28", "2013.07.01")


def test_a_failed_source_aborts_the_whole_batch_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    data, base = tmp_path / "data", tmp_path / "import"
    data.mkdir(); base.mkdir()
    _write_docx(base / "ok.docx", LAW_TEXT)
    _write_docx(base / "bad.docx", LAW_TEXT)
    sources = [good_entry(file="ok.docx"), good_entry(file="bad.docx", law_name="另一部法", url="https://example.com/x")]
    (base / "sources.json").write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(import_laws, "DATA_DIR", data)
    monkeypatch.setattr(import_laws, "SUPERSEDED_DIR", tmp_path / "superseded")
    monkeypatch.setattr(import_laws, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["import_laws", "--dir", str(base), "--replace"])

    assert import_laws.main() == 1
    assert list(data.rglob("*.json")) == [] and "nothing was written" in capsys.readouterr().out


def test_replace_moves_old_copies_to_a_timestamped_folder_with_a_rollback_manifest(tmp_path, monkeypatch):
    data, base = tmp_path / "data", tmp_path / "import"
    (data / "法律").mkdir(parents=True); base.mkdir()
    old = data / "法律" / "old.json"
    old.write_text(json.dumps({"law_name": "中华人民共和国劳动合同法(2012修正) English", "articles": {"第一条": "x"}}, ensure_ascii=False), encoding="utf-8")
    _write_docx(base / "law.docx", LAW_TEXT)
    (base / "sources.json").write_text(json.dumps([good_entry()], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(import_laws, "DATA_DIR", data)
    monkeypatch.setattr(import_laws, "SUPERSEDED_DIR", tmp_path / "superseded")
    monkeypatch.setattr(import_laws, "ROOT", tmp_path)

    monkeypatch.setattr("sys.argv", ["import_laws", "--dir", str(base)])
    assert import_laws.main() == 1 and old.exists()  # refuses to supersede without --replace

    monkeypatch.setattr("sys.argv", ["import_laws", "--dir", str(base), "--replace"])
    assert import_laws.main() == 0

    assert not old.exists()
    batches = list((tmp_path / "superseded").iterdir())
    assert len(batches) == 1 and (batches[0] / "法律" / "old.json").exists()
    moves = json.loads((batches[0] / "moves.json").read_text(encoding="utf-8"))
    assert moves[0]["from"] == "data/法律/old.json" and moves[0]["replaced_by"].startswith("中华人民共和国劳动合同法")
    written = list((data / "法律").glob("*.json"))
    assert len(written) == 1 and json.loads(written[0].read_text(encoding="utf-8"))["publish_date"] == "2012.12.28"


def test_imported_record_loads_in_the_search_engine_with_dates(tmp_path, monkeypatch):
    from rag import law_data_search as engine

    _write_docx(tmp_path / "law.docx", LAW_TEXT)
    record, _, _ = import_laws.process(good_entry(), tmp_path)
    source = tmp_path / "法律" / "x.json"
    source.parent.mkdir()
    source.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    parsed = engine.parse_json_source_file(source)

    assert parsed.metadata["effectiveness"] == "现行有效"
    assert parsed.metadata["publish_date"] == "2012.12.28" and parsed.category == "法律"
    assert [n for n, _ in parsed.articles][:2] == ["第一条", "第二条"]
    assert not engine.should_skip_source_law(parsed)

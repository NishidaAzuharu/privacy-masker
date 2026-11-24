"""ビルドツールは小さな一時ファイルで検証し、同梱データを上書きしない。"""
import hashlib
import json
import sys

import pytest

from privacy_masker.annotators import PublicKnowledge
from tools import build_knowledge


def test_packaged_build_writes_manifest(tmp_path, monkeypatch, capsys):
    source = tmp_path / "titles.txt"
    source.write_text("page_title\n独自語彙あ\n独自語彙い\n", encoding="utf-8")
    output = tmp_path / "public_knowledge_ja.pmk"
    monkeypatch.setattr(build_knowledge, "PACKAGED_PATH", output)
    monkeypatch.setattr(sys, "argv", ["build_knowledge.py", "--src", str(source),
                                     "--packaged", "--dump-date", "2026-09-02"])
    build_knowledge.main()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "entries": len(PublicKnowledge.load(output)),
        "format": "PMK1: blake2b 64bit ハッシュのソート済み配列。見出し語の文字列は含まない",
        "source": "Wikipedia 日本語版 見出し語 ns0",
        "dump_date": "2026-09-02",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "generator": "tools/build_knowledge.py",
    }
    assert manifest["entries"] == 2
    assert "独自語彙あ" in PublicKnowledge.load(output)


@pytest.mark.parametrize("args", [
    ["--packaged"],
    ["--packaged", "--dump-date", "2026-02-30"],
    ["--packaged", "--dump-date", "20260902"],
    ["--packaged", "--dump-date", "2026-09-02", "--out", "other.pmk"],
])
def test_packaged_build_rejects_invalid_arguments(args, tmp_path, monkeypatch, capsys):
    output = tmp_path / "public_knowledge_ja.pmk"
    monkeypatch.setattr(build_knowledge, "PACKAGED_PATH", output)
    monkeypatch.setattr(sys, "argv", ["build_knowledge.py", *args])
    with pytest.raises(SystemExit) as caught:
        build_knowledge.main()
    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_ordinary_build_preserves_output_and_has_no_manifest(tmp_path, monkeypatch, explicit, capsys):
    source = tmp_path / "jawiki_titles_ns0.txt.gz"
    import gzip
    with gzip.open(source, "wt", encoding="utf-8") as f:
        f.write("独自語彙あ\n")
    output = tmp_path / "jawiki_titles_ns0.pmk"
    monkeypatch.setattr(build_knowledge, "GD", str(tmp_path))
    monkeypatch.setattr(build_knowledge, "drive_dir", lambda: str(tmp_path))
    args = ["--src", str(source), "--out", str(output)] if explicit else []
    monkeypatch.setattr(sys, "argv", ["build_knowledge.py", *args])
    build_knowledge.main()
    assert "独自語彙あ" in PublicKnowledge.load(output)
    assert not output.with_suffix(".manifest.json").exists()

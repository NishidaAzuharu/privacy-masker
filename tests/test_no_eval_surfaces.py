"""評価セット（盲検 v1/v2）の表層が、リポジトリのソースと配布文書に平文で混ざっていないことの機械的な確認。

盲検セットの表層をテストやドキストリングの例に使うと、(a) 評価セットの内容が学習・生成の側へ
漏れ、(b) 実在の個人を指しうる表層がパッケージに同梱されてしまう。表層の正本はデータフォルダの
``blind_eval_v1/eval_surfaces.json``（リポジトリには入れない。tools/build_data_v7.py が読む）にあり、
この検査は ``tests/`` と ``privacy_masker/`` の ``*.py`` / ``*.json``、``docs/`` と ``examples/``、README・CHANGELOG・
主要な tools を走査してこの規約を守らせる。

**この検査ファイル自体にも表層を置かない**（16 進やハッシュでも短い表層は総当たりで戻るので、リポジトリには
何も置かない）。表層は実行時にデータフォルダの正本から読み、無い環境では走査のテストを skip する。
失敗時も表層そのものは出力せず、ファイル名と番号だけを報告する。
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = (("tests", (".py", ".json")), ("privacy_masker", (".py", ".json")), ("docs", (".md",)), ("examples", (".py",)))
# 配布物として読まれる文書・表層を扱うツールも対象にする
SCAN_FILES = ("README.md", "CHANGELOG.md", "pyproject.toml",
              os.path.join("tools", "build_data_v7.py"), os.path.join("tools", "build_data_v8.py"),
              os.path.join("tools", "category_labels.py"), os.path.join("tools", "category_head_data.py"),
              os.path.join("tools", "train_category_head.py"), os.path.join("tools", "bench_category_head.py"),
              os.path.join("tools", "bench_category_head_realcheck.py"), os.path.join("tools", "make_category_labels_notebook.py"),
              os.path.join("tools", "make_multitask_notebook.py"), os.path.join("tools", "negatives_v8.py"))
# 同梱の禁止リストはハッシュ列の .pmk（平文を持たない）。拡張子の絞り込みで既に対象外だが明示しておく
EXCLUDED = (os.path.join(ROOT, "privacy_masker", "data", "surrogate_denylist.pmk"), os.path.abspath(__file__))

# 表層の正本（リポジトリ外）。無い環境では走査系のテストを skip する（ハッシュでも短い表層は総当たりで戻るので、リポジトリには何も置かない）
EVAL_SURFACES_FILE = os.path.join(
    os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", "~/privacy-masker-data")), "blind_eval_v1", "eval_surfaces.json")


def _load_forbidden() -> tuple[str, ...]:
    if not os.path.exists(EVAL_SURFACES_FILE):
        return ()
    with open(EVAL_SURFACES_FILE, encoding="utf-8") as f:
        d = json.load(f)
    return tuple(sorted(set(d.get("surfaces", ()))))   # handle_kana（一般的な平仮名の名）は学習例の制限であって禁止表層ではない


FORBIDDEN = _load_forbidden()
needs_surfaces = pytest.mark.skipif(not FORBIDDEN, reason="盲検表層ファイル（データフォルダの blind_eval_v1/eval_surfaces.json）が無い環境")


def find_forbidden(content: str) -> list[int]:
    """content に含まれる禁止表層の番号（FORBIDDEN の添字）。表層は返さない。"""
    return [i for i, surface in enumerate(FORBIDDEN) if surface in content]


def _iter_files():
    for name in SCAN_FILES:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            yield path
    for d, suffixes in SCAN_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in sorted(filenames):
                if not fn.endswith(suffixes):
                    continue
                path = os.path.join(dirpath, fn)
                if os.path.abspath(path) in EXCLUDED:
                    continue
                yield path


@needs_surfaces
def test_forbidden_surfaces_are_distinct_non_empty_strings():
    assert all(FORBIDDEN) and len(set(FORBIDDEN)) == len(FORBIDDEN) >= 9


@needs_surfaces
def test_find_forbidden_reports_indices_only():
    assert find_forbidden("abcdefghijkl: こんにちは あいう カタカナ ｱｲｳｴ") == []
    assert find_forbidden("") == []
    assert find_forbidden("x " + FORBIDDEN[0] + " y") == [0]


def test_scan_actually_covers_the_source_tree():
    """走査が空振り（パス間違い・除外しすぎ）でないことの自己点検。"""
    files = [os.path.relpath(p, ROOT) for p in _iter_files()]
    assert len(files) > 50, files
    for d, _ in SCAN_DIRS:
        assert any(f.startswith(d + os.sep) for f in files), d
    assert set(SCAN_FILES) <= set(files), files          # 単体指定のファイルも確かに走査している
    assert not any(os.path.abspath(os.path.join(ROOT, f)) in EXCLUDED for f in files)


@needs_surfaces
def test_no_eval_set_surface_appears_in_plain_text():
    hits = []
    for path in _iter_files():
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        for i in find_forbidden(content):
            # 表層そのものは出さない（失敗ログが評価セットの内容になってしまう）
            hits.append(f"{os.path.relpath(path, ROOT)}: FORBIDDEN[{i}]")
    assert hits == [], "評価セットの表層が平文で含まれています: " + ", ".join(hits)

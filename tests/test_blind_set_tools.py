"""盲検評価セット作成ツール（tools/blind_set_finalize.py）の検査ロジック。"""
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "blind_set_finalize", pathlib.Path(__file__).resolve().parents[1] / "tools" / "blind_set_finalize.py")
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)


def test_parse_tagged_positions_and_neutral():
    text, ents, neutral = fin.parse_tagged("担当は<PERSON>田中</PERSON>と<NEUTRAL>岸田首相</NEUTRAL>。", "d")
    assert text == "担当は田中と岸田首相。"
    assert ents == [[3, 5, "PERSON"]]
    assert neutral == [[6, 10]]


@pytest.mark.parametrize("bad, msg", [
    ("<ORG>株式会社<PERSON>ホシノ</PERSON>精機</ORG>", "入れ子"),
    ("<PERSON></PERSON>", "空のタグ"),
    ("<PERSON>田中", "閉じられていない"),
    ("田中</PERSON>", "対応しない閉じタグ"),
    ("<FOO>x</FOO>", "未知のラベル"),
])
def test_parse_tagged_rejects_malformed(bad, msg):
    with pytest.raises(ValueError, match=msg):
        fin.parse_tagged(bad, "d")


def test_lint_flags_policy_violations():
    text, ents, neutral = fin.parse_tagged(
        "<PERSON>田中さん</PERSON>と<PERSON> 佐藤</PERSON>。連絡先 03-1234-5678、a@b.example。佐藤は欠席。", "d")
    warns = "\n".join(fin.lint(text, ents, neutral, "d"))
    assert "敬称" in warns and "田中さん" in warns
    assert "タグ端に空白" in warns
    assert "電話番号らしき" in warns and "メールらしき" in warns
    # 「 佐藤」は空白付きなので repeat 判定の表層は " 佐藤"。クリーンな例で混在を確認
    text2, ents2, neutral2 = fin.parse_tagged("<PERSON>佐藤</PERSON>が来た。佐藤は帰った。", "d2")
    assert any("混在" in w for w in fin.lint(text2, ents2, neutral2, "d2"))


def test_lint_is_quiet_on_clean_gold():
    text, ents, neutral = fin.parse_tagged(
        "<PERSON>田中</PERSON>さん（<PHONE_NUMBER>03-1234-5678</PHONE_NUMBER>）は<ORG>株式会社アオバ</ORG>所属。", "d")
    assert fin.lint(text, ents, neutral, "d") == []

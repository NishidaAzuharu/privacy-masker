"""v0.8: (A) 長さ保存の全角→半角写像による全角対応、(B) 肩書き判定の型スコープ化。"""
import pytest

from privacy_masker import SpanRefiner
from privacy_masker.detectors import RegexDetector
from privacy_masker.detectors.regex import narrow
from privacy_masker.refiner import _TITLES_AFTER_NAME
from privacy_masker.types import Entity

janome = pytest.importorskip("janome")
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402


# ---------------------------------------------------------------- (A) 全角対応

def test_narrow_preserves_length_and_leaves_other_chars():
    for s in ("０４８－２２２－７８１５", "ＴＫ１２３４５６７", "㍿株式会社①", "田中太郎", ""):
        assert len(narrow(s)) == len(s)
    assert narrow("０４８－２２２－７８１５") == "048-222-7815"
    assert narrow("ＡＺａｚ") == "AZaz"
    assert narrow("１９９０年３月３日") == "1990年3月3日"
    assert narrow("＠：＋／．　") == "@:+/. "
    assert narrow("㍿①Ⅱ漢字かなカナ") == "㍿①Ⅱ漢字かなカナ"      # 長さの変わる/対象外の文字は触らない


def found(text, etype, **kw):
    return [text[e.start:e.end] for e in RegexDetector.default_ja(**kw).detect(text) if e.entity_type == etype]


@pytest.mark.parametrize("etype, surface, ctx, dates", [
    ("PHONE_NUMBER", "０４８－２２２－７８１５", "", False),
    ("POSTAL_CODE", "〒３４３－０８２４", "", False),
    ("CREDIT_CARD", "４１１１　１１１１　１１１１　１１１１", "", False),
    ("MY_NUMBER", "１２３４　５６７８　９０１８", "", False),
    ("CORPORATE_NUMBER", "７００００１２０５０００２", "法人番号 ", False),
    ("PASSPORT", "ＴＫ１２３４５６７", "旅券番号 ", False),
    ("RESIDENCE_CARD", "ＡＢ１２３４５６７８ＣＤ", "在留カード番号 ", False),
    ("BANK_ACCOUNT", "１１８７２２４", "振込先 普通 ", False),
    ("PENSION_NUMBER", "１２３４－５６７８９０", "基礎年金番号 ", False),
    ("IP_ADDRESS", "１９２．０．２．４４", "", False),
    ("EMAIL", "ｒ．ｍｉｗａ＠ｅｘａｍｐｌｅ．ｊｐ", "", False),
    ("URL", "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｊｐ", "", False),
    ("DATE", "２０２３年３月１１日", "生年月日 ", True),
    ("DATE", "平成２３年３月１１日", "", True),          # 判例文の標準表記
    ("DATE", "令和元年５月１日", "", True),
    ("DATE", "２０２３／０３／１１", "", True),
    ("AGE", "３４歳", "年齢 ", True),
])
def test_fullwidth_is_detected(etype, surface, ctx, dates):
    assert found(ctx + surface, etype, dates=dates) == [surface]


def test_halfwidth_still_works_and_offsets_point_at_original():
    text = "TEL 048-222-7815 / r.miwa@example.jp 〒343-0824 生年月日 1990年3月3日 満34歳"
    det = RegexDetector.default_ja(dates=True)
    got = {e.entity_type: text[e.start:e.end] for e in det.detect(text)}
    assert got["PHONE_NUMBER"] == "048-222-7815" and got["EMAIL"] == "r.miwa@example.jp"
    assert got["POSTAL_CODE"] == "〒343-0824" and got["DATE"] == "1990年3月3日" and got["AGE"] == "満34歳"
    # 全角の本文でも offset は原文（全角）を指す
    zen = "電話 ０４８－２２２－７８１５ です"
    (e,) = [x for x in det.detect(zen) if x.entity_type == "PHONE_NUMBER"]
    assert zen[e.start:e.end] == "０４８－２２２－７８１５"


def test_fullwidth_does_not_create_false_dates():
    # 全角でも検査は同じ: 存在しない月日は取らない
    assert found("１９９０年１３月３日", "DATE", dates=True) == []
    assert found("１９９０年２月３２日", "DATE", dates=True) == []
    assert found("法人番号 ７００００１２０５０００１", "CORPORATE_NUMBER") == []   # 検査用数字が合わない


# ---------------------------------------------------------------- (B) 肩書き判定の型スコープ

TRIM_CASES = [
    ("医療法人伊織会の理事長である河村敬史", "医療法人伊織会の", "ORG", "医療法人伊織会"),
    ("株式会社アオバの社長が来た", "株式会社アオバの", "ORG", "株式会社アオバ"),
    ("鮫島病院の院長に確認", "鮫島病院の", "FACILITY", "鮫島病院"),
    ("新宿駅の改札で", "新宿駅の", "FACILITY", "新宿駅"),
    ("東京都の人口は", "東京都の", "LOCATION", "東京都"),
    ("東京都新宿区の物件", "東京都新宿区の", "ADDRESS", "東京都新宿区"),
]
KEEP_CASES = [
    ("石川まりの\n出席", "石川まりの", "PERSON"),
    ("佐藤まりも氏が出席", "佐藤まりも", "PERSON"),
    ("金子かの、田中", "金子かの", "PERSON"),
    ("担当はあさひなが対応", "あさひなが", "PERSON"),
    ("田中は先生に相談した", "田中は", "PERSON"),   # PERSON は原理的に判別不能 → 安全側で維持
]


def _refined(refiner, text, span, etype):
    s = text.index(span)
    return [text[e.start:e.end] for e in refiner.refine(text, [Entity(etype, s, s + len(span), 0.9, "ner")])]


@pytest.mark.parametrize("text, span, etype, expected", TRIM_CASES)
@pytest.mark.parametrize("refiner_name", ["span", "morphological"])
def test_non_person_trims_particle_even_before_a_title(refiner_name, text, span, etype, expected):
    """肩書き（理事・社長・院長）が続いても、PERSON 以外なら末尾の助詞を削る。
    以前は _TITLES_AFTER_NAME が型に関係なく効いて『医療法人伊織会の』のまま残っていた。"""
    refiner = SpanRefiner() if refiner_name == "span" else MorphologicalRefiner()
    assert _refined(refiner, text, span, etype) == [expected]


@pytest.mark.parametrize("text, span, etype", KEEP_CASES)
@pytest.mark.parametrize("refiner_name", ["span", "morphological"])
def test_person_keeps_trailing_particle(refiner_name, text, span, etype):
    """PERSON は『名前の一部の助詞』と『格助詞』が品詞で区別できない（佐藤まりも氏 の も と
    田中は先生 の は はどちらも 助詞,係助詞）ため、安全側で維持する。"""
    refiner = SpanRefiner() if refiner_name == "span" else MorphologicalRefiner()
    assert _refined(refiner, text, span, etype) == [span]


def test_honorific_types_is_configurable():
    text = "医療法人伊織会の理事長"
    span = "医療法人伊織会の"
    # ORG も honorific_types に入れれば従来どおり残る（挙動を戻せることの確認）
    assert _refined(SpanRefiner(honorific_types=("PERSON", "ORG")), text, span, "ORG") == [span]
    assert _refined(SpanRefiner(), text, span, "ORG") == ["医療法人伊織会"]
    assert "理事" in _TITLES_AFTER_NAME    # リスト自体は PERSON 用として残す

"""データ分割・住所書式拡張ツール（tools/split_by_template.py, tools/augment_address_formats.py）。

どちらも学習データを作る側のツールなので、ライブラリ本体には依存しない。
Drive を読まずに済むよう、レコードは手で組む。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from augment_address_formats import (  # noqa: E402
    VARIANTS,
    address_format_stats,
    augment,
    rewrite,
    split_address,
)
from split_by_template import (  # noqa: E402
    current_split,
    grouped_split,
    template_signature,
)


def rec(text, *ents, source="synth_v4"):
    return {"text": text, "entities": [list(e) for e in ents], "source": source}


# ------------------------------------------------------------ template_signature

def test_signature_replaces_entities_with_type_names():
    text = "患者: 山田太郎 様（東京都港区1-2-3）"
    a = text.index("山田太郎")
    b = text.index("東京都港区1-2-3")
    r = rec(text, (a, a + 4, "PERSON"), (b, b + len("東京都港区1-2-3"), "ADDRESS"))
    assert template_signature(r) == "患者: <PERSON> 様（<ADDRESS>）"


def test_signature_is_same_for_same_template_different_entities():
    a = rec("患者: 山田太郎 様", (4, 8, "PERSON"))
    b = rec("患者: 佐藤 花子 様", (4, 9, "PERSON"))
    assert template_signature(a) == template_signature(b) == "患者: <PERSON> 様"


def test_signature_skips_overlapping_entities():
    # 重なるエンティティは先着を優先し、span が壊れないこと
    r = rec("山田太郎です", (0, 4, "PERSON"), (0, 2, "PERSON"))
    assert template_signature(r) == "<PERSON>です"


def test_loose_signature_collapses_digits_and_spaces():
    a = rec("住所は 1丁目 です")
    b = rec("住所は 27丁目 です")
    assert template_signature(a) != template_signature(b)
    assert template_signature(a, loose=True) == template_signature(b, loose=True)


def test_signature_without_entities_is_the_text():
    assert template_signature(rec("エンティティなし")) == "エンティティなし"


# ------------------------------------------------------------ 分割

def test_current_split_removes_exact_duplicate_texts_from_valid():
    records = [rec(f"文書{i}: 山田太郎", (5, 9, "PERSON")) for i in range(50)]
    train, valid, removed = current_split(records)
    assert len(train) + len(valid) + removed == 50
    texts = {r["text"] for r in train}
    assert not any(r["text"] in texts for r in valid)


def test_grouped_split_never_puts_a_template_on_both_sides():
    # 同じテンプレートから 4 件ずつ作った 40 グループ
    records = []
    for g in range(40):
        for k in range(4):
            records.append(rec(f"テンプレ{g}: 人物{k} 様", (len(f"テンプレ{g}: "),
                                                       len(f"テンプレ{g}: ") + 3, "PERSON")))
    train, valid = grouped_split(records)
    assert train and valid
    tsig = {template_signature(r) for r in train}
    assert [r for r in valid if template_signature(r) in tsig] == []


def test_grouped_split_groups_across_sources():
    """同じテンプレートが別ソースにあっても分かれない（v4 と v5c はテンプレートを共有する）。"""
    records = []
    for g in range(30):
        records.append(rec(f"型{g}: 田中 様", (len(f"型{g}: "), len(f"型{g}: ") + 2, "PERSON"),
                           source="synth_v4"))
        records.append(rec(f"型{g}: 佐藤 様", (len(f"型{g}: "), len(f"型{g}: ") + 2, "PERSON"),
                           source="synth_v5c"))
    train, valid = grouped_split(records)
    tsig = {template_signature(r) for r in train}
    assert [r for r in valid if template_signature(r) in tsig] == []


def test_grouped_split_keeps_source_proportions():
    records = []
    for g in range(100):
        records.append(rec(f"a{g}: 田中", (len(f"a{g}: "), len(f"a{g}: ") + 2, "PERSON"),
                           source="synth_v4"))
    for g in range(100):
        records.append(rec(f"b{g}: 佐藤", (len(f"b{g}: "), len(f"b{g}: ") + 2, "PERSON"),
                           source="out_ner"))
    _, valid = grouped_split(records)
    got = {}
    for r in valid:
        got[r["source"]] = got.get(r["source"], 0) + 1
    # 各ソース 100 件の 10% を上限に入れる
    assert got.get("synth_v4", 0) <= 10
    assert got.get("out_ner", 0) <= 10


# ------------------------------------------------------------ 住所の分解

@pytest.mark.parametrize("addr, pref, city, town, banchi, building", [
    ("東京都文京区本郷2丁目40番14号山崎ビル401号室",
     "東京都", "文京区", "本郷", "2丁目40番14号", "山崎ビル401号室"),
    ("京都府京都市伏見区道阿弥町1-15-1", "京都府", "京都市伏見区", "道阿弥町", "1-15-1", ""),
    ("香川県綾歌郡綾川町陶4番地", "香川県", "綾歌郡綾川町", "陶", "4番地", ""),
])
def test_split_address(addr, pref, city, town, banchi, building):
    p = split_address(addr)
    assert p is not None, addr
    assert (p["pref"], p["city"], p["town"], p["banchi"], p["building"]) == \
        (pref, city, town, banchi, building)


def test_split_address_reconstructs_the_original():
    addr = "静岡県島田市道悦4丁目23番16号 プライムマークタワー38F"
    p = split_address(addr)
    assert p["pref"] + p["city"] + p["town"] + p["banchi"] + p["building"] == addr


def test_split_address_allows_missing_town():
    # 町名が無い住所（番地が市区町村に直接続く）
    p = split_address("山口県萩市4-8-11")
    assert p is not None
    assert p["town"] == "" and p["banchi"] == "4-8-11"


def test_split_address_rejects_jo_sen_style():
    """○条 / ○線 型（北海道）は町名の中の数字を番地と誤認するので分解しない。"""
    assert split_address("北海道旭川市緑が丘南５条5-4-8 メゾン本町901号室") is None
    assert split_address("北海道上川郡鷹栖町１１線2丁目29番15号") is None


def test_split_address_rejects_without_prefecture_or_banchi():
    assert split_address("本郷2丁目40番14号") is None      # 都道府県が無い
    assert split_address("東京都文京区本郷") is None        # 番地が無い


# ------------------------------------------------------------ 書式のバリアント

def test_variants_produce_the_expected_formats():
    p = split_address("京都府京都市中京区壬生辻町5-3-8")
    got = {name: fn(p) for name, fn in VARIANTS.items()}
    assert got["都道府県なし"] == "京都市中京区壬生辻町5-3-8"
    assert got["町名+番地のみ"] == "壬生辻町5-3-8"
    assert got["町名 空白 番地"] == "壬生辻町 5-3-8"
    assert got["番地 空白 町名"] == "5-3-8 壬生辻町"
    assert got["番地+番地語 町名"] == "5-3-8番地 壬生辻町"
    assert got["番地+番地語 町名(空白なし)"] == "5-3-8番地壬生辻町"
    assert got["町名の番地"] == "壬生辻町の5-3-8"


def test_variant_does_not_duplicate_the_banchi_suffix():
    p = split_address("香川県綾歌郡綾川町陶4番地")
    assert VARIANTS["番地+番地語 町名"](p) == "4番地 陶"


# ------------------------------------------------------------ 書き換えと span

def test_rewrite_shifts_following_spans():
    text = "住所は東京都港区1-2-3、担当は田中です"
    a = text.index("東京都港区1-2-3")
    b = text.index("田中")
    r = rec(text, (a, a + len("東京都港区1-2-3"), "ADDRESS"), (b, b + 2, "PERSON"))
    out = rewrite(r, 0, "1-2-3 港区")
    assert out is not None
    got = [(out["text"][s:e], l) for s, e, l in out["entities"]]
    assert got == [("1-2-3 港区", "ADDRESS"), ("田中", "PERSON")]
    assert out["text"] == "住所は1-2-3 港区、担当は田中です"


def test_rewrite_keeps_preceding_spans():
    text = "田中さんの住所は東京都港区1-2-3です"
    b = text.index("東京都港区1-2-3")
    r = rec(text, (0, 2, "PERSON"), (b, b + len("東京都港区1-2-3"), "ADDRESS"))
    out = rewrite(r, 1, "港区1-2-3")
    assert [(out["text"][s:e], l) for s, e, l in out["entities"]] == \
        [("田中", "PERSON"), ("港区1-2-3", "ADDRESS")]


def test_rewrite_refuses_overlapping_entities():
    text = "東京都港区1-2-3"
    r = rec(text, (0, len(text), "ADDRESS"), (0, 3, "LOCATION"))
    assert rewrite(r, 0, "港区1-2-3") is None


# ------------------------------------------------------------ 全体

def test_augment_produces_missing_formats_and_valid_spans():
    text = "ご住所　東京都文京区本郷2丁目40番14号\n担当　田中"
    s = text.index("東京都")
    e = s + len("東京都文京区本郷2丁目40番14号")
    records = [rec(text, (s, e, "ADDRESS"), (text.index("田中"), text.index("田中") + 2, "PERSON"))]
    aug, made, _ = augment(records, per_doc=len(VARIANTS))
    assert len(aug) == len(VARIANTS)
    for r in aug:
        for st, en, lab in r["entities"]:
            assert 0 <= st < en <= len(r["text"])
        # PERSON が壊れていない
        assert any(r["text"][st:en] == "田中" for st, en, lab in r["entities"] if lab == "PERSON")
    stats = address_format_stats(aug)
    assert stats["数字始まり"] > 0
    assert stats["都道府県始まり"] == 0


def test_augment_skips_variants_that_would_start_with_space():
    # 町名が無い住所では並べ替え系の変種を作らない（先頭・末尾に空白が出る）
    text = "住所は山口県萩市4-8-11です"
    s = text.index("山口県")
    records = [rec(text, (s, s + len("山口県萩市4-8-11"), "ADDRESS"))]
    aug, _, skipped = augment(records, per_doc=len(VARIANTS))
    assert skipped["町名が無く並べ替えできない"] > 0
    for r in aug:
        for st, en, lab in r["entities"]:
            assert r["text"][st:en] == r["text"][st:en].strip()

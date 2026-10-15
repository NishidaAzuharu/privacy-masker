"""学習データ v6 の生成ツール（tools/build_data_v6.py）。Drive を読まずレコードは手で組む。"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from build_data_v6 import (  # noqa: E402
    GIVEN_HIRA,
    augment_persons,
    cap_templates,
    make_long_docs,
    person_variants,
    relabel_wiki_addresses,
    stats,
)
from split_by_template import template_signature  # noqa: E402


def rec(text, *ents, source="synth_v4"):
    return {"text": text, "entities": [list(e) for e in ents], "source": source}


def at(text, surface, label):
    s = text.index(surface)
    return s, s + len(surface), label


# ------------------------------------------------------------ wiki の付け替え

def test_relabel_street_level_wiki_locations_to_address():
    t1 = "本社は東京都文京区本郷二丁目40番14号山崎ビル401号室にある"
    t2 = "兵庫県宝塚市すみれガ丘1丁目は住宅地である"
    r1 = rec(t1, at(t1, "東京都文京区本郷二丁目40番14号山崎ビル401号室", "LOCATION"), source="out_ner")
    r2 = rec(t2, at(t2, "兵庫県宝塚市すみれガ丘1丁目", "LOCATION"), source="out_ner")
    n = relabel_wiki_addresses([r1, r2])
    assert n == 1
    assert r1["entities"][0][2] == "ADDRESS"     # 番地・号室あり → ADDRESS
    assert r2["entities"][0][2] == "LOCATION"    # 丁目だけ → LOCATION のまま


def test_relabel_only_touches_out_ner():
    t = "住所は東京都港区1-2-3です"
    r = rec(t, at(t, "東京都港区1-2-3", "LOCATION"), source="synth_v4")
    assert relabel_wiki_addresses([r]) == 0
    assert r["entities"][0][2] == "LOCATION"


# ------------------------------------------------------------ テンプレート上限

def test_cap_templates_limits_docs_per_loose_signature():
    records = []
    for k in range(30):
        text = f"担当: 山田{k:02d} 様（1丁目）"
        records.append(rec(text, at(text, f"山田{k:02d}", "PERSON")))
    kept, dropped = cap_templates(records, cap=4, rng=random.Random(0))
    assert len(kept) == 4
    assert dropped["synth_v4"] == 26
    # 数字を潰した loose 署名では全部同じテンプレート
    assert len({template_signature(r, loose=True) for r in records}) == 1


def test_cap_templates_keeps_singletons_and_drops_exact_duplicates():
    a = rec("文書A: 田中", (4, 6, "PERSON"))
    b = rec("文書B: 佐藤", (4, 6, "PERSON"))
    dup = rec("文書A: 田中", (4, 6, "PERSON"))
    kept, dropped = cap_templates([a, b, dup], cap=4, rng=random.Random(0))
    assert len(kept) == 2
    assert sum(dropped.values()) == 0          # 完全一致重複は「上限で落とした」には数えない


# ------------------------------------------------------------ 人名の書式

def test_person_variants_cover_the_thin_shapes():
    gens = person_variants(random.Random(1), ["Maria"], ["Ivers"])
    got = {k: fn() for k, fn in gens.items()}
    assert re.fullmatch(r"[一-鿿]+ [一-鿿]+", got["漢字姓 空白 漢字名"])
    assert got["かな名 漢字名 漢字姓(3要素)"].count(" ") == 2
    assert re.search(r"[ぁ-ゖ]", got["かな名 英字姓"]) and "Ivers" in got["かな名 英字姓"]
    assert got["英字名 漢字姓"].startswith("Maria ")
    assert "・" in got["カタカナ・カタカナ"]
    assert got["ひらがな単体"] in GIVEN_HIRA
    assert got["英字 空白 英字"] == "Maria Ivers"


def test_augment_persons_rewrites_one_person_and_keeps_other_spans():
    text = "担当は山田太郎、住所は東京都港区1-2-3です"
    r = rec(text, at(text, "山田太郎", "PERSON"), at(text, "東京都港区1-2-3", "ADDRESS"))
    out, made = augment_persons([r] * 40, ratio=1.0, rng=random.Random(2),
                                latin_given=["Maria"], latin_sur=["Ivers"])
    assert out and sum(made.values()) == len(out)
    for d in out:
        labs = [l for _, _, l in d["entities"]]
        assert labs == ["PERSON", "ADDRESS"]
        p = next(d["text"][s:e] for s, e, l in d["entities"] if l == "PERSON")
        a = next(d["text"][s:e] for s, e, l in d["entities"] if l == "ADDRESS")
        assert a == "東京都港区1-2-3"                 # 後続 span はずれていない
        assert p != "山田太郎" and d["variant"].startswith("person:")


def test_augment_persons_respects_ratio():
    text = "担当は山田太郎です"
    r = rec(text, at(text, "山田太郎", "PERSON"))
    out, _ = augment_persons([r] * 200, ratio=0.0, rng=random.Random(3),
                             latin_given=[], latin_sur=[])
    assert out == []


# ------------------------------------------------------------ 長文

def test_make_long_docs_concatenates_distinct_templates_with_offsets():
    # 数字を潰した署名が全部違うように、文書ごとに文面そのものを変える
    records = []
    for k in range(60):
        text = f"件名{chr(0x4e00 + k)}: 山田太郎は{'あ' * 100}を担当{chr(0x4e00 + k)}する。"
        records.append(rec(text, at(text, "山田太郎", "PERSON")))
    assert len({template_signature(r, loose=True) for r in records}) == 60
    out = make_long_docs(records, target_ratio=0.2, rng=random.Random(4))
    assert out
    for d in out:
        assert len(d["text"]) > 400
        assert d["source"] == "concat" and d["variant"].startswith("concat:")
        for s, e, l in d["entities"]:
            assert d["text"][s:e] == "山田太郎"
        assert "\n\n" in d["text"]


def test_make_long_docs_refuses_same_template_parts():
    """同じテンプレートの文書を繋いでも長文の意味がないので作らない。"""
    records = [rec(f"文書{k}: 山田太郎は{'あ' * 100}。", (len(f"文書{k}: "), len(f"文書{k}: ") + 4, "PERSON"))
               for k in range(60)]
    assert len({template_signature(r, loose=True) for r in records}) == 1
    assert make_long_docs(records, target_ratio=0.2, rng=random.Random(4)) == []


def test_make_long_docs_returns_nothing_when_target_is_met():
    records = [rec("x" * 500 + "山田太郎", (500, 504, "PERSON")) for _ in range(10)]
    assert make_long_docs(records, target_ratio=0.1, rng=random.Random(5)) == []


# ------------------------------------------------------------ 統計

def test_stats_three_element_definition_ignores_letter_spacing():
    t1 = "山 田 太 郎 です"
    t2 = "かな 麻里 松本 です"
    r1 = rec(t1, at(t1, "山 田 太 郎", "PERSON"))
    r2 = rec(t2, at(t2, "かな 麻里 松本", "PERSON"))
    st = stats([r1, r2])
    assert st["person"]["空白含む"] == 2
    assert st["person"]["3要素以上"] == 1        # 字間空白は数えない


def test_stats_counts_negatives_and_distinct_negatives():
    st = stats([rec("同じ負例"), rec("同じ負例"), rec("別の負例"), rec("田中です", (0, 2, "PERSON"))])
    assert st["counts"]["negatives"] == 3
    assert st["counts"]["negatives_distinct"] == 2

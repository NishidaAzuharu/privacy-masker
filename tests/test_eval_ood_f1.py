"""OOD F1 とエラー分析（tools/eval_ood_f1.py）の照合ロジック。

照合手順は `train_ja_mask_model (1).ipynb` のエラー分析セルに合わせている:
厳密一致 → IoU 最大で境界ズレ／型混同 → 残りを FP/FN。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from eval_ood_f1 import analyse, short_tag  # noqa: E402

from privacy_masker.types import Entity  # noqa: E402

TEXT = "担当は山田太郎、住所は東京都港区1-2-3、勤務先は株式会社アオバです"


def doc(*ents, neutral=()):
    return {"text": TEXT, "entities": [list(e) for e in ents], "neutral": [list(n) for n in neutral]}


def at(surface, label):
    s = TEXT.index(surface)
    return s, s + len(surface), label


def pred(surface, label, *, offset=0, length=None):
    s = TEXT.index(surface) + offset
    e = s + (length if length is not None else len(surface) - offset)
    return Entity(label, s, e, 0.9, "fake")


def test_exact_match_is_tp():
    a = analyse([doc(at("山田太郎", "PERSON"))], [[pred("山田太郎", "PERSON")]])
    assert a.tp["PERSON"] == 1
    assert sum(a.fp.values()) == sum(a.fn.values()) == 0
    assert a.micro() == (1.0, 1.0, 1.0)


def test_same_label_different_span_is_boundary_error():
    # gold「山田太郎」に対して「山田」だけ → 境界ズレ。FP と FN の両方に数える
    a = analyse([doc(at("山田太郎", "PERSON"))], [[pred("山田太郎", "PERSON", length=2)]])
    assert a.tp["PERSON"] == 0
    assert a.boundary["PERSON"] == 1
    assert a.fp["PERSON"] == 1 and a.fn["PERSON"] == 1
    assert len(a.boundary_examples) == 1
    ex = a.boundary_examples[0]
    assert ex["pred"]["text"] == "山田" and ex["gold"]["text"] == "山田太郎"
    assert 0.0 < ex["iou"] < 1.0


def test_different_label_overlap_is_type_confusion():
    a = analyse([doc(at("東京都港区1-2-3", "ADDRESS"))],
                [[pred("東京都港区1-2-3", "LOCATION")]])
    assert a.type_conf[("LOCATION", "ADDRESS")] == 1
    assert a.fp["LOCATION"] == 1 and a.fn["ADDRESS"] == 1
    assert a.boundary == {}
    assert len(a.type_conf_examples) == 1


def test_unmatched_prediction_is_false_positive():
    a = analyse([doc(at("山田太郎", "PERSON"))],
                [[pred("山田太郎", "PERSON"), pred("株式会社アオバ", "ORG")]])
    assert a.tp["PERSON"] == 1 and a.fp["ORG"] == 1
    assert [e["pred"]["text"] for e in a.false_positives] == ["株式会社アオバ"]


def test_unmatched_gold_is_complete_miss():
    a = analyse([doc(at("山田太郎", "PERSON"), at("株式会社アオバ", "ORG"))],
                [[pred("山田太郎", "PERSON")]])
    assert a.fn["ORG"] == 1
    assert [e["gold"]["text"] for e in a.false_negatives] == ["株式会社アオバ"]
    # 重なる予測があった gold は false_negatives には入らない（境界ズレ側に入る）
    assert all(e["gold"]["text"] != "山田太郎" for e in a.false_negatives)


def test_prediction_inside_neutral_is_not_a_false_positive():
    """部分注釈セット（OpenPII）では gold 以外が neutral なので FP に数えない。"""
    s = TEXT.index("株式会社アオバ")
    a = analyse([doc(at("山田太郎", "PERSON"), neutral=[(s, s + 7)])],
                [[pred("山田太郎", "PERSON"), pred("株式会社アオバ", "ORG")]])
    assert a.fp == {} and a.neutral_skipped == 1
    assert a.false_positives == []


def test_neutral_is_ignored_when_use_neutral_is_false():
    s = TEXT.index("株式会社アオバ")
    a = analyse([doc(at("山田太郎", "PERSON"), neutral=[(s, s + 7)])],
                [[pred("株式会社アオバ", "ORG")]], use_neutral=False)
    assert a.fp["ORG"] == 1 and a.neutral_skipped == 0


def test_partially_overlapping_prediction_is_still_counted():
    """neutral に完全に収まらない（gold と重なる）予測は FP に残る。"""
    gs, ge, _ = at("山田太郎", "PERSON")
    a = analyse([doc(at("山田太郎", "PERSON"), neutral=[(ge, len(TEXT))])],
                [[pred("山田太郎", "PERSON", offset=2, length=4)]])
    assert a.neutral_skipped == 0
    assert a.boundary["PERSON"] == 1


def test_labels_outside_the_target_set_are_ignored():
    a = analyse([doc(at("山田太郎", "PERSON"))],
                [[pred("山田太郎", "PERSON"), pred("1-2-3", "PHONE_NUMBER")]])
    assert sum(a.fp.values()) == 0


def test_label_agnostic_counts_span_only():
    # 型は違うが span は一致 → ラベル無視では TP
    a = analyse([doc(at("東京都港区1-2-3", "ADDRESS"))],
                [[pred("東京都港区1-2-3", "LOCATION")]])
    assert (a.bin_tp, a.bin_fp, a.bin_fn) == (1, 0, 0)
    assert a.binary() == (1.0, 1.0, 1.0)
    # 型が合っても span が違えば FP と FN
    b = analyse([doc(at("山田太郎", "PERSON"))], [[pred("山田太郎", "PERSON", length=2)]])
    assert (b.bin_tp, b.bin_fp, b.bin_fn) == (0, 1, 1)


def test_one_gold_matches_only_one_prediction():
    """同じ gold に 2 つの予測が重なっても TP は 1 件。余りは FP。"""
    a = analyse([doc(at("山田太郎", "PERSON"))],
                [[pred("山田太郎", "PERSON"), pred("山田太郎", "PERSON", length=2)]])
    assert a.tp["PERSON"] == 1
    assert a.fp["PERSON"] == 1
    assert a.fn["PERSON"] == 0        # gold は厳密一致で消費済み
    assert a.boundary["PERSON"] == 0  # 相手の gold が残っていないので境界ズレにはならない


def test_macro_and_micro_differ_with_imbalanced_labels():
    docs = [doc(at("山田太郎", "PERSON"), at("株式会社アオバ", "ORG"))]
    preds = [[pred("山田太郎", "PERSON"), pred("株式会社アオバ", "ORG", length=4)]]
    a = analyse(docs, preds)
    assert a.tp["PERSON"] == 1 and a.boundary["ORG"] == 1
    p, r, f1 = a.micro()
    assert p == r == 0.5 and abs(f1 - 0.5) < 1e-9
    assert abs(a.macro_f1() - 0.5) < 1e-9   # PERSON 1.0 と ORG 0.0 の平均


def test_short_tag_shortens_model_paths(tmp_path):
    d = tmp_path / "outputs_ruri-v3-30m-charhead-v45e" / "best"
    d.mkdir(parents=True)
    assert short_tag(str(d)) == "ruri-v3-30m-charhead-v45e"
    assert short_tag("30m(v45e)") == "30m(v45e)"


def test_uncovered_counts_zero_coverage_only():
    """1 つの予測が隣接する 2 つの gold にまたがると、片方は 1 対 1 照合で未マッチになるが、
    文字は隠れているので被覆ゼロには数えない。v6 モデルの評価で実際に起きた取り違え。"""
    text = "拠点は京都市伏見区醍醐南里町の坂本町6-18-8です"
    loc = text.index("京都市伏見区醍醐南里町")
    addr = text.index("坂本町6-18-8")
    gold = [[loc, loc + len("京都市伏見区醍醐南里町"), "LOCATION"],
            [addr, addr + len("坂本町6-18-8"), "ADDRESS"]]
    doc_ = {"text": text, "entities": gold, "neutral": []}
    wide = [Entity("ADDRESS", loc, addr + len("坂本町6-18-8"), 0.95, "fake")]
    a = analyse([doc_], [wide])
    assert len(a.false_negatives) == 1          # 相手を取られた LOCATION
    assert sum(a.uncovered.values()) == 0       # だが文字は隠れている
    assert a.uncovered_examples == []
    b = analyse([doc_], [[]])
    assert sum(b.uncovered.values()) == 2 and len(b.false_negatives) == 2
    assert b.uncovered["LOCATION"] == 1 and b.uncovered["ADDRESS"] == 1


def test_uncovered_ignores_label_mismatch():
    # 型が違っても文字が隠れていれば被覆ゼロではない
    a = analyse([doc(at("山田太郎", "PERSON"))], [[pred("山田太郎", "ORG")]])
    assert sum(a.uncovered.values()) == 0
    assert a.type_conf[("ORG", "PERSON")] == 1

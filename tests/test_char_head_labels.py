"""R6a / R6b: char_labels_from_entities の平坦化のテスト。

R6a: 未知型が既知型の正解を消していた（UNKNOWN[0,4) + PERSON[1,3) が全 O になる）。
R6b: 「文字ごとに長いスパンを優先」だったため、ORG[0,6) + PERSON[4,9) から元データに
     ない PERSON[6,9) を作って学習させていた。
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from privacy_masker.char_head import (  # noqa: E402
    ENTITY_TYPES,
    ID2LABEL,
    LABEL2ID,
    CharSpan,
    char_labels_from_entities,
    decode_char_labels,
    encode_chunk,
    flatten_entities,
)

CASE = "医療法人伊織会の理事長である河村敬史"


def _tiny_tokenizer(model_max_length: int = 512):
    """``会の`` を 1 トークンにまとめる小さな BPE tokenizer（既存テストと同じ構成）。"""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    chars = sorted(set(CASE + "あいうえおアイウエオ0123456789abcXYZ、。 "))
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    vocab["会の"] = len(vocab)
    raw = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocab, merges=[("会", "の")], unk_token="[UNK]")
    )
    raw.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
        model_max_length=model_max_length,
    )


def _names(labels):
    return [ID2LABEL[i] for i in labels]


def _spans_from_labels(text, labels):
    """BIO ラベル列からスパンを復元する（decode_char_labels を gold 用に使う）。"""
    type_ids, starts = [], []
    for i in labels:
        name = ID2LABEL[i]
        if name == "O":
            type_ids.append(0)
            starts.append(False)
        else:
            type_ids.append(ENTITY_TYPES.index(name[2:]) + 1)
            starts.append(name.startswith("B-"))
    return [
        (s.start, s.end, s.label)
        for s in decode_char_labels(text, type_ids, [1.0] * len(text), starts)
    ]


# ---------------------------------------------------------------------------
# R6a: 未知型
# ---------------------------------------------------------------------------

def test_unknown_type_does_not_suppress_known_nested_type():
    stats = {}
    labels = char_labels_from_entities(
        "あいうえ",
        [{"start": 0, "end": 4, "label": "UNKNOWN"}, {"start": 1, "end": 3, "label": "PERSON"}],
        stats=stats,
    )
    assert _names(labels) == ["O", "B-PERSON", "I-PERSON", "O"]
    assert stats["dropped_unknown_type"] == 1
    assert stats["unknown_type_names"] == ["UNKNOWN"]


def test_unknown_type_can_be_rejected_explicitly():
    with pytest.raises(ValueError, match="未知のエンティティ型"):
        char_labels_from_entities(
            "あいうえ", [{"start": 0, "end": 4, "label": "UNKNOWN"}], unknown_types="error"
        )


def test_unknown_only_input_is_still_all_o_by_default():
    # 既存の契約（未知型だけなら O）は既定では変えない
    assert _names(char_labels_from_entities("abc", [{"start": 0, "end": 3, "label": "EMAIL"}])) == [
        "O", "O", "O",
    ]


# ---------------------------------------------------------------------------
# R6b: 重複
# ---------------------------------------------------------------------------

def test_crossing_spans_do_not_create_an_unannotated_partial_entity():
    text = "あいうえおかきくけ"
    ents = [{"start": 0, "end": 6, "label": "ORG"}, {"start": 4, "end": 9, "label": "PERSON"}]
    stats = {}
    labels = char_labels_from_entities(text, ents, stats=stats)

    # 和集合を 1 スパンに統合する。旧実装が作っていた PERSON[6,9) は元データに
    # 存在しないので作らない。型は「長い方」ではなく _KEEP_RISK_ORDER 順で選ぶので
    # PERSON になる（ORG にすると ORG_PUBLIC 判定で Keep になり人名が素通しになる）。
    assert _names(labels) == ["B-PERSON"] + ["I-PERSON"] * 8
    assert stats["merged_overlap_groups"] == 1
    assert stats["merged_overlap_spans"] == 1

    # 統合後も 4〜8 文字目は必ずマスク対象（負けたスパンを丸ごと落として O にはしない）
    assert all(name != "O" for name in _names(labels)[4:9])


def test_flatten_entities_matches_the_labels_it_produces():
    # 学習ラベルと評価 gold の平坦化を同じ関数に寄せる。両者が厳密に一致すること。
    text = "あいうえおかきくけこさしすせそ"
    ents = [
        {"start": 0, "end": 6, "label": "ORG"},
        {"start": 4, "end": 9, "label": "PERSON"},
        {"start": 11, "end": 14, "label": "LOCATION"},
    ]
    flat = [(s.start, s.end, s.label) for s in flatten_entities(text, ents)]
    labels = char_labels_from_entities(text, ents)
    # 型は Keep 危険度順（PERSON < ORG）で選ぶ
    assert flat == [(0, 9, "PERSON"), (11, 14, "LOCATION")]
    assert _spans_from_labels(text, labels) == flat


def test_overlap_can_be_rejected_explicitly():
    with pytest.raises(ValueError, match="重複するエンティティ"):
        char_labels_from_entities(
            "あいうえおかきくけ",
            [{"start": 0, "end": 6, "label": "ORG"}, {"start": 4, "end": 9, "label": "PERSON"}],
            overlap="error",
        )


def test_nested_span_keeps_the_previous_result():
    # 入れ子は旧実装（文字ごとに長い方を優先）と同じ結果になる
    text = "東京都新宿区西新宿"
    labels = char_labels_from_entities(
        text,
        [{"start": 0, "end": 9, "label": "ADDRESS"}, {"start": 0, "end": 3, "label": "LOCATION"}],
    )
    assert _names(labels) == ["B-ADDRESS"] + ["I-ADDRESS"] * 8


def test_chained_overlaps_become_one_span():
    # 連鎖する重複は 1 つの連結成分にまとめる（部分スパンを作らないため）
    text = "0123456789"
    flat = flatten_entities(
        text,
        [
            {"start": 0, "end": 4, "label": "PERSON"},
            {"start": 3, "end": 6, "label": "ORG"},
            {"start": 5, "end": 10, "label": "LOCATION"},
        ],
    )
    # 連鎖して 1 成分になり、型は成分内で最も Keep になりにくい PERSON が勝つ
    assert [(s.start, s.end, s.label) for s in flat] == [(0, 10, "PERSON")]


def test_touching_spans_are_not_merged():
    # 接しているだけ（end == start）は重複ではない。B- で分かれることを保つ
    text = "山田太郎佐藤花子"
    labels = char_labels_from_entities(
        text,
        [{"start": 0, "end": 4, "label": "PERSON"}, {"start": 4, "end": 8, "label": "PERSON"}],
    )
    assert _names(labels) == ["B-PERSON", "I-PERSON", "I-PERSON", "I-PERSON"] * 2
    # gold も同じ平坦化を通せばラベルからの復元と一致する（2 人が別スパンのまま）
    assert _spans_from_labels(text, labels) == [
        (s.start, s.end, s.label)
        for s in flatten_entities(
            text,
            [{"start": 0, "end": 4, "label": "PERSON"}, {"start": 4, "end": 8, "label": "PERSON"}],
        )
    ]


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        char_labels_from_entities("abc", [], overlap="longest")
    with pytest.raises(ValueError):
        char_labels_from_entities("abc", [], unknown_types="ignore")


# ---------------------------------------------------------------------------
# encode_chunk への伝播（学習データ作成の呼び出し口）
# ---------------------------------------------------------------------------

def test_encode_chunk_reports_dropped_and_merged_counts():
    tok = _tiny_tokenizer()
    ents = [
        {"start": 0, "end": 7, "label": "ORG"},
        {"start": 5, "end": 9, "label": "PERSON"},
        {"start": 14, "end": 18, "label": "MAIL"},
    ]
    f = encode_chunk(CASE, tok, ents)
    assert f["label_stats"]["merged_overlap_groups"] == 1
    assert f["label_stats"]["dropped_unknown_type"] == 1
    # 統合されたのは ORG[0,7) と PERSON[5,9) の和集合 [0,9)。型は PERSON（Keep 危険度順）
    assert _names(f["labels"])[:9] == ["B-PERSON"] + ["I-PERSON"] * 8


def test_encode_chunk_strict_labels_fails_closed():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError):
        encode_chunk(CASE, tok, [{"start": 14, "end": 18, "label": "MAIL"}], strict_labels=True)
    with pytest.raises(ValueError):
        encode_chunk(
            CASE,
            tok,
            [{"start": 0, "end": 7, "label": "ORG"}, {"start": 5, "end": 9, "label": "PERSON"}],
            strict_labels=True,
        )


def test_encode_chunk_without_entities_has_no_label_stats():
    tok = _tiny_tokenizer()
    f = encode_chunk(CASE, tok)
    assert f["labels"] is None
    assert f["label_stats"] is None


# ---------------------------------------------------------------------------
# 回帰: 重複なし・全型既知なら旧実装と完全に同じラベル列
# ---------------------------------------------------------------------------

def _legacy_char_labels(text, spans):
    """旧実装（文字ごとに長いスパンを優先）。比較用にここへ写した。"""
    n = len(text)
    spans = [s for s in spans if 0 <= s.start < s.end <= n]
    owner = [-1] * n
    order = sorted(range(len(spans)), key=lambda i: (-(spans[i].end - spans[i].start), spans[i].start))
    for idx in order:
        for i in range(spans[idx].start, spans[idx].end):
            if owner[i] == -1:
                owner[i] = idx
    labels = [LABEL2ID["O"]] * n
    for i in range(n):
        oi = owner[i]
        if oi == -1:
            continue
        key = ("B-" if (i == 0 or owner[i - 1] != oi) else "I-") + spans[oi].label
        if key in LABEL2ID:
            labels[i] = LABEL2ID[key]
    return labels


def test_non_overlapping_known_types_are_bit_identical_to_the_old_behavior():
    rnd = random.Random(20260908)
    for _ in range(500):
        n = rnd.randint(1, 40)
        text = "".join(rnd.choice("あいうえお漢字abc 1") for _ in range(n))
        spans, pos = [], 0
        while pos < n:
            gap = rnd.randint(0, 3)
            start = pos + gap
            length = rnd.randint(1, 5)
            if start + length > n:
                break
            spans.append(CharSpan(start, start + length, rnd.choice(ENTITY_TYPES)))
            pos = start + length
        assert char_labels_from_entities(text, spans) == _legacy_char_labels(text, spans)

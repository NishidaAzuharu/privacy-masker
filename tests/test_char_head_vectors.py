"""スパンベクトル: encoder の隠れ状態をスパン上で平均して Entity.vector に付ける（設計 §2）。人工モデルで検査する。"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import privacy_masker.char_head as ch  # noqa: E402
from privacy_masker.detectors.char_head import CharHeadNerDetector  # noqa: E402
from tests.test_char_head_predict import ALPHA, RuleModel, TinyTok, _doc, _place  # noqa: E402

DIM = 8


class HiddenRuleModel(RuleModel):
    """RuleModel に、input_ids から決まる 8 次元の隠れ状態を足したもの（encoder の last_hidden_state の代わり）。"""

    def __call__(self, **b):
        out = super().__call__(**b)
        ids = b["input_ids"]
        onehot = torch.nn.functional.one_hot(ids % DIM, DIM).float()
        out["token_hidden"] = onehot * (1.0 + (ids % 3).unsqueeze(-1).float())
        return out


def _setup(seed=1, n=60):
    tok = TinyTok(ALPHA)
    doc = _doc(n, seed)
    doc = _place(doc, 5, "山田太郎")
    doc = _place(doc, 30, "東京都")
    gold = [(5, 9, "PERSON"), (30, 33, "LOCATION")]
    return tok, doc, gold


def test_predict_char_spans_returns_one_vector_per_span_unit_norm_and_deterministic():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    vecs: list[list[bytes | None]] = []
    spans = ch.predict_char_spans([doc], model, tok, vectors_out=vecs)
    assert len(vecs) == 1 and len(vecs[0]) == len(spans[0]) == 2
    for v in vecs[0]:
        assert isinstance(v, bytes) and len(v) == DIM * 2
        floats = ch.vector_to_floats(v)
        assert len(floats) == DIM
        assert abs(sum(x * x for x in floats) ** 0.5 - 1.0) < 1e-2      # L2 正規化（float16 の丸め込み）
    vecs2: list[list[bytes | None]] = []
    ch.predict_char_spans([doc], model, tok, vectors_out=vecs2)
    assert vecs2 == vecs                                                 # 決定的


def test_model_without_hidden_states_gives_none_vectors():
    tok, doc, gold = _setup()
    vecs: list[list[bytes | None]] = []
    spans = ch.predict_char_spans([doc], RuleModel(doc, tok, gold), tok, vectors_out=vecs)
    assert len(vecs[0]) == len(spans[0]) and all(v is None for v in vecs[0])


def test_vector_uses_chunk_that_sees_the_whole_span():
    tok, doc, gold = _setup(seed=3, n=90)
    doc = _place(doc, 70, "佐藤花子")
    gold = gold + [(70, 74, "PERSON")]
    model = HiddenRuleModel(doc, tok, gold)
    vecs: list[list[bytes | None]] = []
    spans = ch.predict_char_spans([doc], model, tok, chunk_chars=40, chunk_overlap=10, vectors_out=vecs)
    # 3 実体がどれかのチャンクに丸ごと入る位置にしてある（token_safe_chunks の境界が変わって実体が割れたら位置を動かす）
    assert [(s.start, s.end) for s in spans[0]] == [(5, 9), (30, 33), (70, 74)]
    assert all(v is not None for v in vecs[0])


def test_span_vector_returns_none_when_no_chunk_overlaps_the_span():
    """スパンがどのチャンクとも重ならない（防御分岐）場合は None（span_vector 冒頭、best is None）。"""
    sp = ch.CharSpan(100, 105, "PERSON")
    chunks = [(0, "abcdefg")]                      # 0-7 文字目しか無く、100-105 とは重ならない
    feat_idx = [0]
    feats = [{"char_to_token": {}, "char_mask": []}]
    local_hidden = [None]
    assert ch.span_vector(sp, chunks, feat_idx, feats, local_hidden) is None


def test_span_vector_returns_none_when_local_hidden_is_none():
    """重なるチャンクは見つかるが、そのチャンクの隠れ状態が無い（モデルが hidden_states を
    返さない等）場合は None（span_vector 中盤、hidden is None）。"""
    sp = ch.CharSpan(1, 3, "PERSON")
    chunks = [(0, "abcdefg")]                      # スパンはチャンク内に完全に収まる
    feat_idx = [0]
    feats = [{"char_to_token": {i: i for i in range(7)}, "char_mask": [True] * 7}]
    local_hidden = [None]                          # このチャンクの隠れ状態が無い
    assert ch.span_vector(sp, chunks, feat_idx, feats, local_hidden) is None


def test_detector_attaches_vectors_and_can_disable():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    det = CharHeadNerDetector(model=model, tokenizer=tok)
    ents = det.detect(doc)
    assert [e.entity_type for e in ents] == ["PERSON", "LOCATION"]
    assert all(isinstance(e.vector, bytes) and len(e.vector) == DIM * 2 for e in ents)
    det_off = CharHeadNerDetector(model=model, tokenizer=tok, span_vectors=False)
    assert all(e.vector is None for e in det_off.detect(doc))


def test_vector_bytes_are_little_endian_regardless_of_host():
    import numpy as np

    from privacy_masker.char_head import vector_to_floats

    data = np.array([1.0, -0.5], dtype="<f2").tobytes()
    assert vector_to_floats(data) == (1.0, -0.5)

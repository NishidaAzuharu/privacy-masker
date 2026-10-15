"""encode_span_vectors は predict_char_spans(vectors_out=) と同じベクトルを、与えたスパンに対して返す（計画 B タスク1）。"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from privacy_masker import char_head as ch  # noqa: E402
from tests.test_char_head_predict import RuleModel  # noqa: E402
from tests.test_char_head_vectors import HiddenRuleModel, _setup  # noqa: E402


def test_matches_predict_char_spans_for_the_same_spans():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    vecs_pred: list[list[bytes | None]] = []
    spans = ch.predict_char_spans([doc], model, tok, vectors_out=vecs_pred)
    given = [[(sp.start, sp.end) for sp in doc_spans] for doc_spans in spans]
    vecs = ch.encode_span_vectors([doc], given, model, tok)
    assert vecs == vecs_pred and all(v is not None for d in vecs for v in d)


def test_arbitrary_spans_and_empty_docs():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    # (0, 4) と (30, 33) は gold のスパンとは無関係に選んだ任意の区間（後者はたまたま東京都と重なる）。
    vecs = ch.encode_span_vectors([doc, ""], [[(0, 4), (30, 33)], []], model, tok)
    assert len(vecs) == 2 and len(vecs[0]) == 2 and vecs[1] == []
    a = ch.vector_to_floats(vecs[0][0])
    assert abs(sum(x * x for x in a) - 1.0) < 1e-2      # L2 正規化（float16 の丸め込み）
    assert vecs[0][0] != vecs[0][1]


def test_model_without_hidden_gives_none():
    tok, doc, gold = _setup()
    assert ch.encode_span_vectors([doc], [[(5, 9)]], RuleModel(doc, tok, gold), tok) == [[None]]


def test_length_mismatch_raises_value_error():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    with pytest.raises(ValueError, match="texts and spans_per_doc"):
        ch.encode_span_vectors([doc, doc], [[(0, 4)]], model, tok)

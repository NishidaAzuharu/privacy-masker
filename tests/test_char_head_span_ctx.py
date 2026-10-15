"""span_ctx 特徴量: [スパン平均; 左 16 サブワード平均; 右 16 サブワード平均]（設計 v2 §2.1）。人工モデルで検査する。"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import privacy_masker.char_head as ch  # noqa: E402
from privacy_masker.detectors.char_head import CharHeadNerDetector  # noqa: E402
from tests.test_char_head_predict import ALPHA, TinyTok, _doc  # noqa: E402
from tests.test_char_head_vectors import DIM, HiddenRuleModel, _setup  # noqa: E402


def _blocks(v: bytes):
    f = ch.vector_to_floats(v)
    assert len(f) == 3 * DIM
    return f[:DIM], f[DIM:2 * DIM], f[2 * DIM:]


def _norm(x):
    return sum(a * a for a in x) ** 0.5


def test_span_ctx_has_three_blocks_and_first_block_is_span_mean():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    mean = ch.encode_span_vectors([doc], [[(5, 9)]], model, tok)[0][0]
    ctx = ch.encode_span_vectors([doc], [[(5, 9)]], model, tok, features="span_ctx")[0][0]
    assert len(mean) == DIM * 2 and len(ctx) == 3 * DIM * 2
    a, left, right = _blocks(ctx)
    assert a == ch.vector_to_floats(mean)                                    # 1 区画目は span_mean と同一のバイト
    assert abs(_norm(left) - 1.0) < 1e-2 and abs(_norm(right) - 1.0) < 1e-2   # 文脈がある側は単位ベクトル
    assert ch.FEATURES == ("span_mean", "span_ctx") and ch.CTX_TOKENS == 16


def test_span_ctx_left_block_is_zero_at_document_start_and_right_block_at_end():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    first = ch.encode_span_vectors([doc], [[(0, 3)]], model, tok, features="span_ctx")[0][0]
    _, left, right = _blocks(first)
    assert all(x == 0.0 for x in left) and abs(_norm(right) - 1.0) < 1e-2
    last = ch.encode_span_vectors([doc], [[(len(doc) - 3, len(doc))]], model, tok, features="span_ctx")[0][0]
    _, left2, right2 = _blocks(last)
    assert abs(_norm(left2) - 1.0) < 1e-2 and all(x == 0.0 for x in right2)


def test_predict_char_spans_vector_features_matches_encode_span_vectors():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    out: list[list[bytes | None]] = []
    spans = ch.predict_char_spans([doc], model, tok, vectors_out=out, vector_features="span_ctx")
    given = [[(sp.start, sp.end) for sp in d] for d in spans]
    assert out == ch.encode_span_vectors([doc], given, model, tok, features="span_ctx")
    assert all(len(v) == 3 * DIM * 2 for v in out[0])


def test_unknown_features_is_value_error():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    with pytest.raises(ValueError, match="features"):
        ch.encode_span_vectors([doc], [[(5, 9)]], model, tok, features="span_max")


def test_detector_follows_the_head_features_and_defaults_to_span_mean():
    tok, doc, gold = _setup()
    det = CharHeadNerDetector(model=HiddenRuleModel(doc, tok, gold), tokenizer=tok, category_head=False)
    assert det.vector_features == "span_mean"
    ents = det.detect(doc)
    assert ents and all(len(e.vector) == DIM * 2 for e in ents)
    det.category_head = types.SimpleNamespace(features="span_ctx")            # 読み込んだヘッドの features に従う
    assert det.vector_features == "span_ctx"
    ents2 = det.detect(doc)
    assert [(e.start, e.end) for e in ents2] == [(e.start, e.end) for e in ents]
    assert all(len(e.vector) == 3 * DIM * 2 for e in ents2)


def test_span_ctx_batch_window_is_sixteen_subwords_hand_computed():
    tok = TinyTok(ALPHA)
    text = "".join(ALPHA[(i * 7) % len(ALPHA)] for i in range(60))
    H = 64
    f = ch.encode_chunk(text, tok, None, piece_cache={}, strict=True)
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    T = batch["input_ids"].shape[1]
    hidden = torch.zeros(1, T, H)
    for t in range(T):
        hidden[0, t, t % H] = float(t + 1)
    cat_spans = torch.tensor([[[30, 33, 0, 0]]], dtype=torch.long)
    cat_mask = torch.ones(1, 1, dtype=torch.bool)
    out, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"], cat_spans, cat_mask)
    span_toks, left_toks, right_toks = [31, 32, 33], list(range(15, 31)), list(range(34, 50))
    def expected(toks):
        v = torch.zeros(H)
        for t in toks:
            v[t % H] += float(t + 1) / len(toks)
        return v / v.norm()
    assert out.shape == (1, 3 * H) and bool(valid[0])
    assert torch.allclose(out[0, :H], expected(span_toks), atol=1e-6)
    assert torch.allclose(out[0, H:2 * H], expected(left_toks), atol=1e-6)
    assert torch.allclose(out[0, 2 * H:], expected(right_toks), atol=1e-6)
    assert len(left_toks) == 16 and len(right_toks) == 16


def _f16_bytes(t):
    return t.detach().cpu().numpy().astype("<f2").tobytes()


def _hidden_and_batch(model, tok, texts, cat_spans_per_chunk, chunk_chars=ch.CHUNK_CHARS, chunk_overlap=ch.CHUNK_OVERLAP):
    feats, chunks, _doc_feats, _ = ch._chunk_and_encode(texts, tok, chunk_chars, chunk_overlap, ch.MAX_LENGTH)
    for f, cs in zip(feats, cat_spans_per_chunk):
        f["cat_spans"] = cs
    batch = ch.CharNerCollator(tok.pad_token_id)(feats)
    with torch.no_grad():
        hidden = model(**batch)["token_hidden"]
    return hidden.to(torch.float16).float(), batch, chunks


def test_span_ctx_batch_matches_span_vector_bytes():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    spans = [(0, 3), (5, 9), (12, 20), (30, 33), (len(doc) - 3, len(doc))]
    ref = ch.encode_span_vectors([doc], [spans], model, tok, features="span_ctx")[0]
    hidden, batch, chunks = _hidden_and_batch(model, tok, [doc], [[(s, e, 0, 0) for s, e in spans]])
    assert len(chunks[0]) == 1
    out, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"], batch["cat_spans"], batch["cat_mask"])
    assert out.shape == (len(spans), 3 * DIM) and bool(valid.all())
    for i, r in enumerate(ref):
        assert _f16_bytes(out[i]) == r


def test_span_ctx_batch_multi_chunk_matches_span_vector():
    tok, _d, _g = _setup()
    long_doc = _doc(130, 7)                                            # 人工モデルはチャンク本文が文書内で一意であることを要する
    model = HiddenRuleModel(long_doc, tok, [])
    spans = [(2, 6), (45, 52), (95, 100), (125, 130)]
    ref = ch.encode_span_vectors([long_doc], [spans], model, tok, features="span_ctx", chunk_chars=60, chunk_overlap=10)[0]
    feats, chunks, _df, _ = ch._chunk_and_encode([long_doc], tok, 60, 10, ch.MAX_LENGTH)
    assert len(chunks[0]) >= 2
    per_chunk = [[(s - off, e - off, 0, 0) for s, e in spans if s >= off and e <= off + len(piece)] for off, piece in chunks[0]]
    hidden, batch, _ = _hidden_and_batch(model, tok, [long_doc], per_chunk, chunk_chars=60, chunk_overlap=10)
    out, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"], batch["cat_spans"], batch["cat_mask"])
    got, i = {}, 0
    for (off, _piece), cs in zip(chunks[0], per_chunk):
        for s, e, _k, _c in cs:
            got.setdefault((s + off, e + off), _f16_bytes(out[i])); i += 1    # 最初に完全に含むチャンクの行が span_vector と一致する
    for (s, e), r in zip(spans, ref):
        assert got[(s, e)] == r


def test_span_ctx_batch_zero_row_for_span_without_subwords():
    tok, doc, gold = _setup()
    f = ch.encode_chunk(doc, tok, None, piece_cache={}, strict=True, category_spans=[(5, 9, 0, 0)])
    f["char_mask"] = [0] * len(doc)
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    hidden = torch.ones(1, batch["input_ids"].shape[1], DIM)
    out, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"], batch["cat_spans"], batch["cat_mask"])
    assert out.shape == (1, 3 * DIM) and not bool(valid[0]) and float(out.abs().sum()) == 0.0


def test_span_ctx_batch_gradient_matches_finite_difference():
    """L2 正規化の分母にも勾配が流れる（分母を定数扱いすると射影項が抜けて数値微分と合わない）。"""
    tok = TinyTok(ALPHA)
    text = "".join(ALPHA[(i * 7) % len(ALPHA)] for i in range(40))
    H = 6
    f = ch.encode_chunk(text, tok, None, piece_cache={}, strict=True, category_spans=[(18, 22, 0, 0), (0, 3, 1, 0)])
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, batch["input_ids"].shape[1], H, generator=gen, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3 * H, generator=gen, dtype=torch.float64)

    def loss_fn(h):
        out, valid = ch.span_ctx_batch(h, batch["attention_mask"], batch["char_to_token"], batch["char_mask"],
                                       batch["cat_spans"], batch["cat_mask"])
        assert out.dtype == torch.float64 and bool(valid.all())
        return (out * weights).sum()

    (grad,) = torch.autograd.grad(loss_fn(hidden), hidden)
    assert float(grad.abs().sum()) > 0
    assert torch.autograd.gradcheck(loss_fn, (hidden,), eps=1e-6, atol=1e-7, rtol=1e-5)

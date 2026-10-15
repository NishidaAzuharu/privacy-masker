"""補助カテゴリ層（第 2 段）: カテゴリスパンの経路、損失、クラス重み、教師の結合。人工モデルで検査する。"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import privacy_masker.char_head as ch  # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES  # noqa: E402
from tests.test_char_head_predict import ALPHA, TinyTok  # noqa: E402
from tests.test_char_head_vectors import DIM, HiddenRuleModel, _setup  # noqa: E402


def test_category_vocabulary_matches_semantic():
    assert ch.CATEGORY_KINDS == ("org", "facility", "location")
    for kind in ch.CATEGORY_KINDS:
        assert ch.CATEGORY_NAMES[kind] == tuple(CATEGORIES[kind])
    assert ch.CATEGORY_KIND_INDEX == {"org": 0, "facility": 1, "location": 2}
    assert ch.CATEGORY_INDEX["org"]["finance"] == CATEGORIES["org"].index("finance")


def test_encode_chunk_carries_and_validates_category_spans():
    tok = TinyTok(ALPHA)
    text = ALPHA[:20]
    f = ch.encode_chunk(text, tok, None, piece_cache={}, strict=True)
    assert f["cat_spans"] == []                                        # 省略時は空（推論経路）
    f2 = ch.encode_chunk(text, tok, None, piece_cache={}, strict=True, category_spans=[(2, 5, 0, 3), (10, 12, 2, 1)])
    assert f2["cat_spans"] == [(2, 5, 0, 3), (10, 12, 2, 1)]
    for bad in ([(5, 5, 0, 0)], [(-1, 3, 0, 0)], [(0, 21, 0, 0)], [(0, 3, 3, 0)], [(0, 3, 1, 8)]):
        with pytest.raises(ValueError, match="category span"):
            ch.encode_chunk(text, tok, None, piece_cache={}, strict=True, category_spans=bad)


def test_collator_pads_cat_spans_and_mask():
    tok = TinyTok(ALPHA)
    a = ch.encode_chunk(ALPHA[:10], tok, None, piece_cache={}, strict=True, category_spans=[(0, 3, 0, 1), (4, 6, 1, 2)])
    b = ch.encode_chunk(ALPHA[:8], tok, None, piece_cache={}, strict=True)
    batch = ch.CharNerCollator(tok.pad_token_id)([a, b])
    assert batch["cat_spans"].shape == (2, 2, 4) and batch["cat_spans"].dtype == torch.long
    assert batch["cat_mask"].tolist() == [[True, True], [False, False]]
    assert batch["cat_spans"][0].tolist() == [[0, 3, 0, 1], [4, 6, 1, 2]] and batch["cat_spans"][1].tolist() == [[0, 0, 0, 0]] * 2
    empty = ch.CharNerCollator(tok.pad_token_id)([b])
    assert empty["cat_spans"].shape == (1, 1, 4) and empty["cat_mask"].tolist() == [[False]]


def test_model_forward_ignores_extra_batch_keys_and_predict_path_unchanged():
    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    f = ch.encode_chunk(doc, tok, None, piece_cache={}, strict=True, category_spans=[(5, 9, 0, 0)])
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    out = model(**batch)                                              # cat_spans / cat_mask は **_unused に落ちる
    assert "logits" in out and out["token_hidden"].shape[-1] == DIM
    spans = ch.predict_char_spans([doc], model, tok)
    assert [(s.start, s.end, s.label) for s in spans[0]] == [(5, 9, "PERSON"), (30, 33, "LOCATION")]


def test_category_loss_value_and_empty_batch():
    tok = TinyTok(ALPHA)
    H = DIM
    text = ALPHA[:30]
    f = ch.encode_chunk(text, tok, None, piece_cache={}, strict=True,
                        category_spans=[(2, 6, 0, 1), (10, 14, 0, 1), (20, 24, 2, 5)])
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    torch.manual_seed(0)
    aux = ch.CategoryAuxHead(H)
    hidden = torch.randn(1, batch["input_ids"].shape[1], H, requires_grad=True)
    loss, n = ch.category_loss(aux, hidden, batch)
    assert n == 3 and loss.ndim == 0 and float(loss.detach()) > 0
    # 手計算: kind ごとの CE を件数で重み付けして平均したもの
    feats, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"],
                                     batch["cat_spans"], batch["cat_mask"])
    ce_org = torch.nn.functional.cross_entropy(aux.layers["org"](feats[:2]), torch.tensor([1, 1]), reduction="sum")
    ce_loc = torch.nn.functional.cross_entropy(aux.layers["location"](feats[2:]), torch.tensor([5]), reduction="sum")
    assert torch.allclose(loss, (ce_org + ce_loc) / 3, atol=1e-6)
    loss.backward()
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0   # 勾配が encoder 側へ流れる
    # クラス重み
    w = {"org": torch.ones(len(ch.CATEGORY_NAMES["org"])) * 2.0, "facility": None, "location": None}
    loss_w, _ = ch.category_loss(aux, hidden.detach(), batch, class_weights=w)
    # 正規化は件数 N で厳密に割る（org だけ重み 2.0。実装は Σ(w·CE)/N であって Σ(w·CE)/Σw ではない）
    assert torch.allclose(loss_w, (2.0 * ce_org + ce_loc) / 3, atol=1e-6)
    # 有効スパンが無い batch → 0（勾配グラフ付き）
    g = ch.encode_chunk(ALPHA[:8], tok, None, piece_cache={}, strict=True)
    empty = ch.CharNerCollator(tok.pad_token_id)([g])
    h2 = torch.randn(1, empty["input_ids"].shape[1], H, requires_grad=True)
    z, n0 = ch.category_loss(aux, h2, empty)
    assert n0 == 0 and float(z.detach()) == 0.0 and z.requires_grad


def test_category_loss_aligns_rows_with_padding_when_batch_size_is_two():
    """B=2・1 行目がカテゴリスパン 1 件で padding される batch で、行対応がズレないことを固定する。

    1 行目は org のスパン 1 件（S_max=2 まで padding）、2 行目は facility・location のスパン 2 件。
    ``cat_mask`` は [[True, False], [True, True]] になるので、``batch["cat_spans"][batch["cat_mask"]]``
    （行優先 = ``span_ctx_batch`` の feats と同じ順）は [1 行目のスパン, 2 行目のスパン a, 2 行目のスパン b]
    の順に並ぶはず。``category_loss`` 内の ``meta = batch["cat_spans"][batch["cat_mask"]]`` の代わりに
    ``cat_spans.reshape(-1, 4)[:N]`` のような padding を無視した詰め方をすると、1 行目の padding 行が
    2 行目の 2 件目より先に拾われてここがズレる（mutation は docstring 止まりで、別途スクラッチで検査する）。
    """
    tok = TinyTok(ALPHA)
    text0 = ALPHA[:30]
    text1 = ALPHA[:30]
    f0 = ch.encode_chunk(text0, tok, None, piece_cache={}, strict=True, category_spans=[(2, 6, 0, 1)])
    f1 = ch.encode_chunk(text1, tok, None, piece_cache={}, strict=True, category_spans=[(2, 6, 1, 0), (20, 24, 2, 3)])
    batch = ch.CharNerCollator(tok.pad_token_id)([f0, f1])
    assert batch["cat_spans"].shape == (2, 2, 4)
    assert batch["cat_mask"].tolist() == [[True, False], [True, True]]
    # 行優先 (b, s) の順 = [1 行目のスパン, 2 行目のスパン a, 2 行目のスパン b]
    assert batch["cat_spans"][batch["cat_mask"]].tolist() == [[2, 6, 0, 1], [2, 6, 1, 0], [20, 24, 2, 3]]

    torch.manual_seed(1)
    H = DIM
    aux = ch.CategoryAuxHead(H)
    hidden = torch.randn(2, batch["input_ids"].shape[1], H, requires_grad=True)
    loss, n = ch.category_loss(aux, hidden, batch)
    assert n == 3

    # 手計算: span_ctx_batch の feats も同じ行優先の順で並ぶので、1 件ずつ正しい kind / category に対応付けて CE を取る
    feats, valid = ch.span_ctx_batch(hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"],
                                     batch["cat_spans"], batch["cat_mask"])
    assert valid.tolist() == [True, True, True]
    ce_org = torch.nn.functional.cross_entropy(aux.layers["org"](feats[0:1]), torch.tensor([1]), reduction="sum")
    ce_fac = torch.nn.functional.cross_entropy(aux.layers["facility"](feats[1:2]), torch.tensor([0]), reduction="sum")
    ce_loc = torch.nn.functional.cross_entropy(aux.layers["location"](feats[2:3]), torch.tensor([3]), reduction="sum")
    assert torch.allclose(loss, (ce_org + ce_fac + ce_loc) / 3, atol=1e-6)


def test_category_loss_skips_spans_without_subwords():
    tok = TinyTok(ALPHA)
    f = ch.encode_chunk(ALPHA[:20], tok, None, piece_cache={}, strict=True, category_spans=[(2, 6, 0, 1), (10, 12, 1, 2)])
    f["char_mask"] = [1] * 8 + [0] * 12                                 # 後半の文字はサブワードに乗っていない扱い
    batch = ch.CharNerCollator(tok.pad_token_id)([f])
    aux = ch.CategoryAuxHead(DIM)
    hidden = torch.randn(1, batch["input_ids"].shape[1], DIM)
    loss, n = ch.category_loss(aux, hidden, batch)
    assert n == 1


def test_sqrt_class_weights_matches_training_tool():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    pytest.importorskip("numpy"); pytest.importorskip("safetensors.numpy")
    import train_category_head as tch
    y = ["a"] * 100 + ["b"] * 1 + ["c"] * 4
    counts = {"a": 100, "b": 1, "c": 4}
    assert ch.sqrt_class_weights(counts, ["a", "b", "c", "d"]) == pytest.approx(tch._class_weights(y, ["a", "b", "c", "d"], mode="sqrt", cap=5.0))
    big = {f"c{i}": 10000 for i in range(8)}; big["c0"] = 1            # 1 クラスだけ稀 → 8/1.07 ≒ 7.5 が上限 5 で切られる
    cats8 = [f"c{i}" for i in range(8)]
    assert max(ch.sqrt_class_weights(big, cats8)) == pytest.approx(5.0)
    y8 = [c for c in cats8 for _ in range(big[c])]
    assert ch.sqrt_class_weights(big, cats8) == pytest.approx(tch._class_weights(y8, cats8, mode="sqrt", cap=5.0))


def test_category_spans_by_text_joins_and_counts():
    labels = [
        {"doc_id": "train:0", "split": "head_train", "kind": "org", "start": 0, "end": 4, "category": "manufacturing"},
        {"doc_id": "train:0", "split": "head_dev", "kind": "org", "start": 5, "end": 8, "category": "it"},
        {"doc_id": "train:1", "split": "head_judge", "kind": "facility", "start": 0, "end": 4, "category": "medical"},   # judge は使わない
        {"doc_id": "train:9", "split": "head_train", "kind": "location", "start": 0, "end": 3, "category": "city"},      # 本文が無い
    ]
    texts = {"train:0": "青葉電機とアオバの件", "train:1": "汐見病院へ"}
    by_text, st = ch.category_spans_by_text(labels, texts)
    assert by_text == {"青葉電機とアオバの件": [(0, 4, 0, ch.CATEGORY_INDEX["org"]["manufacturing"]), (5, 8, 0, ch.CATEGORY_INDEX["org"]["it"])]}
    assert st == {"n_rows": 3, "n_used": 2, "n_unjoined": 1, "by_kind": {"org": 2}, "by_category": {"org/manufacturing": 1, "org/it": 1}}
    with pytest.raises(ValueError, match="category"):
        ch.category_spans_by_text([dict(labels[0], category="nope")], texts)

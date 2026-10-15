"""R3 / O1 / Q2 の回帰テスト。

R3: 全ラベル IGNORE_INDEX の batch で loss が NaN になり、呼び出し側が
    「更新すべきでない batch」だと判別できなかった。
O1: tok_proj を gather より先に掛ける順序交換が、forward と勾配を変えないこと。
Q2: 保存/読み込みの契約（既定は推論専用、for_training で学習設定を復元）。

モデルはダウンロードせず、ローカルで組み立てた小さな BPE tokenizer + 小さな BERT を使う
（test_char_head.py の _tiny_tokenizer / _tiny_model と同じ作り）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


def _load_char_head():
    """テストと同じフォルダの char_head.py を優先し、無ければパッケージ版を使う。

    修正作業中は作業ディレクトリのコピーを、リポジトリへ取り込んだ後は
    privacy_masker.char_head を対象にするため。
    """
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_head.py")
    if os.path.exists(local):
        spec = importlib.util.spec_from_file_location("char_head_under_test", local)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["char_head_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod
    from privacy_masker import char_head  # noqa: PLC0415

    return char_head


ch = _load_char_head()

IGNORE_INDEX = ch.IGNORE_INDEX
LABEL_LIST = ch.LABEL_LIST

CASE = "医療法人伊織会の理事長である河村敬史"
CASE_ENTS = [{"start": 0, "end": 7, "label": "ORG"}, {"start": 14, "end": 18, "label": "PERSON"}]


def _tiny_tokenizer(model_max_length: int = 512):
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


def _tiny_model(tokenizer, **kwargs):
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=16, max_position_embeddings=512,
        pad_token_id=3,
    )
    encoder = transformers.BertModel(config)
    kwargs.setdefault("char_dim", 6)
    kwargs.setdefault("kernel_size", 3)
    kwargs.setdefault("dropout", 0.0)
    return ch.CharBioNerModel(encoder, **kwargs)


def _n_valid(feature) -> int:
    return sum(1 for x in feature["labels"] if x != IGNORE_INDEX)


# ---------------------------------------------------------------------------
# R3: 全無視 batch
# ---------------------------------------------------------------------------

def test_all_ignored_batch_returns_finite_zero_loss_and_zero_valid_count():
    """全ラベル -100 の batch。以前は F.cross_entropy が 0/0 で NaN を返していた。"""
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok)
    coll = ch.CharNerCollator(tok.pad_token_id or 0)

    f = ch.encode_chunk(CASE, tok, CASE_ENTS)
    f["labels"] = [IGNORE_INDEX] * len(f["labels"])
    out = model(**coll([f]))

    assert torch.isfinite(out["loss"]), out["loss"]
    assert out["loss"].item() == 0.0
    # 呼び出し側が「更新すべきでない batch」と判別する材料
    assert int(out["n_valid_labels"]) == 0

    # 誤って backward しても NaN 勾配を作らず、更新量は 0
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            assert p.grad.abs().max().item() == 0.0, name


def test_all_ignored_batch_stays_zero_even_with_nonfinite_logits():
    """0 の作り方が logits に依存しないこと（logits*0 では NaN が残る）。

    学習時の NaN 検出そのものはここでは扱わない。有効ラベル 0 件の batch が
    NaN を loss 集計へ流し込まないことだけを保証する。
    """
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    with torch.no_grad():
        model.head.classifier.bias.fill_(float("nan"))
    coll = ch.CharNerCollator(tok.pad_token_id or 0)

    f = ch.encode_chunk(CASE, tok, CASE_ENTS)
    f["labels"] = [IGNORE_INDEX] * len(f["labels"])
    out = model(**coll([f]))
    assert not torch.isfinite(out["logits"]).any()
    assert out["loss"].item() == 0.0
    assert int(out["n_valid_labels"]) == 0


def test_all_o_negative_sample_is_a_valid_training_sample():
    """全ラベル O の負例は「無視対象」ではない。除外せず有限の loss を出す。"""
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    coll = ch.CharNerCollator(tok.pad_token_id or 0)

    text = "あいうえお"
    f = ch.encode_chunk(text, tok, [])
    assert set(f["labels"]) == {ch.LABEL2ID["O"]}          # 前提: 全 O
    with torch.no_grad():
        out = model(**coll([f]))
    assert int(out["n_valid_labels"]) == len(text)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0.0


def test_partially_ignored_batch_counts_only_valid_labels():
    """batch の一部だけが全無視でも、残りの有効ラベルで学習できる。"""
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    coll = ch.CharNerCollator(tok.pad_token_id or 0)

    fa = ch.encode_chunk("河村敬史", tok, [{"start": 0, "end": 4, "label": "PERSON"}])
    fb = ch.encode_chunk("あいうえお", tok, [])
    fb["labels"] = [IGNORE_INDEX] * len(fb["labels"])
    with torch.no_grad():
        out = model(**coll([fa, fb]))
    assert int(out["n_valid_labels"]) == _n_valid(fa)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0.0


def test_loss_is_still_the_valid_character_weighted_mean():
    """既存の性質の維持: batch の loss = サンプルごとの loss の有効文字数加重平均。"""
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    coll = ch.CharNerCollator(tok.pad_token_id or 0)

    fa = ch.encode_chunk("河村敬史", tok, [{"start": 0, "end": 4, "label": "PERSON"}])
    fb = ch.encode_chunk(CASE, tok, CASE_ENTS)
    with torch.no_grad():
        batched = model(**coll([fa, fb]))
        la = model(**coll([fa]))["loss"].item()
        lb = model(**coll([fb]))["loss"].item()
    na, nb = _n_valid(fa), _n_valid(fb)
    assert int(batched["n_valid_labels"]) == na + nb
    assert batched["loss"].item() == pytest.approx((la * na + lb * nb) / (na + nb), rel=1e-6)


def test_padding_still_never_reaches_the_loss():
    """既存の性質の維持: padding 位置の入力を汚しても loss は変わらない。"""
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    coll = ch.CharNerCollator(tok.pad_token_id or 0)
    b = coll([
        ch.encode_chunk("河村敬史", tok, [{"start": 0, "end": 4, "label": "PERSON"}]),
        ch.encode_chunk(CASE, tok, CASE_ENTS),
    ])
    pad = b["char_mask"] == 0
    assert bool(pad.any())
    with torch.no_grad():
        base = model(**b)["loss"].item()
        dirty = {k: v.clone() for k, v in b.items()}
        dirty["char_piece_ids"][pad] = 4
        dirty["char_script"][pad] = 3
        assert model(**dirty)["loss"].item() == pytest.approx(base, abs=1e-6)


# ---------------------------------------------------------------------------
# O1: tok_proj と gather の順序交換
# ---------------------------------------------------------------------------

def _gather_then_project(head, token_hidden, piece_embeds, char_to_token,
                         char_script, char_pos_fwd, char_pos_bwd, char_mask):
    """O1 変更前の順序（文字数へ展開してから tok_proj）の参照実装。"""
    _, t, h = token_hidden.shape
    idx = char_to_token.clamp(min=0, max=max(t - 1, 0))
    gathered = torch.gather(token_hidden, 1, idx.unsqueeze(-1).expand(-1, -1, h))
    x = (
        head.tok_proj(gathered)
        + head.piece_proj(piece_embeds)
        + head.script_emb(char_script)
        + head.pos_fwd_emb(char_pos_fwd)
        + head.pos_bwd_emb(char_pos_bwd)
    )
    x = F.gelu(head.in_norm(x))
    m = char_mask.to(x.dtype).unsqueeze(-1)
    x = x * m
    if head.use_mix:
        y = ch.depthwise_shift_mix(x, head.mix_weight) + head.mix_bias
        y = head.pw(F.gelu(y))
        x = head.mix_norm(x + head.dropout(y)) * m
    else:
        x = head.dropout(x)
    return head.classifier(x)


def test_projection_before_gather_matches_the_previous_order():
    """forward と全パラメータ勾配が変更前の順序と一致する。

    許容誤差: 各テンソルの最大絶対値に対する相対差 <= 1e-5。float32 で
    C=256 項を足す和の順序が変わるので厳密一致は要求しない（eps*sqrt(N) 相当）。
    """
    torch.manual_seed(0)
    hidden, char_dim, b, t, c = 32, 16, 2, 24, 41
    head = ch.CharMixHead(hidden_size=hidden, char_dim=char_dim, kernel_size=5, dropout=0.0).eval()

    g = torch.Generator().manual_seed(1)
    inputs = dict(
        piece_embeds=torch.randn(b, c, hidden, generator=g),
        char_to_token=torch.randint(0, t, (b, c), generator=g),
        char_script=torch.randint(0, ch.N_SCRIPT, (b, c), generator=g),
        char_pos_fwd=torch.randint(0, ch.N_INTRA, (b, c), generator=g),
        char_pos_bwd=torch.randint(0, ch.N_INTRA, (b, c), generator=g),
        char_mask=torch.ones(b, c, dtype=torch.long),
    )
    inputs["char_mask"][:, c - 5:] = 0                      # padding も含める
    token_hidden = torch.randn(b, t, hidden, generator=g)
    grad_out = torch.randn(b, c, ch.NUM_LABELS, generator=g)

    def run(fn):
        th = token_hidden.clone().requires_grad_(True)
        out = fn(token_hidden=th, **inputs)
        out.backward(grad_out)
        grads = {n: p.grad.clone() for n, p in head.named_parameters()}
        grads["<token_hidden>"] = th.grad.clone()
        head.zero_grad(set_to_none=True)
        return out.detach(), grads

    out_new, grad_new = run(head.forward)
    out_old, grad_old = run(lambda **kw: _gather_then_project(head, **kw))

    scale = out_old.abs().max().item()
    assert (out_new - out_old).abs().max().item() <= 1e-5 * scale

    assert set(grad_new) == set(grad_old)
    for name in grad_old:
        ref = grad_old[name]
        diff = (grad_new[name] - ref).abs().max().item()
        assert diff <= 1e-5 * max(ref.abs().max().item(), 1e-6), (name, diff)


# ---------------------------------------------------------------------------
# Q2: 保存 / 読み込みの契約
# ---------------------------------------------------------------------------

def _spans(model, tok, text=CASE):
    return [(s.start, s.end, s.label, round(s.score, 9))
            for s in ch.predict_char_spans([text], model, tok)[0]]


def test_inference_round_trip_reproduces_logits_exactly(tmp_path):
    """既定（for_training=False）は推論専用。保存前と logits まで一致する。"""
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok, dropout=0.3, class_weights=[2.0] * len(LABEL_LIST)).eval()
    coll = ch.CharNerCollator(tok.pad_token_id or 0)
    b = coll([ch.encode_chunk(CASE, tok, CASE_ENTS)])
    with torch.no_grad():
        before_logits = model(**b)["logits"].clone()
    before = _spans(model, tok)

    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)
    restored = ch.load_char_model(str(save_dir))

    assert not restored.training
    assert restored.head.dropout.p == 0.0
    assert restored.class_weights is None            # 推論には効かないので戻さない
    with torch.no_grad():
        after_logits = restored(**b)["logits"]
    assert torch.equal(before_logits, after_logits)
    assert before == _spans(restored, tok)


def test_load_for_training_restores_dropout_and_class_weights(tmp_path):
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    weights = [3.0 if lab.startswith("B-") else 1.0 for lab in LABEL_LIST]
    model = _tiny_model(tok, dropout=0.25, class_weights=weights)

    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)
    cfg = json.loads((save_dir / ch.HEAD_CONFIG_NAME).read_text(encoding="utf-8"))
    assert cfg["dropout"] == pytest.approx(0.25)
    assert cfg["class_weights"] == pytest.approx(weights)

    resumed = ch.load_char_model(str(save_dir), for_training=True)
    assert resumed.training
    assert resumed.head.dropout.p == pytest.approx(0.25)
    assert resumed.class_weights is not None
    assert resumed.class_weights.tolist() == pytest.approx(weights)

    # class_weights が実際に loss へ効いていること（buffer が飾りでないこと）
    plain = ch.load_char_model(str(save_dir))
    coll = ch.CharNerCollator(tok.pad_token_id or 0)
    b = coll([ch.encode_chunk(CASE, tok, CASE_ENTS)])
    resumed.eval()
    with torch.no_grad():
        assert resumed(**b)["loss"].item() != pytest.approx(plain(**b)["loss"].item(), abs=1e-6)


def test_inference_load_rejects_training_only_arguments(tmp_path):
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)

    with pytest.raises(ValueError, match="for_training=True"):
        ch.load_char_model(str(save_dir), dropout=0.1)
    with pytest.raises(ValueError, match="for_training=True"):
        ch.load_char_model(str(save_dir), class_weights=[1.0] * len(LABEL_LIST))


def _make_legacy(save_dir):
    """dropout / class_weights を書かない旧形式の config に落とす。"""
    path = os.path.join(str(save_dir), ch.HEAD_CONFIG_NAME)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("dropout", None)
    cfg.pop("class_weights", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def test_legacy_config_still_loads_for_inference(tmp_path):
    """後方互換: 旧形式の char_head_config.json でも推論復元はできる。"""
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)
    before = _spans(model, tok)

    cfg = _make_legacy(save_dir)
    assert "dropout" not in cfg and "class_weights" not in cfg

    restored = ch.load_char_model(str(save_dir))
    assert restored.head.dropout.p == 0.0
    assert before == _spans(restored, tok)


def test_legacy_config_requires_an_explicit_dropout_for_training(tmp_path):
    """旧形式で学習を再開するなら、既定値へ黙って化けさせず明示させる。"""
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)
    _make_legacy(save_dir)

    with pytest.raises(ValueError, match="dropout"):
        ch.load_char_model(str(save_dir), for_training=True)

    resumed = ch.load_char_model(str(save_dir), for_training=True, dropout=0.1)
    assert resumed.training
    assert resumed.head.dropout.p == pytest.approx(0.1)
    assert resumed.class_weights is None


def test_save_records_no_optimizer_state(tmp_path):
    """契約の明示: 保存物は重み＋構成＋学習設定まで。厳密な学習再開には足りない。"""
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    save_dir = tmp_path / "best"
    ch.save_char_model(model, str(save_dir), tokenizer=tok)
    names = {p.name for p in save_dir.iterdir()}
    assert not any("optim" in n or "scheduler" in n or "scaler" in n for n in names), names
    # docstring に成立しない保証を書かないための固定。契約が消えたら落ちる。
    for doc in (ch.save_char_model.__doc__, ch.load_char_model.__doc__):
        assert "optimizer" in doc
        assert "厳密な学習再開" in doc

"""文字単位BIOヘッドのテスト。

モデルはダウンロードせず、ローカルで組み立てた小さな BPE tokenizer + 小さな BERT を使う。
tokenizer には ``会の`` を 1 トークンにする merge を仕込んであるので、本番モデルで
起きている「エンティティ境界がトークン内部に落ちる」状況をそのまま再現できる。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from privacy_masker.char_head import (  # noqa: E402
    ENTITY_TYPES,
    ID2LABEL,
    IGNORE_INDEX,
    LABEL_LIST,
    CharBioNerModel,
    CharCoverageError,
    CharNerCollator,
    CharSpan,
    char_labels_from_entities,
    core_ranges,
    decode_char_labels,
    encode_chunk,
    load_char_model,
    merge_adjacent_same_type,
    predict_char_spans,
    save_char_model,
    script_class,
    token_safe_chunks,
)

CASE = "医療法人伊織会の理事長である河村敬史"
CASE_ENTS = [{"start": 0, "end": 7, "label": "ORG"}, {"start": 14, "end": 18, "label": "PERSON"}]


def _tiny_tokenizer(model_max_length: int = 512):
    """``会の`` を 1 トークンにまとめる小さな BPE tokenizer を作る。"""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    chars = sorted(set(CASE + "あいうえおアイウエオ0123456789abcXYZ、。 "))
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    vocab["会の"] = len(vocab)                      # これが境界をまたぐトークン
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
    return CharBioNerModel(encoder, **kwargs)


# ---------------------------------------------------------------------------
# ラベル生成
# ---------------------------------------------------------------------------

def test_char_labels_put_the_boundary_inside_a_token():
    tok = _tiny_tokenizer()
    # 前提: tokenizer が実際に境界をまたぐトークンを作る
    assert "会の" in tok.tokenize(CASE)

    labels = char_labels_from_entities(CASE, CASE_ENTS)
    assert len(labels) == len(CASE)
    assert ID2LABEL[labels[0]] == "B-ORG"        # 医
    assert ID2LABEL[labels[6]] == "I-ORG"        # 会 = スパン最終文字
    assert ID2LABEL[labels[7]] == "O"            # の = 同一トークンだが境界の外
    assert ID2LABEL[labels[14]] == "B-PERSON"    # 河
    assert ID2LABEL[labels[17]] == "I-PERSON"    # 史


def test_char_labels_prefer_the_longer_span_on_overlap():
    text = "東京都新宿区西新宿"
    labels = char_labels_from_entities(
        text, [{"start": 0, "end": 9, "label": "ADDRESS"}, {"start": 0, "end": 3, "label": "LOCATION"}]
    )
    assert [ID2LABEL[i] for i in labels] == ["B-ADDRESS"] + ["I-ADDRESS"] * 8


def test_char_labels_drop_unknown_types_as_o():
    labels = char_labels_from_entities("abc", [{"start": 0, "end": 3, "label": "EMAIL"}])
    assert [ID2LABEL[i] for i in labels] == ["O", "O", "O"]


def test_script_class_separates_the_scripts_used_for_boundaries():
    assert script_class("会") != script_class("の")          # 漢字 vs ひらがな
    assert script_class("ア") != script_class("あ")
    assert script_class("1") != script_class("a")
    assert script_class(" ") == script_class("　")


# ---------------------------------------------------------------------------
# 分割と core 割り当て
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["医", "abc ", "0123456789"])
def test_token_safe_chunks_never_exceed_max_length(unit):
    tok = _tiny_tokenizer()
    text = (unit * 500)[:900]
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=20, max_length=48)
    assert chunks
    for _, piece in chunks:
        n = len(tok(piece, add_special_tokens=True, truncation=False)["input_ids"])
        assert n <= 48, f"{n} tokens > 48"


def test_token_safe_chunks_cover_every_character():
    tok = _tiny_tokenizer()
    text = "医療法人伊織会の理事長である河村敬史" * 40
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=20, max_length=64)
    seen = set()
    for off, piece in chunks:
        seen.update(range(off, off + len(piece)))
    assert seen == set(range(len(text)))


def test_core_ranges_tile_the_document_exactly():
    tok = _tiny_tokenizer()
    text = "医" * 500
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=20, max_length=256)
    cores = core_ranges(chunks, len(text))
    assert len(cores) == len(chunks)
    assert cores[0][0] == 0 and cores[-1][1] == len(text)
    for (a, b), (c, _) in zip(cores, cores[1:]):
        assert a < b == c                       # 隙間も重なりもない
    for (ca, cb), (off, piece) in zip(cores, chunks):
        assert off <= ca < cb <= off + len(piece)   # core は必ず担当チャンクの内側


def test_core_ranges_handles_a_single_chunk():
    assert core_ranges([(0, "abc")], 3) == [(0, 3)]
    assert core_ranges([], 0) == []


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def test_encode_chunk_maps_every_character_to_a_token():
    tok = _tiny_tokenizer()
    f = encode_chunk(CASE, tok, CASE_ENTS)
    assert f["uncovered"] == 0
    assert len(f["char_to_token"]) == len(CASE)
    # 会 と の は同じトークンに乗るが、トークン内オフセットで区別できる
    assert f["char_to_token"][6] == f["char_to_token"][7]
    assert (f["char_pos_fwd"][6], f["char_pos_bwd"][6]) == (0, 1)
    assert (f["char_pos_fwd"][7], f["char_pos_bwd"][7]) == (1, 0)
    # ラベルは境界どおり
    assert ID2LABEL[f["labels"][6]] == "I-ORG"
    assert ID2LABEL[f["labels"][7]] == "O"


def test_encode_chunk_is_fail_closed_on_truncation():
    tok = _tiny_tokenizer()
    text = "医" * 200
    with pytest.raises(CharCoverageError) as ei:
        encode_chunk(text, tok, None, max_length=16, strict=True)
    assert "token_safe_chunks" in str(ei.value)

    f = encode_chunk(text, tok, [{"start": 0, "end": 200, "label": "ORG"}], max_length=16, strict=False)
    assert f["uncovered"] > 0
    # 未被覆文字は O ではなく「教えない」
    assert f["labels"][-1] == IGNORE_INDEX
    assert f["char_mask"][-1] == 0


def test_collator_pads_both_axes_and_ignores_padding():
    tok = _tiny_tokenizer()
    coll = CharNerCollator(tok.pad_token_id or 0)
    fa = encode_chunk("河村敬史", tok, [{"start": 0, "end": 4, "label": "PERSON"}])
    fb = encode_chunk(CASE, tok, CASE_ENTS)
    b = coll([fa, fb])
    assert b["labels"].shape == (2, len(CASE))
    pad = b["char_mask"] == 0
    assert pad.any()
    assert bool((b["labels"][pad] == IGNORE_INDEX).all())


# ---------------------------------------------------------------------------
# デコード
# ---------------------------------------------------------------------------

def _ids(*types):
    return [0 if t is None else ENTITY_TYPES.index(t) + 1 for t in types]


def test_decode_splits_on_b_so_adjacent_entities_stay_separate():
    text = "山田太郎佐藤花子"
    ids = _ids(*(["PERSON"] * 8))
    starts = [i in (0, 4) for i in range(8)]
    spans = decode_char_labels(text, ids, [0.9] * 8, starts)
    assert [(s.start, s.end) for s in spans] == [(0, 4), (4, 8)]


def test_decode_respects_a_b_that_lands_on_a_chunk_seam():
    """継ぎ目に正しい B- が来ても分離する（レビュー R1）。

    以前は継ぎ目の B- を無条件に無視してスパンを継続していた。担当が切り替わる位置を
    担当するのは後続チャンクで、そこには overlap の半分の左文脈があるので継続なら I- を
    出せる。文脈不足で偽 B- が出るのはチャンク自身のテキスト先頭で、そこは core の外。
    つまり抑制の利益は極めて狭く、代償は「隣接する別人が 1 人に融合する」ことだった。
    """
    text = "山田太郎佐藤花子"
    ids = _ids(*(["PERSON"] * 8))
    starts = [i == 4 for i in range(8)]
    seam = [i == 4 for i in range(8)]
    spans = decode_char_labels(text, ids, [0.9] * 8, starts, seam)
    assert [(s.start, s.end) for s in spans] == [(0, 4), (4, 8)]


def test_decode_seam_suppression_is_opt_in():
    """旧挙動は suppress_seam_b=True でだけ得られる（"core" 戦略の後方互換用）。"""
    text = "山田太郎佐藤花子"
    ids = _ids(*(["PERSON"] * 8))
    starts = [i == 4 for i in range(8)]
    seam = [i == 4 for i in range(8)]
    spans = decode_char_labels(text, ids, [0.9] * 8, starts, seam, suppress_seam_b=True)
    assert [(s.start, s.end) for s in spans] == [(0, 8)]


def test_merge_adjacent_same_type_only_merges_touching_same_type():
    assert merge_adjacent_same_type(
        [CharSpan(0, 1, "PERSON", 1.0), CharSpan(1, 4, "PERSON", 0.5)]
    ) == [CharSpan(0, 4, "PERSON", pytest.approx(0.625))]
    # 隙間があれば結合しない
    assert len(merge_adjacent_same_type([CharSpan(0, 4, "PERSON"), CharSpan(5, 9, "PERSON")])) == 2
    # 型が違えば結合しない
    assert len(merge_adjacent_same_type([CharSpan(0, 3, "ORG"), CharSpan(3, 6, "PERSON")])) == 2


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def test_forward_returns_one_logit_row_per_character():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    coll = CharNerCollator(tok.pad_token_id or 0)
    b = coll([encode_chunk(CASE, tok, CASE_ENTS)])
    out = model(**b)
    assert out["logits"].shape == (1, len(CASE), len(LABEL_LIST))
    assert torch.isfinite(out["loss"])


def test_padding_never_reaches_the_loss():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok).eval()
    coll = CharNerCollator(tok.pad_token_id or 0)
    b = coll([
        encode_chunk("河村敬史", tok, [{"start": 0, "end": 4, "label": "PERSON"}]),
        encode_chunk(CASE, tok, CASE_ENTS),
    ])
    pad = b["char_mask"] == 0
    with torch.no_grad():
        base = model(**b)["loss"].item()
        dirty = {k: v.clone() for k, v in b.items()}
        dirty["char_piece_ids"][pad] = 4
        dirty["char_script"][pad] = 3
        assert model(**dirty)["loss"].item() == pytest.approx(base, abs=1e-6)


def test_class_weights_change_the_loss():
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    plain = _tiny_model(tok).eval()
    weights = [3.0 if lab.startswith("B-") else 1.0 for lab in LABEL_LIST]
    weighted = _tiny_model(tok, class_weights=weights).eval()
    weighted.encoder.load_state_dict(plain.encoder.state_dict())
    weighted.head.load_state_dict(plain.head.state_dict())
    coll = CharNerCollator(tok.pad_token_id or 0)
    b = coll([encode_chunk(CASE, tok, CASE_ENTS)])
    with torch.no_grad():
        assert plain(**b)["loss"].item() != pytest.approx(weighted(**b)["loss"].item(), abs=1e-6)


def test_class_weights_length_is_validated():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="class_weights"):
        _tiny_model(tok, class_weights=[1.0, 2.0])


def test_added_parameters_stay_a_small_fraction_of_the_encoder():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    assert model.num_added_parameters() == sum(p.numel() for p in model.head.parameters())
    assert model.num_added_parameters() < sum(p.numel() for p in model.encoder.parameters())


def test_predict_covers_every_character_of_a_long_document():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok).eval()
    text = CASE * 30
    spans = predict_char_spans(
        [text], model, tok, chunk_chars=100, chunk_overlap=20, max_length=64
    )[0]
    for s in spans:
        assert 0 <= s.start < s.end <= len(text)
        assert s.label in ENTITY_TYPES
    # 隣接同型を結合しても文字の被覆は変わらない
    merged = predict_char_spans(
        [text], model, tok, chunk_chars=100, chunk_overlap=20, max_length=64, merge_adjacent=True
    )[0]
    assert {i for s in spans for i in range(s.start, s.end)} == {
        i for s in merged for i in range(s.start, s.end)
    }


def test_predict_is_deterministic():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok).eval()
    runs = [
        [(s.start, s.end, s.label) for s in predict_char_spans([CASE], model, tok)[0]]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_save_and_load_round_trip(tmp_path):
    pytest.importorskip("safetensors")
    tok = _tiny_tokenizer()
    model = _tiny_model(tok).eval()
    before = [(s.start, s.end, s.label, round(s.score, 6)) for s in predict_char_spans([CASE], model, tok)[0]]

    save_dir = tmp_path / "best"
    save_char_model(model, str(save_dir), tokenizer=tok, extra={"note": "test"})
    assert (save_dir / "char_head.safetensors").exists()
    assert (save_dir / "char_head_config.json").exists()

    restored = load_char_model(str(save_dir))
    after = [(s.start, s.end, s.label, round(s.score, 6)) for s in predict_char_spans([CASE], restored, tok)[0]]
    assert before == after


def test_load_rejects_a_plain_token_model_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="char_head_config.json"):
        load_char_model(str(tmp_path))


def test_boundary_is_learnable_within_a_token():
    """トークン内境界が原理的に学習可能であることの確認（過学習で足りる）。

    ``会の`` が 1 トークンなので、``会`` を I-ORG・``の`` を O にできるのは
    トークン内の位置情報が文字表現に入っているからである。
    """
    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok, char_dim=32)
    coll = CharNerCollator(tok.pad_token_id or 0)
    b = coll([encode_chunk(CASE, tok, CASE_ENTS)])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    first = None
    for _ in range(150):
        loss = model(**b)["loss"]
        first = first if first is not None else loss.item()
        loss.backward(); opt.step(); opt.zero_grad()
    assert loss.item() < first * 0.1

    model.eval()
    got = {s.label: CASE[s.start:s.end] for s in predict_char_spans([CASE], model, tok, merge_adjacent=True)[0]}
    assert got.get("ORG") == "医療法人伊織会", got
    assert got.get("PERSON") == "河村敬史", got

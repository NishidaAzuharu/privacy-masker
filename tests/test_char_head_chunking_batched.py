"""まとめトークン化の出力を、文書ごとに従来経路を通した結果と全キーで比較する。

既存テストと同じく小さな BPE をローカルで組み立てる（Hub には接続しない）。
byte 単位の語彙で日本語・絵文字の膨張も再現し、二分探索と重なりの破棄を通す。
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

import privacy_masker.char_head as ch  # noqa: E402
from tests.test_char_head import _tiny_tokenizer  # noqa: E402
from tests.test_char_head_predict import TinyTok  # noqa: E402


class CountingTokenizer:
    """呼び出しの形と文字語彙の参照順を記録する。符号化は実 tokenizer に委ねる。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.piece_calls = []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, texts, **kwargs):
        self.calls.append((isinstance(texts, list), len(texts), kwargs.copy()))
        return self.tokenizer(texts, **kwargs)

    def convert_tokens_to_ids(self, piece):
        value = self.tokenizer.convert_tokens_to_ids(piece)
        self.piece_calls.append((piece, value))
        return value


def legacy_chunk_and_encode(texts, tokenizer, chunk_chars, chunk_overlap, max_length):
    """公開関数を文書順に呼ぶ、まとめ処理前の参照経路。"""
    cache = {}
    feats, doc_chunks, doc_feats, cores = [], [], [], []
    for text in texts:
        chunks = ch.token_safe_chunks(text, tokenizer, chunk_chars, chunk_overlap, max_length)
        doc_chunks.append(chunks)
        cores.append(ch.core_ranges(chunks, len(text)))
        indices = []
        for _, piece in chunks:
            indices.append(len(feats))
            feats.append(ch.encode_chunk(piece, tokenizer, None, max_length, cache, strict=True))
        doc_feats.append(indices)
    return feats, doc_chunks, doc_feats, cores


@pytest.fixture
def byte_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    for c in sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()):
        vocab[c] = len(vocab)
    word = "abcdefghij"
    merges = []
    for i in range(1, len(word)):
        vocab[word[: i + 1]] = len(vocab)
        merges.append((word[:i], word[i]))
    raw = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, merges, unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
    )


def _random_documents():
    rng = random.Random(20261103)
    alphabet = "あいうえお漢字カタカナabcXYZ0123456789😀🧬🎉 \t\n"
    lengths = [0, 1, 39, 40, 41, 399, 400, 401, 3000] + [rng.randrange(3001) for _ in range(200)]
    return ["".join(rng.choices(alphabet, k=n)) for n in lengths]


@pytest.mark.parametrize("settings", [(400, 50, 512), (40, 5, 48), (40, 39, 18), (80, 0, 96)])
def test_random_documents_match_legacy_exactly(byte_tokenizer, settings):
    texts = _random_documents()
    old_tok = CountingTokenizer(byte_tokenizer)
    new_tok = CountingTokenizer(byte_tokenizer)
    expected = legacy_chunk_and_encode(texts, old_tok, *settings)
    actual = ch._chunk_and_encode(texts, new_tok, *settings)
    assert actual == expected
    # 同じキーを同じ値で一度だけキャッシュし、文書順の更新も維持する。
    assert new_tok.piece_calls == old_tok.piece_calls
    assert len(new_tok.piece_calls) == len(set("".join(texts)))
    feats, chunks, _, _ = actual
    assert all(len(f["input_ids"]) <= settings[2] and f["uncovered"] == 0 for f in feats)
    for text, parts in zip(texts, chunks):
        covered = {i for off, piece in parts for i in range(off, off + len(piece))}
        assert covered == set(range(len(text)))
        assert all(a < b <= a + len(p) < b + len(q) for (a, p), (b, q) in zip(parts, parts[1:]))


def test_one_round_reuses_encoding_for_all_documents():
    texts = ["あいうえおabc012 " * 20] * 500
    old_tok = CountingTokenizer(_tiny_tokenizer())
    new_tok = CountingTokenizer(_tiny_tokenizer())
    expected = legacy_chunk_and_encode(texts, old_tok, 400, 50, 512)
    assert ch._chunk_and_encode(texts, new_tok, 400, 50, 512) == expected
    assert len(old_tok.calls) == 1000
    assert new_tok.calls == [(True, 500, dict(
        add_special_tokens=True, truncation=False, return_offsets_mapping=True,
    ))]


def test_rounds_only_include_unfinished_nonempty_documents():
    tok = CountingTokenizer(_tiny_tokenizer())
    texts = ["", "あ", "い" * 90, "", "う" * 50]
    expected = legacy_chunk_and_encode(texts, _tiny_tokenizer(), 40, 5, 48)
    assert ch._chunk_and_encode(texts, tok, 40, 5, 48) == expected
    assert [(batch, size) for batch, size, _ in tok.calls] == [(True, 3), (True, 2), (True, 1)]


@pytest.mark.parametrize("texts", [[], [""], ["", "", ""]])
def test_empty_documents_do_not_call_tokenizer(texts):
    tok = CountingTokenizer(_tiny_tokenizer())
    assert ch._chunk_and_encode(texts, tok, 400, 50, 512) == legacy_chunk_and_encode(
        texts, tok, 400, 50, 512,
    )
    assert tok.calls == []


def test_truncation_does_not_change_accepted_encodings(byte_tokenizer):
    texts = ["", "abあ😀 \t\n", "\n " * 20, "漢字" * 20]
    batched = byte_tokenizer(texts, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    for i, text in enumerate(texts):
        # 上限ぴったりでも特殊トークン・offsets が一致する。
        scalar = byte_tokenizer(text, add_special_tokens=True, truncation=True,
                                max_length=len(batched["input_ids"][i]), return_offsets_mapping=True)
        for key in ("input_ids", "attention_mask", "offset_mapping"):
            assert batched[key][i] == scalar[key]


def test_dense_suffix_discards_overlap(byte_tokenizer):
    # 先頭の長いピースが窓の移動で崩れ、次の窓が直前の終端まで届かなくなる。
    text = "abcdefghijxxxxxx"
    expected = legacy_chunk_and_encode([text], byte_tokenizer, 12, 11, 6)
    assert [(off, len(piece)) for off, piece in expected[1][0]] == [(0, 12), (12, 4)]
    assert ch._chunk_and_encode([text], byte_tokenizer, 12, 11, 6) == expected


@pytest.mark.parametrize("max_length", [2, 5])
def test_unfittable_character_fails_closed(byte_tokenizer, max_length):
    texts = ["", "a", "😀"]
    with pytest.raises(ch.CharCoverageError) as old:
        legacy_chunk_and_encode(texts, byte_tokenizer, 40, 5, max_length)
    with pytest.raises(ch.CharCoverageError) as new:
        ch._chunk_and_encode(texts, byte_tokenizer, 40, 5, max_length)
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("texts", [[""], ["a"]])
@pytest.mark.parametrize("chunk_chars, overlap", [(40, 40), (40, -1), (0, 0)])
def test_invalid_settings_match_legacy(texts, chunk_chars, overlap):
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError) as old:
        legacy_chunk_and_encode(texts, tok, chunk_chars, overlap, 48)
    with pytest.raises(ValueError) as new:
        ch._chunk_and_encode(texts, tok, chunk_chars, overlap, 48)
    assert str(new.value) == str(old.value)


def test_scalar_only_tokenizer_remains_usable():
    texts = ["あいう" * 40, "", "ういあ" * 20]
    tok = TinyTok("あいう")
    assert ch._chunk_and_encode(texts, tok, 40, 5, 48) == legacy_chunk_and_encode(texts, tok, 40, 5, 48)

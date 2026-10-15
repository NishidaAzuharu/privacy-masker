"""R4a / R4b: token_safe_chunks の停止性と token 上限厳守のテスト（単体で動く）。

人工 tokenizer は tests/test_char_head.py の方針に合わせて自前で組む（重いモデルを
落とさない）。実 tokenizer を使う既定設定の確認だけは、ローカルに best/ がある場合に
実行する（無ければ skip）。
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from privacy_masker.char_head import (  # noqa: E402
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    MAX_LENGTH,
    CharCoverageError,
    core_ranges,
    token_safe_chunks,
)


class _FixedRateTok:
    """1 文字 = ``tok_per_char`` token、特殊 token 2 個の人工 tokenizer。

    呼び出し回数に上限を設けて、分割が進まない（無限ループ）状態を有限時間で検出する。
    """

    pad_token_id = 0
    unk_token_id = 0
    is_fast = True

    def __init__(self, tok_per_char: int = 1, call_limit: int | None = None):
        self.tok_per_char = tok_per_char
        self.call_limit = call_limit
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        if self.call_limit is not None and self.calls > self.call_limit:
            raise AssertionError(f"tokenizer 呼び出しが {self.call_limit} 回を超過（分割が進んでいない）")
        n = len(text) * self.tok_per_char
        return {
            "input_ids": [1] + [2] * n + [3],
            "attention_mask": [1] * (n + 2),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(n)] + [(0, 0)],
        }

    def convert_tokens_to_ids(self, ch):  # encode_chunk 用（このテストでは未使用）
        return 5


def _assert_layout_is_usable(text, chunks, tok, max_length):
    """返却チャンクが「上限内・全文字被覆」で、core_ranges が欠落なく担当できることを確認する。"""
    n = len(text)
    assert chunks, "空の分割結果を返してはいけない"
    for off, piece in chunks:
        assert text[off : off + len(piece)] == piece
        assert len(tok(piece, add_special_tokens=True, truncation=False)["input_ids"]) <= max_length

    covered = set()
    for off, piece in chunks:
        covered.update(range(off, off + len(piece)))
    assert covered == set(range(n))

    cores = core_ranges(chunks, n)
    assert len(cores) == len(chunks)
    assert cores[0][0] == 0 and cores[-1][1] == n
    for (a, b), (c, _) in zip(cores, cores[1:]):
        assert a < b == c, f"core に隙間か重なりがある: {cores}"
    for (ca, cb), (off, piece) in zip(cores, chunks):
        assert off <= ca < cb <= off + len(piece), f"core {(ca, cb)} が chunk {(off, len(piece))} の外"


# ---------------------------------------------------------------------------
# R4a: 分割が必ず進む
# ---------------------------------------------------------------------------

def test_split_makes_progress_with_overlap_larger_than_chunk_piece():
    """レビューの再現条件（1文字1token・20文字・max_length=10・overlap=50）が有限回で終わる。"""
    tok = _FixedRateTok(tok_per_char=1, call_limit=200)
    text = "あ" * 20
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=50, max_length=10)
    _assert_layout_is_usable(text, chunks, tok, 10)


@pytest.mark.parametrize("n_chars", [2, 3, 7, 8, 9, 20, 61, 200])
@pytest.mark.parametrize("max_length", [3, 4, 10, 33])
def test_split_terminates_for_hostile_overlap(n_chars, max_length):
    """overlap が chunk_chars 直前まで大きくても、文字数に対して線形の呼び出し回数で終わる。

    再帰的に半分ずつ割る方式では重なりが各段で積み上がり、呼び出し回数が文字数に対して
    超線形（実測で 200 文字・max_length=10 で 5000 回超）になったので、上限も線形で置く。
    """
    tok = _FixedRateTok(tok_per_char=1, call_limit=12 * n_chars + 50)
    text = "あ" * n_chars
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=99, max_length=max_length)
    _assert_layout_is_usable(text, chunks, tok, max_length)


# ---------------------------------------------------------------------------
# R4b: 最小サイズでも token 上限を免除しない
# ---------------------------------------------------------------------------

def test_small_chunks_still_obey_token_limit():
    """8文字・10token・max_length=9・min_chunk_chars=8 で上限超過を返さない。"""
    tok = _FixedRateTok(tok_per_char=1)
    text = "あ" * 8
    chunks = token_safe_chunks(text, tok, chunk_chars=100, chunk_overlap=2, max_length=9, min_chunk_chars=8)
    _assert_layout_is_usable(text, chunks, tok, 9)


@pytest.mark.parametrize("min_chunk_chars", [1, 8, 50, 400])
def test_min_chunk_chars_does_not_exempt_the_token_limit(min_chunk_chars):
    tok = _FixedRateTok(tok_per_char=2)
    text = "🧬" * 120
    chunks = token_safe_chunks(
        text, tok, chunk_chars=100, chunk_overlap=20, max_length=40, min_chunk_chars=min_chunk_chars
    )
    _assert_layout_is_usable(text, chunks, tok, 40)


def test_single_character_over_the_limit_fails_explicitly():
    """1 文字が上限を超える病的ケースは黙って通さず、原因と対処を書いた例外を出す。"""
    tok = _FixedRateTok(tok_per_char=20, call_limit=5000)
    with pytest.raises(CharCoverageError) as ei:
        token_safe_chunks("あ" * 4, tok, chunk_chars=100, chunk_overlap=2, max_length=10)
    msg = str(ei.value)
    assert "22 token" in msg and "max_length=10" in msg          # 原因（実測 token 数と上限）
    assert "max_length を増やす" in msg                          # 対処


def test_special_tokens_alone_over_the_limit_fails_explicitly():
    tok = _FixedRateTok(tok_per_char=1, call_limit=5000)
    with pytest.raises(CharCoverageError):
        token_safe_chunks("あい", tok, chunk_chars=100, chunk_overlap=2, max_length=2)


# ---------------------------------------------------------------------------
# 従来どおりの挙動（回帰）
# ---------------------------------------------------------------------------

def test_short_text_and_edge_lengths_unchanged():
    tok = _FixedRateTok(tok_per_char=1)
    assert token_safe_chunks("", tok) == []
    assert token_safe_chunks("あ", tok) == [(0, "あ")]
    text = "あ" * CHUNK_CHARS
    assert token_safe_chunks(text, tok) == [(0, text)]           # 1 チャンクで収まる
    assert core_ranges([(0, text)], len(text)) == [(0, len(text))]


def test_default_layout_without_resplit_is_unchanged():
    """再分割が起きない条件では、従来と同じ 400/step350 の配置になる。"""
    tok = _FixedRateTok(tok_per_char=1)
    text = "あ" * 1000
    chunks = token_safe_chunks(text, tok, chunk_chars=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)
    assert [(off, len(p)) for off, p in chunks] == [(0, 400), (350, 400), (700, 300)]
    _assert_layout_is_usable(text, chunks, tok, MAX_LENGTH)


def test_invalid_overlap_is_rejected():
    tok = _FixedRateTok()
    with pytest.raises(ValueError):
        token_safe_chunks("あ" * 10, tok, chunk_chars=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        token_safe_chunks("あ" * 10, tok, chunk_chars=10, chunk_overlap=-1)


# ---------------------------------------------------------------------------
# 再分割後の配置で推論が通る（core 割り当ての欠落が出ない）
# ---------------------------------------------------------------------------

def test_predict_covers_every_char_when_chunks_are_reduced():
    """token 上限で短く切られた配置でも、core 割り当てが全文字を担当して未予測が出ない。"""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from privacy_masker.char_head import CharBioNerModel, predict_char_spans

    text = "医療法人伊織会の理事長である河村敬史" * 12          # 216 文字
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
    for c in sorted(set(text)):
        vocab.setdefault(c, len(vocab))
    raw = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    raw.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
    )
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=1,
        intermediate_size=16, max_position_embeddings=64, pad_token_id=3,
    )
    model = CharBioNerModel(transformers.BertModel(config), char_dim=6, kernel_size=3, dropout=0.0)

    # max_length=18 なので 40 文字のチャンクは入らず、必ず短く切られる
    chunks = token_safe_chunks(text, tok, chunk_chars=40, chunk_overlap=10, max_length=18)
    assert max(len(p) for _, p in chunks) < 40
    _assert_layout_is_usable(text, chunks, tok, 18)

    with torch.no_grad():
        out = predict_char_spans(
            [text], model, tok, chunk_chars=40, chunk_overlap=10, max_length=18, batch_size=4
        )
    assert len(out) == 1          # CharCoverageError にならず 1 文書分返る


# ---------------------------------------------------------------------------
# 実 tokenizer・既定設定（無ければ skip）
# ---------------------------------------------------------------------------

# データフォルダは環境変数 PRIVACY_MASKER_DRIVE_DIR で差し替える（tools/_paths.py と同じ既定値）
_DRIVE_DIR = os.path.expanduser(os.environ.get(
    "PRIVACY_MASKER_DRIVE_DIR",
    "~/privacy-masker-data",
))
_MODEL_DIR = os.path.join(_DRIVE_DIR, "outputs_ruri-v3-30m-v45e", "best")

_SAMPLES = {
    "日本語": "山田太郎さんは東京都新宿区西新宿2-8-1に住んでいます。連絡先は03-1234-5678です。",
    "英語": "Taro Yamada lives at 2-8-1 Nishi-Shinjuku, Shinjuku-ku, Tokyo and his phone is 03-1234-5678. ",
    "絵文字": "🧬🎉👨‍👩‍👧‍👦🇯🇵😀🥰🙏🏻🧑🏽‍💻",
    "タイ語": "นายยามาดะ ทาโร่ อาศัยอยู่ที่กรุงเทพมหานคร ",
    "韓国語": "야마다 타로는 서울특별시 강남구에 살고 있습니다. ",
    "簡体字": "山田太郎住在中华人民共和国北京市朝阳区。",
}


@pytest.mark.skipif(not os.path.isdir(_MODEL_DIR), reason="実 tokenizer が無い環境")
@pytest.mark.parametrize("lang", sorted(_SAMPLES))
@pytest.mark.parametrize("n_chars", [400, 800, 2000])
def test_real_tokenizer_default_settings_split_without_error(lang, n_chars):
    """既定設定 (400/50/512) で 6 言語 × 複数長が例外なく分割され、全文字被覆になる。"""
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL_DIR)
    unit = _SAMPLES[lang]
    text = (unit * (n_chars // len(unit) + 1))[:n_chars]
    chunks = token_safe_chunks(text, tok)
    _assert_layout_is_usable(text, chunks, tok, MAX_LENGTH)

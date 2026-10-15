"""R1 / R2 / R5 / O2 の受入テスト（char_head.py の decode / predict 周り）。

R1: チャンク継ぎ目で正しい B- を無効化していた（2 実体が 1 つに融合した）
R2: 有効文字位置の NaN/Inf を「エンティティなし」として正常返却していた
R5: core 割り当てはエンティティ全体の保持を保証しない（反例を保持する）
O2: argmax と最大確率を batch 単位でまとめた（出力は変えない）

モデルはダウンロードしない。1 文字 1 トークンの tokenizer と、
「そのチャンクから何が見えるか」だけで予測が決まる規則ベースの人工モデルを使う。
人工モデルは入力 input_ids からチャンク本文を復元して文書上の offset を逆引きするので、
「端で切られたチャンクは検出できない」「左文脈が無いと偽 B を出す」を再現できる。
"""

from __future__ import annotations

import random
import warnings

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

try:                                        # リポジトリに統合した後
    import privacy_masker.char_head as ch
except ImportError:                         # 単体の作業ディレクトリで走らせるとき
    import privacy_masker.char_head as ch


# ---------------------------------------------------------------------------
# 人工 tokenizer / モデル
# ---------------------------------------------------------------------------

ALPHA = ("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろ"
         "山田太郎佐藤花子東京都新宿区西新宿医療法人伊織会")


class TinyTok:
    """1 文字 = 1 トークン。offset_mapping も 1 文字ずつ返す。"""

    pad_token_id = 0
    unk_token_id = 1
    is_fast = True

    def __init__(self, alphabet: str):
        self.itos = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted(set(alphabet))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def _id(self, c: str) -> int:
        return self.stoi.get(c, 1)

    def __call__(self, text, **kw):
        ids = [2] + [self._id(c) for c in text] + [3]
        offs = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
        ml = kw.get("max_length")
        if kw.get("truncation") and ml is not None and len(ids) > ml:
            ids = ids[: ml - 1] + [3]
            offs = offs[: ml - 1] + [(0, 0)]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "offset_mapping": offs}
        if not kw.get("return_offsets_mapping", True):
            out.pop("offset_mapping")
        return out

    def convert_tokens_to_ids(self, c):
        return self._id(c)

    def decode_ids(self, ids):
        return "".join(self.itos[i] for i in ids if i > 3)


class RuleModel(nn.Module):
    """gold と「チャンクから何が見えるか」だけで決まる規則ベースの人工モデル。

    ``require_full_visibility=True``
        実体全体がチャンク内に収まっていないと検出しない（端で切られた無知なチャンク
        は黙る。R5 の反例をこれで作る）。
    ``forced_b_context=K``
        チャンク本文の先頭 K 文字は左文脈不足で必ず B- を出す（継ぎ目の偽 B の再現）。
    """

    def __init__(self, doc, tok, gold, *, require_full_visibility=True, forced_b_context=1):
        super().__init__()
        self.doc, self.tok, self.gold = doc, tok, list(gold)
        self.require_full_visibility = require_full_visibility
        self.forced_b_context = int(forced_b_context)

    def eval(self):
        return self

    def __call__(self, **b):
        ids, mask = b["input_ids"], b["char_mask"]
        n_rows, n_cols = mask.shape
        logits = torch.full((n_rows, n_cols, len(ch.LABEL_LIST)), -8.0)
        for r in range(n_rows):
            piece = self.tok.decode_ids([int(v) for v in ids[r].tolist()])
            assert self.doc.count(piece) == 1, "チャンク本文が文書内で一意でない"
            off = self.doc.index(piece)
            nb = int(mask[r].sum())
            names = ["O"] * nb
            for gs, ge, gt in self.gold:
                if self.require_full_visibility and not (off <= gs and ge <= off + nb):
                    continue
                for g in range(max(gs, off), min(ge, off + nb)):
                    li = g - off
                    first = (g == gs) or li < self.forced_b_context
                    names[li] = ("B-" if first else "I-") + gt
            for li, nm in enumerate(names):
                logits[r, li, ch.LABEL_LIST.index(nm)] = 8.0
        return {"logits": logits}


class ConstModel(nn.Module):
    """全位置を同じ値で埋めるモデル。``pad_only`` なら padding だけその値にする。"""

    def __init__(self, value: float, pad_only: bool = False):
        super().__init__()
        self.value, self.pad_only = value, pad_only

    def eval(self):
        return self

    def __call__(self, **b):
        m = b["char_mask"]
        lg = torch.full((m.shape[0], m.shape[1], len(ch.LABEL_LIST)), self.value)
        if self.pad_only:
            lg = torch.full((m.shape[0], m.shape[1], len(ch.LABEL_LIST)), -5.0)
            lg[:, :, 0] = 5.0
            lg[m == 0] = self.value
        return {"logits": lg}


def _doc(n: int, seed: int) -> str:
    """チャンク本文が文書内で一意になるようランダムな平仮名で埋める。"""
    rnd = random.Random(seed)
    a = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろ"
    return "".join(rnd.choice(a) for _ in range(n))


def _place(doc: str, start: int, s: str) -> str:
    return doc[:start] + s + doc[start + len(s):]


def _t(spans):
    return [(s.start, s.end, s.label) for s in spans]


def _ids(*types):
    return [0 if t is None else ch.ENTITY_TYPES.index(t) + 1 for t in types]


# ---------------------------------------------------------------------------
# R1: 継ぎ目の B-
# ---------------------------------------------------------------------------

def test_decode_respects_a_correct_b_at_a_chunk_seam():
    """既定 400/50 の最初の継ぎ目 375 に別人の境界が来ても 2 スパンに分かれる。"""
    n = 479
    ids = [0] * n
    starts = [False] * n
    for i in range(371, 379):
        ids[i] = ch.ENTITY_TYPES.index("PERSON") + 1
    for i in (371, 375):
        starts[i] = True
    seam = [i == 375 for i in range(n)]
    spans = ch.decode_char_labels("あ" * n, ids, [0.9] * n, starts, seam)
    assert [(s.start, s.end) for s in spans] == [(371, 375), (375, 379)]


def test_decode_seam_suppression_is_opt_in():
    """旧挙動（継ぎ目の B- を無視して融合）は明示オプションのときだけ。"""
    ids = _ids(*["PERSON"] * 8)
    starts = [i == 4 for i in range(8)]
    seam = [i == 4 for i in range(8)]
    assert [(s.start, s.end) for s in ch.decode_char_labels(
        "山田太郎佐藤花子", ids, [0.9] * 8, starts, seam, suppress_seam_b=True)] == [(0, 8)]
    # 既定では分かれる
    assert [(s.start, s.end) for s in ch.decode_char_labels(
        "山田太郎佐藤花子", ids, [0.9] * 8, starts, seam)] == [(0, 4), (4, 8)]


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_predict_keeps_two_people_apart_at_the_default_seam(strategy):
    doc = _place(_doc(479, 1), 371, "山田太郎佐藤花子")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(371, 375, "PERSON"), (375, 379, "PERSON")],
                      require_full_visibility=False)
    assert ch.core_ranges(ch.token_safe_chunks(doc, tok), len(doc))[1][0] == 375
    got = _t(ch.predict_char_spans([doc], model, tok, merge_strategy=strategy)[0])
    assert got == [(371, 375, "PERSON"), (375, 379, "PERSON")]


def test_predict_seam_suppression_reproduces_the_old_fusion():
    """R1 の元バグ。明示オプションでしか起きないことを固定する。"""
    doc = _place(_doc(479, 1), 371, "山田太郎佐藤花子")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(371, 375, "PERSON"), (375, 379, "PERSON")],
                      require_full_visibility=False)
    got = _t(ch.predict_char_spans([doc], model, tok,
                                   merge_strategy="core", suppress_seam_b=True)[0])
    assert got == [(371, 379, "PERSON")]


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_a_span_crossing_the_seam_is_not_split_without_a_b(strategy):
    doc = _place(_doc(479, 3), 370, "東京都新宿区西新宿医")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(370, 380, "ADDRESS")], require_full_visibility=False)
    assert _t(ch.predict_char_spans([doc], model, tok, merge_strategy=strategy)[0]) == [
        (370, 380, "ADDRESS")
    ]


# ---------------------------------------------------------------------------
# R2: 数値異常
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_nonfinite_logits_are_rejected_instead_of_empty_entities(value, strategy):
    tok = TinyTok(ALPHA)
    with pytest.raises(FloatingPointError):
        ch.predict_char_spans(["山田太郎"], ConstModel(value), tok, merge_strategy=strategy)


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_all_o_with_finite_logits_still_returns_no_spans(strategy):
    """全 O は正常な「検出なし」。数値異常と区別する。"""
    tok = TinyTok(ALPHA)
    model = ConstModel(0.0, pad_only=True)      # 有効位置は全 O、padding も有限
    out = ch.predict_char_spans(["山田太郎", _doc(300, 9)], model, tok, merge_strategy=strategy)
    assert out == [[], []]


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_nonfinite_padding_positions_are_not_checked(strategy):
    """padding 位置の NaN は検査対象にしない（長さ違いを 1 バッチに入れて確認）。"""
    tok = TinyTok(ALPHA)
    model = ConstModel(float("nan"), pad_only=True)
    out = ch.predict_char_spans(["山田太郎", _doc(300, 9)], model, tok, merge_strategy=strategy)
    assert out == [[], []]


# ---------------------------------------------------------------------------
# R5: core 割り当てはエンティティ全体を保証しない
# ---------------------------------------------------------------------------

def test_core_ranges_counterexample_is_kept():
    """反例の core 配置そのもの。gold [360,410) の先頭 15 文字は A の担当になる。"""
    cores = ch.core_ranges([(0, "x" * 400), (350, "x" * 110)], 460)
    assert cores == [(0, 375), (375, 460)]
    assert cores[0][1] == 375 > 360      # gold の開始 360 は chunk A の担当区間の中


def test_core_loses_the_head_of_a_span_but_safe_union_keeps_it():
    """R5 の反例。chunk A は 400 で切れて gold 全体が見えず検出しない。

    chunk A=[0,400) core=[0,375) / chunk B=[350,460) core=[375,460) / gold=[360,410)
    """
    doc = _doc(460, 5)
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(360, 410, "ADDRESS")], require_full_visibility=True)
    assert [(o, o + len(p)) for o, p in ch.token_safe_chunks(doc, tok)] == [(0, 400), (350, 460)]
    assert _t(ch.predict_char_spans([doc], model, tok, merge_strategy="core")[0]) == [
        (375, 410, "ADDRESS")          # 先頭 15 文字が未マスク = 漏れ
    ]
    assert _t(ch.predict_char_spans([doc], model, tok, merge_strategy="safe_union")[0]) == [
        (360, 410, "ADDRESS")          # 全体が見えた chunk B の判断を採用する
    ]


def test_safe_union_recovers_an_entity_that_no_chunk_sees_whole():
    """全チャンクの端に掛かる超長エンティティ。フォールバックが働き、報告も残る。"""
    doc = _doc(1200, 31)
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(100, 1100, "ADDRESS")], require_full_visibility=False)
    assert len(ch.token_safe_chunks(doc, tok)) == 4
    fb: list = []
    # 警告は既定では出さない（ランダム重みで 4% の文書が発火しログを埋めるため）。
    # 明示的に warn_fallback=True を指定したときだけ出る。
    with pytest.warns(UserWarning, match="core 割り当てだけ"):
        got = ch.predict_char_spans([doc], model, tok, merge_strategy="safe_union",
                                    fallback_out=fb, warn_fallback=True)[0]
    assert _t(got) == [(100, 1100, "ADDRESS")]
    assert [(d, s.start, s.end) for d, s in fb] == [(0, 100, 1100)]


def test_no_fallback_warning_for_an_ordinary_entity():
    """普通に 1 チャンク内で見えている実体では警告を出さない（警告を意味のあるものに保つ）。"""
    doc = _place(_doc(1200, 32), 500, "医療法人伊織会")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(500, 507, "ORG")], require_full_visibility=True)
    fb: list = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = ch.predict_char_spans([doc], model, tok, merge_strategy="safe_union",
                                    fallback_out=fb)[0]
    assert _t(got) == [(500, 507, "ORG")]
    assert fb == [] and [w for w in rec if issubclass(w.category, UserWarning)] == []


def test_safe_union_does_not_adopt_a_false_positive_at_a_truncated_edge():
    """端で切られた側の判断は採らない。別チャンクが全体を見て O と言うならそれに従う。"""
    doc = _doc(460, 33)
    tok = TinyTok(ALPHA + doc)

    class EdgeFP(nn.Module):
        """chunk0 だけが自分の右端ぎりぎり [397,400) を PERSON と誤検出する。"""

        def eval(self):
            return self

        def __call__(self, **b):
            ids, m = b["input_ids"], b["char_mask"]
            lg = torch.full((m.shape[0], m.shape[1], len(ch.LABEL_LIST)), -8.0)
            lg[:, :, 0] = 8.0
            for r in range(m.shape[0]):
                piece = tok.decode_ids([int(v) for v in ids[r].tolist()])
                if doc.index(piece) == 0:
                    for k in range(397, 400):
                        lg[r, k, 0] = -8.0
                        lg[r, k, ch.LABEL_LIST.index("B-PERSON" if k == 397 else "I-PERSON")] = 8.0
            return {"logits": lg}

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # 警告が出たら失敗させる
        assert ch.predict_char_spans([doc], EdgeFP(), tok, merge_strategy="safe_union")[0] == []


# ---------------------------------------------------------------------------
# 共通: 戦略の指定、merge_adjacent、被覆
# ---------------------------------------------------------------------------

def test_unknown_merge_strategy_is_rejected():
    with pytest.raises(ValueError, match="merge_strategy"):
        ch.predict_char_spans(["あ"], ConstModel(0.0, pad_only=True), TinyTok(ALPHA),
                              merge_strategy="voting")


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_merge_adjacent_covers_the_same_characters_in_both_strategies(strategy):
    doc = _place(_doc(479, 11), 371, "山田太郎佐藤花子")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(371, 375, "PERSON"), (375, 379, "PERSON")],
                      require_full_visibility=False)
    plain = _t(ch.predict_char_spans([doc], model, tok, merge_strategy=strategy)[0])
    merged = _t(ch.predict_char_spans([doc], model, tok, merge_strategy=strategy,
                                      merge_adjacent=True)[0])
    assert plain == [(371, 375, "PERSON"), (375, 379, "PERSON")]
    assert merged == [(371, 379, "PERSON")]
    assert {i for s, e, _ in plain for i in range(s, e)} == {
        i for s, e, _ in merged for i in range(s, e)
    }


@pytest.mark.parametrize("strategy", ["safe_union", "core"])
def test_long_document_is_fully_covered_and_fail_closed(strategy):
    doc = _place(_doc(3000, 7), 1180, "医療法人伊織会")
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [(1180, 1187, "ORG")], require_full_visibility=True)
    assert len(ch.token_safe_chunks(doc, tok)) > 1
    assert _t(ch.predict_char_spans([doc], model, tok, merge_strategy=strategy)[0]) == [
        (1180, 1187, "ORG")
    ]


def test_predict_is_fail_closed_when_a_character_is_left_unpredicted(monkeypatch):
    """統合の不整合で 1 文字でも予測が入らなければ黙って O にせず落とす。"""
    doc = _doc(1000, 13)
    tok = TinyTok(ALPHA + doc)
    model = RuleModel(doc, tok, [], require_full_visibility=True)
    real = ch.core_ranges

    def broken(chunks, n_chars):
        got = real(chunks, n_chars)
        return got[:-1] + [(got[-1][0], n_chars - 5)]      # 末尾 5 文字を誰も担当しない

    monkeypatch.setattr(ch, "core_ranges", broken)
    with pytest.raises(ch.CharCoverageError, match="未予測"):
        ch.predict_char_spans([doc], model, tok, merge_strategy="core")


# ---------------------------------------------------------------------------
# O2: batch 一括の argmax が 1 文字ずつの argmax と同じ結果になる
# ---------------------------------------------------------------------------

def test_batched_argmax_matches_per_character_argmax():
    """実 encoder（ランダム重み）の logits から素朴に組んだスパンと一致すること。

    単一チャンクに収まる長さで比べる（チャンク統合の影響を混ぜない）。
    """
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    tok = TinyTok(ALPHA)
    cfg = transformers.BertConfig(
        vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=512,
        pad_token_id=0,
    )
    model = ch.CharBioNerModel(transformers.BertModel(cfg), char_dim=8, kernel_size=3,
                               dropout=0.0).eval()
    doc = _doc(120, 41)
    for threshold in (0.0, 0.4):
        feat = ch.encode_chunk(doc, tok, None, 512, {}, strict=True)
        batch = ch.CharNerCollator(tok.pad_token_id)([feat])
        with torch.no_grad():
            probs = torch.softmax(model(**batch)["logits"].float(), dim=-1)[0]
        type_ids, conf, is_b = [], [], []
        for li in range(len(doc)):
            lab = int(probs[li].argmax())            # 旧実装と同じ 1 文字ずつの argmax
            sc = float(probs[li, lab])
            name = ch.LABEL_LIST[lab]
            hit = lab != 0 and sc >= threshold
            type_ids.append(ch.ENTITY_TYPES.index(name[2:]) + 1 if hit else 0)
            is_b.append(hit and name.startswith("B-"))
            conf.append(sc)
        expect = ch.decode_char_labels(doc, type_ids, conf, is_b)
        for strategy in ("safe_union", "core"):
            got = ch.predict_char_spans([doc], model, tok, threshold=threshold,
                                        merge_strategy=strategy)[0]
            assert [(s.start, s.end, s.label, round(s.score, 10)) for s in got] == [
                (s.start, s.end, s.label, round(s.score, 10)) for s in expect
            ]


# ---------------------------------------------------------------------------
# 第2次レビュー: 型競合の解消が学習ラベル側と同じ優先順位を使うこと
# ---------------------------------------------------------------------------

def test_type_conflict_keeps_the_more_sensitive_span_whole():
    """スコアが低くても機密な型を丸ごと残す。

    以前はスコアだけで勝者を決めていたため、ORG[355,395) が score 0.90、
    PERSON[360,390) が score 0.60 のとき PERSON が完全に内包されて消えていた。
    「どちらの型でもマスクは掛かる」は ``Policy.preset("minimal"/"standard")`` では
    成立せず、``ORG_PUBLIC`` は Keep（マスクしない）になるので人名が素通しになる。
    """
    got = ch._resolve_type_conflicts(
        [ch.CharSpan(355, 395, "ORG", 0.90), ch.CharSpan(360, 390, "PERSON", 0.60)]
    )
    assert [(s.start, s.end, s.label) for s in got] == [
        (355, 360, "ORG"),
        (360, 390, "PERSON"),
        (390, 395, "ORG"),
    ]
    # 文字の被覆は失われない
    covered = {i for s in got for i in range(s.start, s.end)}
    assert covered == set(range(355, 395))


def test_type_conflict_matches_the_training_label_policy():
    """推論側の型優先順位が flatten_entities（学習ラベル側）と一致していること。"""
    ents = [{"start": 5, "end": 20, "label": "ORG"}, {"start": 8, "end": 15, "label": "PERSON"}]
    train_label = ch.flatten_entities("あ" * 30, ents)[0].label
    infer = ch._resolve_type_conflicts(
        [ch.CharSpan(5, 20, "ORG", 0.99), ch.CharSpan(8, 15, "PERSON", 0.10)]
    )
    assert train_label == "PERSON"
    assert "PERSON" in [s.label for s in infer]


def test_address_absorbing_person_is_still_masked_everywhere():
    """ADDRESS は Keep になり得ないので、PERSON を吸収しても未マスク文字は出ない。"""
    got = ch._resolve_type_conflicts(
        [ch.CharSpan(100, 200, "ADDRESS", 0.5), ch.CharSpan(120, 140, "PERSON", 0.9)]
    )
    assert [(s.start, s.end, s.label) for s in got] == [(100, 200, "ADDRESS")]
    covered = {i for s in got for i in range(s.start, s.end)}
    assert covered == set(range(100, 200))


def test_same_risk_types_still_fall_back_to_score():
    """同じ Keep 危険度の型どうし（ORG と FACILITY）はスコアで決める。"""
    got = ch._resolve_type_conflicts(
        [ch.CharSpan(100, 200, "ORG", 0.5), ch.CharSpan(120, 140, "FACILITY", 0.9)]
    )
    assert [(s.start, s.end, s.label) for s in got] == [
        (100, 120, "ORG"),
        (120, 140, "FACILITY"),
        (140, 200, "ORG"),
    ]

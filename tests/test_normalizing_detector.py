"""字間空白の前処理（strip_letter_spacing / NormalizingDetector）。

字間空白「千 葉 県 市 川 市 南 八 幡 2 - 1 - 9」は token・文字ヘッドの両モデルで
gold 被覆が 0.43〜0.50 に落ちていた。正規化して推論し元位置に戻すと 1.00 になる。
ここでは (1) 消してよい空白だけを消していること、(2) 逆写像が連続区間であること、
(3) 壊れた back_map が黙って通らないことを固定する。
"""
from __future__ import annotations

import random

import pytest

from privacy_masker.detectors import NormalizingDetector
from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError, DetectorError
from privacy_masker.normalize import strip_letter_spacing
from privacy_masker.types import Entity

SPACED_ADDRESS = "千 葉 県 市 川 市 南 八 幡 2 - 1 - 9"
DOC = f"住 所  {SPACED_ADDRESS}"


class NeedleDetector:
    """渡されたテキストから needle を全部拾う。どのテキストで呼ばれたかを記録する。"""

    def __init__(self, needles, name="needle", batch=True):
        self.needles = list(needles)
        self.name = name
        self.seen: list[str] = []
        if not batch:
            self.detect_batch = None  # detect_batch を持たない検出器の代わり

    def detect(self, text):
        self.seen.append(text)
        out = []
        for nd in self.needles:
            i = text.find(nd)
            while i >= 0:
                out.append(Entity("ADDRESS", i, i + len(nd), 0.9, self.name))
                i = text.find(nd, i + 1)
        return out

    def detect_batch(self, texts):
        return [self.detect(t) for t in texts]


# ---- strip_letter_spacing ----------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "金 東賢が来日した。",                            # 姓名間の単発の空白
        "Kevin Marsh and I will arrive at Nagoya plant",  # 欧文の語間
        "担当は Sato さんと Kevin Marsh です。",
        "・東京都新宿区西新宿2-8-1",
        "金 額  1 2 3 4 5 6 円",                          # 和文が min_run 未満
        "医療法人伊織会の理事長である河村敬史",
        "　伊集院　　　薫　　　　営業部",                  # 表組みの桁揃え（空白2個以上）
        "・ あ ・ い ・ う ・ え",                          # 記号は連鎖の手がかりにしない
        "",
        "   ",
        "　　　　　　",
        "都",
    ],
)
def test_untouched(text):
    out, back = strip_letter_spacing(text)
    assert out == text
    assert list(back) == list(range(len(text)))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (DOC, "住 所  千葉県市川市南八幡2-1-9"),
        ("り ょ う け さ ん ち ょ う め", "りょうけさんちょうめ"),
        ("東 京 都 港 区 芝 3 - 1 - 1 with Kevin Marsh", "東京都港区芝3-1-1 with Kevin Marsh"),
    ],
)
def test_stripped(text, expected):
    assert strip_letter_spacing(text)[0] == expected


def test_min_run_must_be_positive():
    with pytest.raises(ValueError):
        strip_letter_spacing(DOC, min_run=0)


def _assert_invariants(text):
    out, back = strip_letter_spacing(text)
    assert len(back) == len(out)
    prev = -1
    for i, o in enumerate(back):
        assert isinstance(o, int)
        assert prev < o < len(text)          # 狭義単調増加かつ範囲内
        assert text[o] == out[i]
        prev = o
    dropped = {text[k] for k in set(range(len(text))) - set(back)}
    assert not (dropped - set(" \u3000\u00a0")), "空白以外を削除している"


def test_back_map_invariants_random():
    """和文・欧文・数字・空白・改行・絵文字の混在でも back_map が壊れない。"""
    alphabet = list("あいうえおアイウエオ亜伊宇江尾一二三ABCdefg0123456789-−/、。・")
    alphabet += [" ", " ", " ", "\u3000", "\u00a0", "\n", "\t", "\U0001f600", "ｱ", "Ａ", "１"]
    rng = random.Random(20260909)
    changed = 0
    for _ in range(200):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        if rng.random() < 0.5:  # 純ランダムだと 4 連鎖がまれなので字間空白を混ぜる
            seg = " ".join(rng.choice("あア亜一Ａ１ｱ") for _ in range(rng.randint(2, 12)))
            pos = rng.randint(0, len(t))
            t = t[:pos] + seg + t[pos:]
        _assert_invariants(t)
        changed += strip_letter_spacing(t)[0] != t
    assert changed > 0, "1 本も正規化されていないなら検査になっていない"


def test_long_text():
    text = ((SPACED_ADDRESS + " " + "これは普通の日本語の文章です。" * 20) * 40)[:10000]
    _assert_invariants(text)


# ---- NormalizingDetector -----------------------------------------------------

def test_span_is_mapped_back_as_one_interval():
    """文字集合ではなく連続区間で戻す（間の空白がマスクから抜けないこと）。"""
    det = NormalizingDetector(NeedleDetector(["千葉県市川市南八幡2-1-9"]))
    (ent,) = det.detect(DOC)
    assert DOC[ent.start:ent.end] == SPACED_ADDRESS
    assert det.name == "needle" and ent.source == "needle"


def test_unchanged_text_bypasses_the_normalizer():
    texts = ["金 東賢が来日した。", "山田太郎は東京都千代田区に住む。", "Kevin Marsh"]
    needles = ["東京都", "金 東賢", "Kevin Marsh"]
    inner = NeedleDetector(needles)
    wrapped = NormalizingDetector(NeedleDetector(needles))
    assert [inner.detect(t) for t in texts] == [wrapped.detect(t) for t in texts]
    assert inner.detect_batch(texts) == wrapped.detect_batch(texts)
    assert wrapped.inner.seen[: len(texts)] == texts  # 元テキストがそのまま渡っている


def test_union_contains_the_baseline_result():
    """正規化後にしか当たらない span と元テキストにしか当たらない span の両方を返す。"""
    needles = ["千 葉 県", "千葉県市川市南八幡2-1-9"]
    baseline = {(e.start, e.end) for e in NeedleDetector(needles).detect(DOC)}
    union = {(e.start, e.end) for e in NormalizingDetector(NeedleDetector(needles)).detect(DOC)}
    replaced = {(e.start, e.end) for e in
                NormalizingDetector(NeedleDetector(needles), mode="replace").detect(DOC)}
    assert baseline <= union
    assert baseline - replaced, "replace は元テキスト側の結果を落とす（union が必要な理由）"


def test_union_batch_matches_single():
    needles = ["千 葉 県", "千葉県市川市南八幡2-1-9"]
    det = NormalizingDetector(NeedleDetector(needles))
    key = lambda es: sorted((e.entity_type, e.start, e.end) for e in es)  # noqa: E731
    batched = det.detect_batch([DOC, "金 東賢", DOC])
    assert key(batched[0]) == key(batched[2]) == key(det.detect(DOC))
    assert batched[1] == []


def test_inner_without_detect_batch():
    det = NormalizingDetector(NeedleDetector(["千葉県市川市南八幡2-1-9"], batch=False))
    got = det.detect_batch([DOC, "金 東賢"])
    assert [(e.start, e.end) for e in got[0]] == [(DOC.index("千"), len(DOC))]
    assert got[1] == []


@pytest.mark.parametrize("kind", ["short", "long", "nonmono", "dup", "oob", "negative",
                                  "mismatch", "notint", "notseq"])
def test_broken_back_map_is_fail_closed(kind):
    """壊れた back_map はスパンを別の位置に移すので、黙って通さず落とす。"""

    def normalizer(text):
        out, back = strip_letter_spacing(text)
        back = list(back)
        if kind == "short":
            back = back[:-1]
        elif kind == "long":
            back = back + [len(text) - 1]
        elif kind == "nonmono":
            back[1], back[2] = back[2], back[1]
        elif kind == "dup":
            back[2] = back[1]
        elif kind == "oob":
            back[-1] = len(text) + 5
        elif kind == "negative":
            back[0] = -1
        elif kind == "mismatch":
            back[0] = back[-1]
        elif kind == "notint":
            back[0] = 0.0
        elif kind == "notseq":
            back = 123
        return out, back

    det = NormalizingDetector(NeedleDetector(["千葉県"]), normalizer=normalizer)
    with pytest.raises(DetectorError) as exc:
        det.detect(DOC)
    assert "千" not in str(exc.value)  # 例外メッセージに原文を載せない


def test_inner_entity_out_of_range_is_fail_closed():
    class OutOfRange:
        name = "oor"

        def detect(self, text):
            return [Entity("PERSON", 0, len(text) + 3, 0.9, self.name)]

    with pytest.raises(DetectorError):
        NormalizingDetector(OutOfRange()).detect(DOC)


def test_batch_length_mismatch_is_fail_closed():
    class BadCount:
        name = "badcount"

        def detect(self, text):
            return []

        def detect_batch(self, texts):
            return [[]]

    with pytest.raises(DetectorError):
        NormalizingDetector(BadCount()).detect_batch([DOC, DOC])


@pytest.mark.parametrize(("inner", "kwargs"), [
    (None, {}),
    (object(), {}),
    (NeedleDetector([]), {"normalizer": 3}),
    (NeedleDetector([]), {"mode": "both"}),
])
def test_configuration_errors(inner, kwargs):
    with pytest.raises(ConfigurationError):
        NormalizingDetector(inner, **kwargs)


@pytest.mark.parametrize("text", ["", "   ", "　　", "都", "\n\n"])
def test_edge_texts(text):
    det = NormalizingDetector(NeedleDetector(["千葉県市川市南八幡2-1-9"]))
    assert det.detect(text) == []
    assert det.detect_batch([text]) == [[]]


def test_engine_end_to_end():
    engine = PrivacyEngine([NormalizingDetector(NeedleDetector(["千葉県市川市南八幡2-1-9"]))])
    res = engine.transform(DOC)
    assert "千 葉 県" not in res.text and "南 八 幡" not in res.text
    assert res.failed_detectors == ()
    (te,) = res.entities
    assert (te.original_start, te.original_end) == (DOC.index("千"), len(DOC))


def test_engine_fails_closed_on_broken_normalizer():
    def normalizer(text):
        out, back = strip_letter_spacing(text)
        back = list(back)
        back[-1] = len(text) + 5
        return out, back

    engine = PrivacyEngine([NormalizingDetector(NeedleDetector(["千葉県"]), normalizer=normalizer)])
    with pytest.raises(DetectorError):
        engine.transform(DOC)


# ---------------------------------------------------------------------------
# 検証者の指摘への追加テスト（親が追加）
# ---------------------------------------------------------------------------

def test_union_keeps_the_higher_score():
    """同一スパンが両方から出たらスコアの高い方を残す。

    正規化側を無条件に採ると前処理なしより score が下がり（実測で最大 0.36 低下）、
    ``Engine`` の ``score_threshold`` を上げた構成ではそれだけで検出が消える＝漏れになる。
    """
    from privacy_masker.detectors import NormalizingDetector
    from privacy_masker.types import Entity

    class _Stub:
        name = "stub"

        def __init__(self):
            self.calls = []

        def detect(self, text):
            self.calls.append(text)
            # 全文を 1 スパンとして返す。正規化後のスパンを元位置へ戻すと
            # 元テキスト側のスパンと厳密に一致するので、merge の対象になる。
            # 元テキスト（空白あり）では高スコア、正規化後は低スコアにする。
            score = 0.9 if " " in text else 0.4
            return [Entity("PERSON", 0, len(text), score, self.name)]

    inner = _Stub()
    det = NormalizingDetector(inner)
    text = "あ い う え お か"          # 字間空白あり → union で 2 回呼ばれる
    got = det.detect(text)
    assert len(inner.calls) == 2, inner.calls
    assert len(got) == 1, [(e.start, e.end, e.score) for e in got]
    assert (got[0].start, got[0].end) == (0, len(text))
    assert got[0].score == 0.9, f"低い方が残っている: {got[0].score}"


def test_letter_spaced_digits_are_normalized():
    """字間空白の電話番号・カード番号・マイナンバーは、そのままでは平文で漏れる。

    実測: 正規化前は ``RegexDetector.default_ja()`` が 1 件も検出しない。
    """
    from privacy_masker.detectors import RegexDetector
    from privacy_masker.normalize import strip_letter_spacing

    rx = RegexDetector.default_ja()
    for raw, expect in (
        ("連絡先は 0 9 0 - 1 2 3 4 - 5 6 7 8 です。", "PHONE_NUMBER"),
        ("カード番号 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1", "CREDIT_CARD"),
        ("個人番号 1 2 3 4 5 6 7 8 9 0 1 8", "MY_NUMBER"),
    ):
        assert rx.detect(raw) == [], f"前処理なしで検出できてしまう: {raw}"
        norm, back = strip_letter_spacing(raw)
        assert norm != raw
        found = [e.entity_type for e in rx.detect(norm)]
        assert expect in found, f"{raw} → {norm} で {expect} が出ない: {found}"


def test_short_digit_runs_are_left_alone():
    """短い数値列は巻き込まない（金額・年号）。"""
    from privacy_masker.normalize import strip_letter_spacing

    for t in ("金 額  1 2 3 4 5 6 円", "平 成 3 1 年", "第 1 2 条"):
        assert strip_letter_spacing(t)[0] == t, t


def test_min_run_and_min_digit_run_have_floors():
    """下限を下回る設定は拒否する（min_run=1 は姓名間の空白まで潰す）。"""
    from privacy_masker.normalize import strip_letter_spacing

    for kw in ({"min_run": 1}, {"min_run": 2}, {"min_digit_run": 6}):
        with pytest.raises(ValueError):
            strip_letter_spacing("あ い う え お", **kw)
    assert strip_letter_spacing("金 東賢が来日した。")[0] == "金 東賢が来日した。"

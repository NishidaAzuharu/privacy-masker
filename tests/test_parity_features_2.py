"""倣った機能の第 2 弾: Mask の別名引数、engine.explain()、restore_batch()、DateShift / Bucketize / FormatPreserving。"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.crypto import FernetProvider
from privacy_masker.detectors import RegexDetector
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Bucketize, DateShift, FormatPreserving, Keep, Mask, Tokenize
from privacy_masker.stores import MemoryStore
from privacy_masker.types import Entity
from tests.conftest import FakeDetector


def E(t, text, surface, score=0.9, source="fake"):
    s = text.index(surface); return Entity(t, s, s + len(surface), score, source)


def test_mask_alias_arguments():
    e = Entity("PHONE_NUMBER", 0, 12, 1.0)
    assert Mask(chars_to_mask=4, masking_char="#", from_end=True).apply("03-1234-5678", e, None) == "03-1234-####"
    assert Mask(chars_to_mask=4).apply("03-1234-5678", e, None) == "****234-5678"
    assert Mask(keep_suffix=4).apply("03-1234-5678", e, None) == "********5678"      # 従来の引数はそのまま
    assert Mask(chars_to_mask=0).apply("abc", e, None) == "***"                       # 素通しは全マスクに倒す
    assert Mask(chars_to_mask=99).apply("abc", e, None) == "***"


def test_explain_returns_entities_with_trace_without_original_text():
    text = "田中太郎（03-1234-5678）"
    ents = [E("PERSON", text, "田中太郎", score=0.9, source="ner")]
    eng = PrivacyEngine(detectors=[FakeDetector(ents), RegexDetector.default_ja()], policy=Policy.preset("strict"), refiner=None)
    out = eng.explain(text)
    assert [x.entity.entity_type for x in out] == ["PERSON", "PHONE_NUMBER"]
    assert "detector=ner(0.90)" in out[0].explanation and "regex" in out[1].explanation
    assert "田中" not in out[0].explanation and "1234" not in out[1].explanation
    assert [x.entity for x in out] == eng.detect(text)                                  # detect() と同じ結果


def test_restore_batch_roundtrip():
    pytest.importorskip("cryptography")
    text1, text2 = "担当は田中太郎。", "田中太郎と山田花子が出席。"
    ents1 = [E("PERSON", text1, "田中太郎")]; ents2 = [E("PERSON", text2, "田中太郎"), E("PERSON", text2, "山田花子")]
    class Det:
        name = "fake"
        def detect(self, t): return ents1 if t == text1 else ents2
    eng = PrivacyEngine(detectors=[Det()], policy=Policy(default=Tokenize(strategy="vault")), refiner=None,
                        secret_key=b"k" * 16, crypto=FernetProvider(FernetProvider.generate_key()), store=MemoryStore())
    outs = eng.transform_batch([text1, text2], scope_id="s1")
    assert "田中太郎" not in outs[0].text
    back = eng.restore_batch([o.text for o in outs], scope_id="s1")
    assert [b.text for b in back] == [text1, text2] and all(b.unresolved == () for b in back)


def test_date_shift_preserves_intervals_and_formats():
    text = "生年月日 1990年3月3日、入社 2015/04/01、退職 令和7年9月1日、不明 3月3日"
    ents = [E("DATE", text, "1990年3月3日"), E("DATE", text, "2015/04/01"), E("DATE", text, "令和7年9月1日"),
            Entity("DATE", text.rindex("3月3日"), text.rindex("3月3日") + 4, 0.9, "fake")]
    eng = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(rules={"DATE": DateShift(days=10)}, default=Keep()), refiner=None)
    r = eng.transform(text)
    assert r.text == "生年月日 1990年3月13日、入社 2015/04/11、退職 令和7年9月11日、不明 <DATE>"
    # 鍵から決まるシフト: 同じ鍵なら同じ日数、2 つの日付の間隔が保たれる
    eng2 = PrivacyEngine(detectors=[FakeDetector(ents[:2])], policy=Policy(rules={"DATE": DateShift(max_days=100)}, default=Keep()),
                         refiner=None, secret_key=b"secret-key-0001")
    import datetime as dt, re
    out = eng2.transform(text).text
    d1 = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", out); d2 = re.search(r"(\d{4})/(\d{2})/(\d{2})", out)
    a = dt.date(*map(int, d1.groups())); b = dt.date(*map(int, d2.groups()))
    assert (b - a) == (dt.date(2015, 4, 1) - dt.date(1990, 3, 3))
    assert eng2.transform(text).text == out                                             # 決定的
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector(ents[:1])], policy=Policy(rules={"DATE": DateShift()}, default=Keep()), refiner=None).transform(text)


def test_bucketize_generalizes_numbers_and_keeps_units():
    e = Entity("AGE", 0, 3, 1.0)
    assert Bucketize(10).apply("34歳", e, None) == "30-39歳"
    assert Bucketize(10, template="{lo}代").apply("満34歳", e, None) == "満30代歳"
    assert Bucketize(1_000_000).apply("年収 4,560,000円", e, None) == "年収 4000000-4999999円"
    assert Bucketize(10).apply("不明", e, None) == "<NUMBER>"
    with pytest.raises(ConfigurationError):
        Bucketize(0)


def test_format_preserving_keeps_shape_and_is_deterministic():
    text = "TEL 03-1234-5678 / 旅券 TK1234567 / 全角 ０９０"
    ents = [E("PHONE_NUMBER", text, "03-1234-5678"), E("PASSPORT", text, "TK1234567"), E("PHONE_NUMBER", text, "０９０")]
    pol = Policy(default=FormatPreserving(key=b"k1"))
    r = PrivacyEngine(detectors=[FakeDetector(ents)], policy=pol, refiner=None).transform(text)
    import re
    assert re.fullmatch(r"TEL \d{2}-\d{4}-\d{4} / 旅券 [A-Z]{2}\d{7} / 全角 [０-９]{3}", r.text)
    assert "1234-5678" not in r.text
    assert PrivacyEngine(detectors=[FakeDetector(ents)], policy=pol, refiner=None).transform(text).text == r.text
    other = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(default=FormatPreserving(key=b"k2")), refiner=None).transform(text).text
    assert other != r.text
    # 鍵を省くと engine の secret_key から派生、無ければエラー
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(default=FormatPreserving()), refiner=None).transform(text)
    ok = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(default=FormatPreserving()), refiner=None, secret_key=b"s" * 16).transform(text)
    assert re.fullmatch(r"TEL \d{2}-\d{4}-\d{4} / 旅券 [A-Z]{2}\d{7} / 全角 [０-９]{3}", ok.text)

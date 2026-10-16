"""他ライブラリに倣った機能: Hash・Encrypt・Custom オペレータ、呼び出し時オプション
（entities / score_threshold / allow_list / deny_list）、explain。"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.detectors import RegexDetector
from privacy_masker.exceptions import ConfigurationError, RestoreError
from privacy_masker.operators import Custom, Encrypt, Hash, Keep, Tokenize, decrypt_text
from privacy_masker.types import Entity
from tests.conftest import FakeDetector


def E(t, text, surface, score=0.9, source="fake"):
    s = text.index(surface)
    return Entity(t, s, s + len(surface), score, source)


TEXT = "田中太郎は株式会社アオバに勤務。連絡は 03-1234-5678 まで。"


def engine_for(text, ents, policy=None, **kw):
    return PrivacyEngine(detectors=[FakeDetector(ents), RegexDetector.default_ja()],
                         policy=policy or Policy.preset("strict"), refiner=None, **kw)


# ---------------------------------------------------------------- operators

def test_hash_is_deterministic_and_salted():
    ents = [E("PERSON", TEXT, "田中太郎")]
    pol = Policy(rules={"PERSON": Hash(length=12)}, default=Keep())
    r1 = engine_for(TEXT, ents, pol).transform(TEXT)
    r2 = engine_for(TEXT, ents, pol).transform(TEXT)
    h = r1.entities[0].replacement
    assert r1.text == r2.text and len(h) == 12 and all(c in "0123456789abcdef" for c in h)
    salted = Policy(rules={"PERSON": Hash(length=12, salt="s1")}, default=Keep())
    assert engine_for(TEXT, ents, salted).transform(TEXT).entities[0].replacement != h
    with pytest.raises(ConfigurationError):
        Hash(algorithm="crc32")
    assert Hash(template="<{type}:{hash}>", length=8).apply("x", Entity("PERSON", 0, 1, 1.0), None).startswith("<PERSON:")


def test_encrypt_roundtrip_via_decrypt_text():
    pytest.importorskip("cryptography")
    key = Encrypt.generate_key()
    ents = [E("PERSON", TEXT, "田中太郎"), E("ORG", TEXT, "株式会社アオバ")]
    pol = Policy(rules={"PERSON": Encrypt(key), "ORG": Encrypt(key)}, default=Tokenize())
    out = engine_for(TEXT, ents, pol).transform(TEXT)
    assert "田中太郎" not in out.text and "gAAAAA" in out.text
    back = decrypt_text(out.text, key)
    assert "田中太郎は株式会社アオバに勤務。連絡は <PHONE_NUMBER_001> まで。" == back
    other = Encrypt.generate_key()
    assert decrypt_text(out.text, other) == out.text          # 別の鍵では復号せずそのまま
    with pytest.raises(RestoreError):
        decrypt_text(out.text, other, strict=True)
    with pytest.raises(ConfigurationError):
        Encrypt(b"not-a-valid-key")


def test_custom_operator_receives_value_and_entity():
    seen = []
    def fn(value, entity):
        seen.append((value, entity.entity_type)); return f"[{entity.entity_type}:{len(value)}]"
    ents = [E("PERSON", TEXT, "田中太郎")]
    out = engine_for(TEXT, ents, Policy(rules={"PERSON": Custom(fn)}, default=Keep())).transform(TEXT)
    assert out.text.startswith("[PERSON:4]は") and seen == [("田中太郎", "PERSON")]
    bad = Policy(rules={"PERSON": Custom(lambda v, e: 42)}, default=Keep())
    with pytest.raises(ConfigurationError):
        engine_for(TEXT, ents, bad).transform(TEXT)


# ---------------------------------------------------------------- call-time options

def test_entities_filter_limits_types():
    ents = [E("PERSON", TEXT, "田中太郎"), E("ORG", TEXT, "株式会社アオバ")]
    eng = engine_for(TEXT, ents)
    r = eng.transform(TEXT, entities=["PERSON"])
    assert [e.entity_type for e in r.entities] == ["PERSON"]          # ORG も PHONE_NUMBER も対象外
    r = eng.transform(TEXT, entities=["PERSON", "PHONE_NUMBER"])
    assert [e.entity_type for e in r.entities] == ["PERSON", "PHONE_NUMBER"]


def test_score_threshold_float_and_per_type_overrides():
    ents = [E("PERSON", TEXT, "田中太郎", score=0.55), E("ORG", TEXT, "株式会社アオバ", score=0.95)]
    eng = engine_for(TEXT, ents)
    assert [e.entity_type for e in eng.transform(TEXT, score_threshold=0.6).entities] == ["ORG", "PHONE_NUMBER"]
    r = eng.transform(TEXT, score_threshold={"DEFAULT": 0.9, "PERSON": 0.5})
    assert [e.entity_type for e in r.entities] == ["PERSON", "ORG", "PHONE_NUMBER"]
    # engine 既定を呼び出しで上書き
    eng2 = engine_for(TEXT, ents, score_threshold=0.9)
    assert [e.entity_type for e in eng2.transform(TEXT).entities] == ["ORG", "PHONE_NUMBER"]
    assert len(eng2.transform(TEXT, score_threshold=0.0).entities) == 3


def test_allow_list_exact_and_regex():
    ents = [E("PERSON", TEXT, "田中太郎"), E("ORG", TEXT, "株式会社アオバ")]
    eng = engine_for(TEXT, ents)
    r = eng.transform(TEXT, allow_list=["株式会社アオバ"])
    assert "株式会社アオバ" in r.text and "田中太郎" not in r.text
    r = eng.transform(TEXT, allow_list=[r"株式会社.*"], allow_list_match="regex")
    assert "株式会社アオバ" in r.text
    with pytest.raises(ConfigurationError):
        eng.transform(TEXT, allow_list=["x"], allow_list_match="fuzzy")


def test_deny_list_adds_entities_with_given_type():
    text = "案件コード ハヤブサ計画 は社外秘。担当は田中。"
    ents = [E("PERSON", text, "田中")]
    eng = engine_for(text, ents)
    r = eng.transform(text, deny_list={"ハヤブサ計画": "PROJECT"})
    types = [(e.entity_type, e.operator) for e in r.entities]
    assert ("PROJECT", "tokenize") in types and "ハヤブサ計画" not in r.text
    r2 = eng.transform(text, deny_list=["ハヤブサ計画"])               # 型省略 → CUSTOM
    assert any(e.entity_type == "CUSTOM" for e in r2.entities)
    assert any(e.source == "deny_list" for e in eng.detect(text, deny_list=["ハヤブサ計画"]))


def test_explain_traces_detector_and_operator():
    ents = [E("PERSON", TEXT, "田中太郎", score=0.93, source="fake_ner")]
    r = engine_for(TEXT, ents).transform(TEXT, explain=True)
    person = next(e for e in r.entities if e.entity_type == "PERSON")
    assert "detector=fake_ner(0.93)" in person.explanation and "operator=tokenize" in person.explanation
    assert "田中" not in person.explanation                              # 原文は載せない
    assert engine_for(TEXT, ents).transform(TEXT).entities[0].explanation is None


def test_explain_shows_refiner_and_annotator_steps():
    from privacy_masker import IdentifiabilityAnnotator, SpanRefiner
    text = "岸田首相と田中さんが会談した。"
    ents = [E("PERSON", text, "岸田", source="ner"), E("PERSON", text, "田中さん", source="ner")]
    eng = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy.preset("standard"),
                        refiner=SpanRefiner(), annotators=[IdentifiabilityAnnotator(knowledge=None)])
    r = eng.transform(text, explain=True)
    by_type = {e.entity_type: e.explanation for e in r.entities}
    assert "annotator PERSON→PERSON_PUBLIC" in by_type["PERSON_PUBLIC"]
    assert "refined" in by_type["PERSON"]                                # 田中さん → 田中

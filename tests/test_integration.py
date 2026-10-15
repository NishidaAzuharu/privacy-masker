"""スペック§3の使用例に対応する統合テスト。"""
from privacy_masker import (
    Entity,
    Policy,
    PrivacyEngine,
    RestoreResult,
    TransformedEntity,
    TransformResult,
    entity_types,
    exceptions,
)
from privacy_masker.detectors import RegexDetector
from privacy_masker.operators import Mask, Tokenize

from .conftest import FakeCrypto, FakeDetector


def test_spec_example_end_to_end():
    text = "山田太郎さんは株式会社ABCに勤務しており、東京都新宿区に住んでいる。"
    ner = FakeDetector(
        [
            Entity("PERSON", 0, 4, 0.98, "ner"),
            Entity("ORG", 7, 14, 0.96, "ner"),
            Entity("ADDRESS", 22, 28, 0.91, "ner"),
        ],
        name="ner",
    )
    engine = PrivacyEngine(detectors=[ner], policy=Policy.preset("llm_input"))
    result = engine.transform(text)
    assert result.text == "<PERSON_001>さんは<ORG_001>に勤務しており、<ADDRESS_001>に住んでいる。"


def test_mixed_detectors_and_policy():
    text = "山田太郎の連絡先はtaro@example.comと03-1234-5678。"
    ner = FakeDetector([Entity("PERSON", 0, 4, 0.99, "ner")], name="ner")
    policy = Policy(
        rules={
            "PERSON": Tokenize(strategy="hmac"),
            "PHONE_NUMBER": Mask(keep_suffix=4),
        },
    )
    engine = PrivacyEngine(
        detectors=[ner, RegexDetector.default_ja()],
        policy=policy,
        secret_key=b"integration-key",
    )
    result = engine.transform(text)
    assert "山田太郎" not in result.text
    assert "taro@example.com" not in result.text  # EMAILはdefault(sequential)に落ちる
    assert "<EMAIL_001>" in result.text
    assert "********5678" in result.text


def test_vault_roundtrip_via_public_api():
    text = "患者の山田太郎さん"
    ner = FakeDetector([Entity("PERSON", 3, 7, 0.99)])
    engine = PrivacyEngine(
        detectors=[ner],
        policy=Policy(rules={}, default=Tokenize(strategy="vault")),
        crypto=FakeCrypto(),
        secret_key=b"k" * 16,
    )
    masked = engine.transform(text, scope_id="patient_123")
    restored = engine.restore(masked.text, scope_id="patient_123")
    assert restored.text == text
    # 対応表の破棄（delete_scope）後は復元不能
    engine.store.delete_scope("patient_123")
    assert engine.restore(masked.text, scope_id="patient_123").unresolved != ()


def test_public_exports():
    assert TransformResult is not None
    assert TransformedEntity is not None
    assert RestoreResult is not None
    assert entity_types.PERSON == "PERSON"
    assert issubclass(exceptions.ConfigurationError, exceptions.PrivacyMaskerError)

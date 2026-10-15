import pytest

from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError, RestoreError
from privacy_masker.operators import Tokenize
from privacy_masker.policy import Policy
from privacy_masker.types import Entity

from .conftest import FakeCrypto, FakeDetector

TEXT = "山田太郎さんは株式会社ABCに勤務している。"
ENTS = [Entity("PERSON", 0, 4, 0.98), Entity("ORG", 7, 14, 0.95)]


def vault_engine(**kw):
    kw.setdefault("policy", Policy(rules={}, default=Tokenize(strategy="vault")))
    kw.setdefault("crypto", FakeCrypto())
    kw.setdefault("secret_key", b"k" * 16)
    kw.setdefault("detectors", [FakeDetector(ENTS)])
    return PrivacyEngine(**kw)


def test_vault_roundtrip():
    engine = vault_engine()
    r = engine.transform(TEXT, scope_id="case_42")
    assert "山田太郎" not in r.text
    restored = engine.restore(r.text, scope_id="case_42")
    assert restored.text == TEXT
    assert restored.unresolved == ()


def test_restore_leaves_unknown_tokens_unresolved():
    engine = vault_engine()
    engine.transform(TEXT, scope_id="case_42")
    out = engine.restore("<PERSON_AAAABBBBCCCCDDDDEEEEFFFF>を参照", scope_id="case_42")
    assert out.unresolved == ("<PERSON_AAAABBBBCCCCDDDDEEEEFFFF>",)
    assert "<PERSON_AAAABBBBCCCCDDDDEEEEFFFF>" in out.text


def test_restore_unknown_scope_returns_all_unresolved():
    engine = vault_engine()
    r = engine.transform(TEXT, scope_id="case_42")
    out = engine.restore(r.text, scope_id="no_such_scope")
    assert len(out.unresolved) == 2
    assert out.text == r.text


def test_restore_detects_swapped_blobs():
    engine = vault_engine()
    r = engine.transform(TEXT, scope_id="case_42")
    toks = [e.replacement for e in r.entities]
    store = engine.store
    # store内の暗号文を2トークン間で入れ替える（取り違え攻撃のシミュレーション）
    v0 = store.get_value("case_42", toks[0])
    v1 = store.get_value("case_42", toks[1])
    store._values[("case_42", toks[0])] = v1  # MemoryStoreの内部を直接操作
    store._values[("case_42", toks[1])] = v0
    with pytest.raises(RestoreError):
        engine.restore(r.text, scope_id="case_42")


def test_restore_without_crypto_is_configuration_error():
    engine = PrivacyEngine(detectors=[FakeDetector(ENTS)], secret_key=b"k" * 16)
    with pytest.raises(ConfigurationError):
        engine.restore("<PERSON_001>", scope_id="s")


def test_restore_respects_custom_template():
    policy = Policy(rules={}, default=Tokenize(strategy="vault", template="[[{type}:{id}]]"))
    engine = vault_engine(policy=policy)
    r = engine.transform(TEXT, scope_id="c")
    assert engine.restore(r.text, scope_id="c").text == TEXT


def test_restore_multiple_templates_single_pass():
    # PERSON原文が既存のORGトークン文字列と同一でも、二重置換されない
    policy = Policy(rules={"ORG": Tokenize(strategy="vault", template="[[{type}:{id}]]")},
                    default=Tokenize(strategy="vault"))
    ents_org = [Entity("ORG", 0, 6, 1.0)]
    engine = PrivacyEngine(detectors=[FakeDetector(ents_org)], policy=policy,
                           crypto=FakeCrypto(), secret_key=b"k" * 16)
    r1 = engine.transform("ダミー組織です", scope_id="c1")
    org_token = r1.entities[0].replacement
    # 別文書: PERSON原文がorg_tokenと同一文字列
    text2 = org_token + "は人物です"
    ents_person = [Entity("PERSON", 0, len(org_token), 1.0)]
    engine2 = PrivacyEngine(detectors=[FakeDetector(ents_person)], policy=policy,
                            crypto=FakeCrypto(), secret_key=b"k" * 16, store=engine.store)
    r2 = engine2.transform(text2, scope_id="c1")
    restored = engine2.restore(r2.text, scope_id="c1")
    assert restored.text == text2  # ORGの復号値に化けない
    assert restored.unresolved == ()

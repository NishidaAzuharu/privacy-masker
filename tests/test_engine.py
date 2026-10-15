import sys
import traceback
from array import array

import pytest

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError, DetectorError, StoreError
from privacy_masker.operators import Mask, Surrogate, Tokenize
from privacy_masker.operators.surrogate_filter import SurrogateFilter
from privacy_masker.policy import Policy
from privacy_masker.stores import MemoryStore
from privacy_masker.types import Entity

from .conftest import FakeCrypto, FakeDetector

TEXT = "山田太郎さんは株式会社ABCに勤務している。"
# 山田太郎: 0-4, 株式会社ABC: 7-14
# (ブリーフのコメントは 7-13 だったが、"株式会社ABC" は7文字のため実際は 7-14 が正しい範囲)
ENTS = [
    Entity("PERSON", 0, 4, 0.98, "fake"),
    Entity("ORG", 7, 14, 0.95, "fake"),
]


def make_engine(**kw):
    kw.setdefault("detectors", [FakeDetector(ENTS)])
    return PrivacyEngine(**kw)


def test_transform_basic_sequential():
    result = make_engine().transform(TEXT)
    assert result.text == "<PERSON_001>さんは<ORG_001>に勤務している。"
    assert result.scope_id is None
    assert result.failed_detectors == ()
    assert [e.entity_type for e in result.entities] == ["PERSON", "ORG"]
    assert result.entities[0].original_start == 0
    assert result.entities[0].original_end == 4
    assert result.entities[0].replacement == "<PERSON_001>"
    assert result.entities[0].operator == "tokenize"


def test_transform_offsets_with_emoji():
    text = "😀😀山田太郎さん"
    ents = [Entity("PERSON", 2, 6, 1.0)]
    engine = PrivacyEngine(detectors=[FakeDetector(ents)])
    assert engine.transform(text).text == "😀😀<PERSON_001>さん"


def test_document_scope_is_independent_between_calls():
    engine = make_engine()
    r1 = engine.transform(TEXT)
    r2 = engine.transform(TEXT)
    assert r1.text == r2.text  # どちらも <PERSON_001>


def test_scope_id_consistency_across_calls():
    engine = make_engine(secret_key=b"k" * 16)
    r1 = engine.transform(TEXT, scope_id="case_1")
    text2 = "佐藤花子さんと山田太郎さん"
    ents2 = [Entity("PERSON", 0, 4, 1.0), Entity("PERSON", 7, 11, 1.0)]
    engine2 = PrivacyEngine(
        detectors=[FakeDetector(ents2)], secret_key=b"k" * 16, store=engine.store
    )
    r2 = engine2.transform(text2, scope_id="case_1")
    # 山田太郎はr1で<PERSON_001>を得ているので、r2でも同じトークン
    assert "<PERSON_001>" in r2.text  # 山田太郎
    assert "<PERSON_002>" in r2.text  # 佐藤花子（新規）


def test_detect_returns_resolved_entities():
    engine = make_engine()
    out = engine.detect(TEXT)
    assert out == ENTS


def test_empty_text_returns_unchanged():
    engine = make_engine()
    assert engine.transform("").text == ""
    assert engine.transform("  \n").text == "  \n"


def test_unknown_type_gets_default_operator():
    ents = [Entity("SOME_CUSTOM", 0, 4, 1.0)]
    engine = PrivacyEngine(detectors=[FakeDetector(ents)])
    assert engine.transform("あいうえお").text == "<SOME_CUSTOM_001>お"


def test_invalid_entity_raises_detector_error():
    engine = PrivacyEngine(detectors=[FakeDetector([Entity("PERSON", 3, 2, 1.0)])])
    with pytest.raises(DetectorError):
        engine.transform("あいうえお")
    engine2 = PrivacyEngine(detectors=[FakeDetector([Entity("PERSON", 0, 99, 1.0)])])
    with pytest.raises(DetectorError):
        engine2.transform("あいうえお")


def test_detector_failure_raises_by_default():
    class Boom:
        name = "boom"

        def detect(self, text):
            raise RuntimeError("model crashed")

    engine = PrivacyEngine(detectors=[Boom()])
    with pytest.raises(DetectorError):
        engine.transform("テキスト")


def test_detector_failure_skip_mode_records_failure():
    class Boom:
        name = "boom"

        def detect(self, text):
            raise RuntimeError("model crashed")

    engine = PrivacyEngine(
        detectors=[Boom(), FakeDetector(ENTS)], on_detector_error="skip"
    )
    result = engine.transform(TEXT)
    assert result.failed_detectors == ("boom",)
    assert "<PERSON_001>" in result.text  # 生きているdetector分は適用される


def test_construction_validation():
    # vaultにはcryptoとsecret_keyが必要
    vault_policy = Policy(rules={"PERSON": Tokenize(strategy="vault")})
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], policy=vault_policy,
                      secret_key=b"k" * 16)  # cryptoなし
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], policy=vault_policy,
                      crypto=FakeCrypto())  # secretなし
    # hmacはoperator keyかengine secretが必要
    hmac_policy = Policy(rules={}, default=Tokenize(strategy="hmac"))
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], policy=hmac_policy)
    PrivacyEngine(detectors=[FakeDetector([])], policy=hmac_policy, secret_key=b"k" * 16)
    # secret_keyとsecret_key_envの同時指定
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], secret_key=b"k", secret_key_env="X")
    # detectorゼロ
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[])
    # 不正なon_detector_error
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], on_detector_error="ignore")


def test_call_validation():
    vault_policy = Policy(rules={"PERSON": Tokenize(strategy="vault")})
    engine = PrivacyEngine(
        detectors=[FakeDetector(ENTS)], policy=vault_policy,
        crypto=FakeCrypto(), secret_key=b"k" * 16,
    )
    with pytest.raises(ConfigurationError):
        engine.transform(TEXT)  # vaultはscope_id必須
    engine.transform(TEXT, scope_id="ok")  # これは通る
    # scope_id指定にはsecret_keyが必要
    plain = PrivacyEngine(detectors=[FakeDetector(ENTS)])
    with pytest.raises(ConfigurationError):
        plain.transform(TEXT, scope_id="s1")


def test_detector_error_does_not_leak_input_text():
    class LeakyDetector:
        name = "leaky"

        def detect(self, text):
            raise RuntimeError(f"failed to process: {text}")

    engine = PrivacyEngine(detectors=[LeakyDetector()])
    # secret_textを変数経由で渡す：リテラルのままtransform()呼び出しに埋め込むと、
    # トレースバックのソース行スニペット自体に入力文字列が写り込んでしまい、
    # 「例外連鎖経由で漏れていないか」という本来の検証ができなくなるため。
    secret_text = "山田太郎の秘密情報"
    try:
        engine.transform(secret_text)
    except DetectorError:
        tb = traceback.format_exc()
        assert "山田太郎" not in tb
        assert "RuntimeError" in str(sys.exc_info()[1])  # 型名は残る
    else:
        pytest.fail("DetectorError was not raised")


def test_hmac_token_stable_across_engines():
    policy = Policy(rules={}, default=Tokenize(strategy="hmac"))
    e1 = PrivacyEngine(detectors=[FakeDetector(ENTS)], policy=policy, secret_key=b"k" * 16)
    e2 = PrivacyEngine(detectors=[FakeDetector(ENTS)], policy=policy, secret_key=b"k" * 16)
    assert e1.transform(TEXT).text == e2.transform(TEXT).text


def test_operator_receives_document_text_via_context():
    seen = []

    class Peek:
        name = "peek"

        def apply(self, value, entity, ctx):
            seen.append(getattr(ctx, "text", "MISSING"))
            return "X"

    text = "山田太郎さんは株式会社ABCに勤務している。"
    ents = [Entity("PERSON", 0, 4, 0.98)]
    engine = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(rules={"PERSON": Peek()}),
                           refiner=None)
    engine.transform(text)
    assert seen == [text]
    engine2 = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy(rules={"PERSON": Peek()}),
                            refiner=None, secret_key=b"k" * 16)
    seen.clear()
    engine2.transform(text, scope_id="s1")
    assert seen == [text]


def test_transform_batch_operator_sees_each_documents_own_text_via_context():
    """transform_batch は文書ごとに ctx.text を差し替える（前の文書の全文が次の文書の apply() に
    漏れて渡らないことのピンポイント確認）。detector も文書ごとに違う entity を返す。"""
    seen = []

    class Peek:
        name = "peek"

        def apply(self, value, entity, ctx):
            seen.append(ctx.text)
            return "X"

    text1 = "山田太郎が来た。"
    text2 = "田中花子と鈴木一郎が来た。"
    ent1 = Entity("PERSON", 0, 4, 0.9)
    ents2 = [Entity("PERSON", 0, 4, 0.9), Entity("PERSON", 5, 9, 0.9)]

    class TwoDocDetector:
        name = "fake"

        def detect(self, t):
            return [ent1] if t == text1 else ents2

    engine = PrivacyEngine(detectors=[TwoDocDetector()], policy=Policy(rules={"PERSON": Peek()}), refiner=None)
    engine.transform_batch([text1, text2])
    assert seen == [text1, text2, text2]


def test_engine_forgets_document_text_after_transform():
    """長命の operator（Surrogate）が文書終了後も全文のメモを持ち続けないこと（資源管理）。"""
    op = Surrogate(kind="person", key=b"k")
    eng = PrivacyEngine([FakeDetector([Entity("PERSON", 0, 4, 0.9)])], policy=Policy(rules={"PERSON": op}), refiner=None)
    eng.transform("山田太郎が来た。")
    assert op._filter._text_cache is None
    eng.transform_batch(["山田太郎が来た。", "山田太郎が帰った。"])
    assert op._filter._text_cache is None


class _MemoizeThenRaise:
    """operator のふり: apply() が SurrogateFilter と同じ経路で全文を記憶してから例外を送出する。
    Fix round 1: transform()/transform_batch() が例外経路でも forget_document() を呼ぶことの検査用。
    """

    name = "memo_then_raise"

    def __init__(self, raise_on_call: int = 1):
        self._filter = SurrogateFilter(packaged=PublicKnowledge(array("Q")))
        self._raise_on_call = raise_on_call
        self._calls = 0

    def apply(self, value, entity, ctx):
        self._filter._normalized_text(ctx.text)   # 安全フィルタと同じ経路で全文を記憶させる
        self._calls += 1
        if self._calls == self._raise_on_call:
            raise StoreError("synthetic failure for test")
        return "X"

    def forget_document(self) -> None:
        self._filter.forget_document()


def test_transform_forgets_document_text_even_when_apply_raises():
    """operator.apply() が StoreError 等で例外を送出しても、直前に記憶した全文のメモは
    try/finally で必ず捨てられる（保留 Minor レビュー Fix round 1）。"""
    op = _MemoizeThenRaise(raise_on_call=1)
    eng = PrivacyEngine([FakeDetector([Entity("PERSON", 0, 4, 0.9)])], policy=Policy(rules={"PERSON": op}), refiner=None)
    with pytest.raises(StoreError):
        eng.transform("山田太郎が来た。")
    assert op._filter._text_cache is None


def test_transform_batch_forgets_document_text_even_when_a_later_document_raises():
    """transform_batch は複数文書のうち後の文書の apply() が例外を送出しても、直前の文書で
    SurrogateFilter が記憶した全文のメモを try/finally で必ず捨てる。"""
    op = _MemoizeThenRaise(raise_on_call=2)     # 1文書目は成功して記憶が残り、2文書目で例外
    eng = PrivacyEngine(
        [FakeDetector([Entity("PERSON", 0, 4, 0.9)])], policy=Policy(rules={"PERSON": op}), refiner=None,
    )
    with pytest.raises(StoreError):
        eng.transform_batch(["山田太郎が来た。", "田中花子が来た。"])
    assert op._filter._text_cache is None

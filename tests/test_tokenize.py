import pytest

from privacy_masker.encoding import hmac_token_id
from privacy_masker.exceptions import ConfigurationError, StoreError
from privacy_masker.operators import Tokenize
from privacy_masker.types import Entity

E = Entity(entity_type="PERSON", start=0, end=4, score=1.0)
KEY = b"engine-secret"


class FakeContext:
    """TransformContext Protocolを満たすテスト用実装。"""

    def __init__(self, scope_id="s1", fail_saves=0):
        self.scope_id = scope_id
        self.tokens: dict[tuple[str, str], str] = {}
        self.seq: dict[str, int] = {}
        self.saved: list[tuple[str, str, str, str | None]] = []
        self._fail_saves = fail_saves

    def next_seq(self, entity_type):
        self.seq[entity_type] = self.seq.get(entity_type, 0) + 1
        return self.seq[entity_type]

    def get_token(self, entity_type, normalized_value):
        return self.tokens.get((entity_type, normalized_value))

    def save_token(self, entity_type, normalized_value, token, original_value=None):
        if self._fail_saves > 0:
            self._fail_saves -= 1
            raise StoreError("token already exists in scope")
        self.tokens[(entity_type, normalized_value)] = token
        self.saved.append((entity_type, normalized_value, token, original_value))

    def hmac_token(self, entity_type, normalized_value, key=None, length=8):
        return hmac_token_id(key or KEY, self.scope_id or "", entity_type, normalized_value, length)


def test_sequential_numbering_and_reuse():
    op = Tokenize(strategy="sequential")
    ctx = FakeContext()
    assert op.apply("山田太郎", E, ctx) == "<PERSON_001>"
    assert op.apply("山田太郎", E, ctx) == "<PERSON_001>"  # 同値は再利用
    assert op.apply("佐藤花子", E, ctx) == "<PERSON_002>"


def test_sequential_normalizes_before_lookup():
    op = Tokenize(strategy="sequential")
    ctx = FakeContext()
    assert op.apply("山田太郎", E, ctx) == op.apply("山田　太郎", E, ctx)


def test_hmac_deterministic_and_charset():
    op = Tokenize(strategy="hmac")
    ctx = FakeContext()
    tok = op.apply("山田太郎", E, ctx)
    assert tok == op.apply("山田太郎", E, ctx)
    inner = tok[len("<PERSON_"):-1]
    assert len(inner) == 8
    assert set(inner) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


def test_hmac_operator_key_overrides():
    ctx = FakeContext()
    t1 = Tokenize(strategy="hmac").apply("山田太郎", E, ctx)
    t2 = Tokenize(strategy="hmac", key=b"another-key").apply("山田太郎", E, ctx)
    assert t1 != t2


def test_vault_token_shape_and_original_saved():
    op = Tokenize(strategy="vault")
    ctx = FakeContext()
    tok = op.apply("山田　太郎", E, ctx)
    inner = tok[len("<PERSON_"):-1]
    assert len(inner) == 24  # 15バイト=120bit → base32 24文字
    assert set(inner) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    # 原文はraw値のまま保存される（正規化前）
    assert ctx.saved == [("PERSON", "山田太郎", tok, "山田　太郎")]
    assert op.apply("山田　太郎", E, ctx) == tok  # 再利用


def test_vault_collision_retries():
    op = Tokenize(strategy="vault")
    ctx = FakeContext(fail_saves=2)
    tok = op.apply("山田太郎", E, ctx)  # 2回失敗→3回目成功
    assert tok.startswith("<PERSON_")


def test_vault_collision_exhausted():
    op = Tokenize(strategy="vault")
    ctx = FakeContext(fail_saves=3)
    with pytest.raises(StoreError):
        op.apply("山田太郎", E, ctx)


class ConcurrentSaveContext(FakeContext):
    """get_token→save_tokenの間に別スレッドが同じ正規化値を登録した体のFakeContext。

    save_token呼び出し時にStoreError（value_hash重複）を投げると同時に、
    tokens辞書へ他スレッドが先に登録したトークンを注入する。
    """

    def __init__(self, concurrent_token, scope_id="s1"):
        super().__init__(scope_id=scope_id)
        self._concurrent_token = concurrent_token
        self._injected = False

    def save_token(self, entity_type, normalized_value, token, original_value=None):
        if not self._injected:
            self._injected = True
            self.tokens[(entity_type, normalized_value)] = self._concurrent_token
            raise StoreError("value_hash already exists in scope")
        super().save_token(entity_type, normalized_value, token, original_value)


def test_sequential_concurrent_save_falls_back_to_existing_token():
    op = Tokenize(strategy="sequential")
    ctx = ConcurrentSaveContext(concurrent_token="<PERSON_007>")
    assert op.apply("山田太郎", E, ctx) == "<PERSON_007>"


def test_vault_concurrent_save_falls_back_to_existing_token():
    op = Tokenize(strategy="vault")
    ctx = ConcurrentSaveContext(concurrent_token="<PERSON_ABCDEFGHIJKLMNOPQRSTUV>")
    assert op.apply("山田太郎", E, ctx) == "<PERSON_ABCDEFGHIJKLMNOPQRSTUV>"


def test_custom_template():
    op = Tokenize(strategy="sequential", template="[[{type}:{id}]]")
    assert op.apply("山田太郎", E, FakeContext()) == "[[PERSON:001]]"


def test_config_validation():
    with pytest.raises(ConfigurationError):
        Tokenize(strategy="unknown")
    with pytest.raises(ConfigurationError):
        Tokenize(strategy="vault", token_bytes=4)  # 下限8バイト
    with pytest.raises(ConfigurationError):
        Tokenize(strategy="hmac", key=b"k", key_env="X")  # 同時指定


def test_token_length_validation():
    # SHA-256のbase32はパディング抜き52文字が上限
    with pytest.raises(ConfigurationError):
        Tokenize(strategy="hmac", token_length=0)
    with pytest.raises(ConfigurationError):
        Tokenize(strategy="hmac", token_length=53)
    Tokenize(strategy="hmac", token_length=1)
    Tokenize(strategy="hmac", token_length=52)

"""テスト共通のフェイク実装。coreのテストをstdlibのみで完結させる（スペック§16）。"""
from privacy_masker.encoding import hmac_token_id
from privacy_masker.exceptions import RestoreError, StoreError


class FakeCrypto:
    """CryptoProvider Protocolを満たすテスト用実装（本物の暗号ではない）。"""

    def encrypt(self, value: bytes) -> bytes:
        return b"enc:" + value[::-1]

    def decrypt(self, value: bytes) -> bytes:
        if not value.startswith(b"enc:"):
            raise RestoreError("decryption failed")
        return value[4:][::-1]


class FakeDetector:
    """固定のEntityリストを返すDetector。"""

    def __init__(self, entities, name="fake"):
        self._entities = list(entities)
        self.name = name

    def detect(self, text):
        return list(self._entities)


class SurrogateCtx:
    """Surrogate operator 単体テスト用の TransformContext。next_seq は使わないことを保証する。"""

    def __init__(self, key: bytes = b"k", scope_id: str | None = "s1", text: str | None = None):
        self.key, self.scope_id, self.text = key, scope_id, text
        self.tokens: dict[tuple[str, str], str] = {}
        self.saved_originals: dict[str, str] = {}

    def next_seq(self, entity_type):
        raise AssertionError("surrogate must not use next_seq")

    def get_token(self, entity_type, normalized_value):
        return self.tokens.get((entity_type, normalized_value))

    def save_token(self, entity_type, normalized_value, token, original_value=None):
        if token in self.tokens.values():
            raise StoreError("token already exists in scope")
        self.tokens[(entity_type, normalized_value)] = token
        if original_value is not None:
            self.saved_originals[token] = original_value

    def hmac_token(self, entity_type, normalized_value, key=None, length=8):
        return hmac_token_id(key or self.key, self.scope_id or "", entity_type, normalized_value, length)

"""TransformContext実装（スペック§8）。document scope / scope_id（Store管理）。"""
from __future__ import annotations

from .crypto.base import CryptoProvider
from .encoding import encode_fields, hmac_token_id, value_hash
from .exceptions import ConfigurationError, StoreError
from .secrets_types import SecretBytes
from .stores.base import PseudonymStore


class _HmacMixin:
    scope_id: str | None
    _secret: SecretBytes | None

    def hmac_token(
        self, entity_type: str, normalized_value: str,
        key: bytes | None = None, length: int = 8,
    ) -> str:
        k = key if key is not None else (self._secret.reveal() if self._secret else None)
        if k is None:
            raise ConfigurationError(
                "hmac strategy requires a key (operator key/key_env or engine secret_key)"
            )
        return hmac_token_id(k, self.scope_id or "", entity_type, normalized_value, length)


class DocumentContext(_HmacMixin):
    """呼び出し内で完結するcontext。Store不要・restore不可。"""

    def __init__(self, secret: SecretBytes | None, text: str | None = None) -> None:
        self.scope_id: str | None = None
        self.text = text
        self._secret = secret
        self._tokens: dict[tuple[str, str], str] = {}
        self._seq: dict[str, int] = {}

    def next_seq(self, entity_type: str) -> int:
        self._seq[entity_type] = self._seq.get(entity_type, 0) + 1
        return self._seq[entity_type]

    def get_token(self, entity_type: str, normalized_value: str) -> str | None:
        return self._tokens.get((entity_type, normalized_value))

    def save_token(
        self, entity_type: str, normalized_value: str, token: str,
        original_value: str | None = None,
    ) -> None:
        if original_value is not None:  # vaultはdocument scopeでは使えない（防御的）
            raise ConfigurationError("vault strategy requires scope_id")
        if token in self._tokens.values():  # 別実体への同一トークン割当を拒否（Surrogateが再生成する）
            raise StoreError("token already exists in scope")
        self._tokens[(entity_type, normalized_value)] = token


class StoreContext(_HmacMixin):
    """scope_id指定時のStoreバックドcontext。"""

    def __init__(
        self, scope_id: str, store: PseudonymStore,
        crypto: CryptoProvider | None, secret: SecretBytes, text: str | None = None,
    ) -> None:
        self.scope_id: str | None = scope_id
        self.text = text
        self._store = store
        self._crypto = crypto
        self._secret: SecretBytes | None = secret

    def _vh(self, entity_type: str, normalized_value: str) -> str:
        assert self._secret is not None
        assert self.scope_id is not None
        return value_hash(self._secret.reveal(), self.scope_id, entity_type, normalized_value)

    def next_seq(self, entity_type: str) -> int:
        assert self.scope_id is not None
        return self._store.next_seq(self.scope_id, entity_type)

    def get_token(self, entity_type: str, normalized_value: str) -> str | None:
        assert self.scope_id is not None
        return self._store.get_token(self.scope_id, entity_type, self._vh(entity_type, normalized_value))

    def save_token(
        self, entity_type: str, normalized_value: str, token: str,
        original_value: str | None = None,
    ) -> None:
        assert self.scope_id is not None
        encrypted: bytes | None = None
        if original_value is not None:
            if self._crypto is None:
                raise ConfigurationError("vault strategy requires a CryptoProvider")
            # 平文にscope_idとtokenを内包し、復号時に取り違えを検知する（スペック§10）
            encrypted = self._crypto.encrypt(encode_fields(self.scope_id, token, original_value))
        self._store.put_token(
            self.scope_id, entity_type, self._vh(entity_type, normalized_value), token, encrypted
        )

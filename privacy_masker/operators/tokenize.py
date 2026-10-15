"""Tokenize operator（スペック§7）。戦略: sequential / hmac / vault。"""
from __future__ import annotations

import base64
import secrets
from typing import Callable

from ..exceptions import ConfigurationError, StoreError
from ..normalize import default_normalize
from ..secrets_types import SecretBytes
from ..types import Entity
from .base import TransformContext

_STRATEGIES = ("sequential", "hmac", "vault")
_VAULT_RETRIES = 3


class Tokenize:
    name = "tokenize"

    def __init__(
        self,
        strategy: str = "sequential",
        template: str = "<{type}_{id}>",
        token_length: int = 8,
        token_bytes: int = 15,
        seq_width: int = 3,
        key: bytes | None = None,
        key_env: str | None = None,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ConfigurationError(f"unknown tokenize strategy: {strategy!r}")
        if token_bytes < 8:
            raise ConfigurationError("token_bytes must be >= 8 (64bit minimum)")
        if not 1 <= token_length <= 52:
            raise ConfigurationError("token_length must be between 1 and 52")
        if key is not None and key_env is not None:
            raise ConfigurationError("key and key_env are mutually exclusive")
        self.strategy = strategy
        self.template = template
        self.token_length = token_length
        self.token_bytes = token_bytes
        self.seq_width = seq_width
        self.key: SecretBytes | None
        if key is not None:
            self.key = SecretBytes(key)
        elif key_env is not None:
            self.key = SecretBytes.from_env(key_env)
        else:
            self.key = None
        self.normalizer = normalizer or default_normalize

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        norm = self.normalizer(value)
        etype = entity.entity_type
        if self.strategy == "sequential":
            token = ctx.get_token(etype, norm)
            if token is None:
                n = ctx.next_seq(etype)
                token = self.template.format(type=etype, id=f"{n:0{self.seq_width}d}")
                try:
                    ctx.save_token(etype, norm, token)
                except StoreError:
                    # get_token→save_tokenの間に別スレッドが同じ正規化値を
                    # 登録した可能性がある（lost update）。既存トークンがあれば
                    # それを採用し、無ければ元の例外を送出する。
                    existing = ctx.get_token(etype, norm)
                    if existing is None:
                        raise
                    token = existing
            return token
        if self.strategy == "hmac":
            key = self.key.reveal() if self.key is not None else None
            tid = ctx.hmac_token(etype, norm, key=key, length=self.token_length)
            return self.template.format(type=etype, id=tid)
        # vault
        token = ctx.get_token(etype, norm)
        if token is not None:
            return token
        last_error: StoreError | None = None
        for _ in range(_VAULT_RETRIES):
            tid = base64.b32encode(secrets.token_bytes(self.token_bytes)).decode("ascii").rstrip("=")
            token = self.template.format(type=etype, id=tid)
            try:
                ctx.save_token(etype, norm, token, original_value=value)
                return token
            except StoreError as e:
                # value_hash重複には2通りの原因があり得る：
                # (a) 別スレッドが同じ正規化値を並行登録した（lost update）
                # (b) 生成したトークン自体が別エンティティと衝突した（真の衝突）
                # (a)なら既存トークンを再利用して確定させ、(b)ならトークンを
                # 再生成してリトライする。
                existing = ctx.get_token(etype, norm)
                if existing is not None:
                    return existing
                last_error = e
        raise StoreError("vault token collision persisted after retries") from last_error

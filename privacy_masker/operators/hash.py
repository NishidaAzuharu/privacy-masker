"""Hash: 値を暗号学的ハッシュの16進文字列に置き換える。

同じ値は同じ出力になるので、文書をまたいだ参照整合性が保てる。
注意: 電話番号のような低エントロピー値は、salt が無いと全数照合で元に戻せる。
鍵で保護したい場合は Tokenize(strategy="hmac") か Surrogate を使うこと。
"""
from __future__ import annotations

import hashlib

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext

_ALGORITHMS = ("sha256", "sha512", "sha3_256", "blake2b", "md5")


class Hash:
    name = "hash"

    def __init__(
        self,
        algorithm: str = "sha256",
        salt: bytes | str | None = None,
        length: int | None = None,
        template: str = "{hash}",
    ) -> None:
        if algorithm not in _ALGORITHMS:
            raise ConfigurationError(f"hash algorithm must be one of {_ALGORITHMS}")
        if length is not None and length < 4:
            raise ConfigurationError("hash length must be >= 4")
        if "{hash}" not in template:
            raise ConfigurationError("hash template must contain '{hash}'")
        self.algorithm = algorithm
        self.salt = salt.encode("utf-8") if isinstance(salt, str) else salt
        self.length = length
        self.template = template

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        h = hashlib.new(self.algorithm)
        if self.salt:
            h.update(self.salt)
        h.update(value.encode("utf-8"))
        digest = h.hexdigest()
        if self.length is not None:
            digest = digest[: self.length]
        return self.template.format(hash=digest, type=entity.entity_type)

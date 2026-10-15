"""Encrypt: 値を Fernet で暗号化し、暗号文（URL-safe base64）を本文に埋める（Presidio の `encrypt` / `decrypt` に相当）。

`Tokenize(strategy="vault")` は原文を Store に保存してトークンを置くが、Encrypt は暗号文そのものを本文に置くので
Store が要らず、鍵さえあれば `decrypt_text()` で本文から直接復元できる。`pip install 'privacy-masker[crypto]'` が必要。
"""
from __future__ import annotations

import re

from ..exceptions import ConfigurationError, RestoreError
from ..types import Entity
from .base import TransformContext

# Fernet トークンは先頭バイト 0x80 のため base64 で必ず "gAAAAA" から始まる
_FERNET_TOKEN_RE = re.compile(r"gAAAAA[A-Za-z0-9_-]{60,}={0,2}")


def _fernet(key: bytes | str):
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as e:  # pragma: no cover
        raise ImportError("Encrypt requires the 'cryptography' package. Install with: pip install 'privacy-masker[crypto]'") from e
    try:
        return Fernet(key), InvalidToken
    except (ValueError, TypeError) as e:
        raise ConfigurationError("Encrypt key must be a 32-byte url-safe base64 key (use Encrypt.generate_key())") from e


class Encrypt:
    name = "encrypt"

    def __init__(self, key: bytes | str) -> None:
        self._fernet, _ = _fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_text(text: str, key: bytes | str, *, strict: bool = False) -> str:
    """本文中の Fernet トークンをすべて復号して元の値に戻す。

    strict=False（既定）では復号できないトークン（別の鍵・改変）はそのまま残す。strict=True なら RestoreError。
    """
    fernet, InvalidToken = _fernet(key)

    def _sub(m: "re.Match[str]") -> str:
        try:
            return fernet.decrypt(m.group(0).encode("ascii")).decode("utf-8")
        except InvalidToken:
            if strict:
                raise RestoreError("encrypted token could not be decrypted with this key")
            return m.group(0)

    return _FERNET_TOKEN_RE.sub(_sub, text)

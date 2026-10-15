"""FormatPreserving: 桁数と文字種を保った決定的な仮名（Google SDP の cryptoReplaceFfxFpeConfig の用途に相当）。

数字は数字、英大文字は英大文字、英小文字は英小文字に置き換え、区切り記号・空白・漢字かなはそのまま残す。
「03-1234-5678」→「07-9302-4416」のように見た目の形式を壊さずに置換でき、同じ鍵なら同じ入力に同じ出力になる。
**可逆ではない**（HMAC ベースの置換であり FF1 のような暗号ではない）。復元が必要なら Tokenize(strategy="vault") か Encrypt を使うこと。
Luhn や検査用数字は保存しないので、検証器にかけると不正な番号になる（意図的。実在番号と衝突しにくい）。
"""
from __future__ import annotations

import hashlib
import hmac
import string

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext

_DIGITS = string.digits
_UPPER = string.ascii_uppercase
_LOWER = string.ascii_lowercase


class FormatPreserving:
    name = "format_preserving"

    def __init__(self, key: bytes | str | None = None, *, key_env: str | None = None) -> None:
        import os
        if key is not None and key_env is not None:
            raise ConfigurationError("key and key_env are mutually exclusive")
        if key_env is not None:
            key = os.environ.get(key_env)
            if not key:
                raise ConfigurationError(f"environment variable {key_env} is not set")
        self.key = key.encode("utf-8") if isinstance(key, str) else key

    def _stream(self, value: str, entity: Entity, ctx: TransformContext):
        k = self.key
        if k is None:
            k = ctx.hmac_token("FORMAT_PRESERVING", "", length=16).encode("ascii")   # engine の secret_key から派生
        counter = 0
        while True:
            block = hmac.new(k, f"{entity.entity_type}\x1f{value}\x1f{counter}".encode("utf-8"), hashlib.sha256).digest()
            for b in block:
                yield b
            counter += 1

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        try:
            gen = self._stream(value, entity, ctx)
        except Exception as e:  # noqa: BLE001
            raise ConfigurationError("FormatPreserving requires key=, key_env=, or an engine secret_key") from e
        out = []
        for ch in value:
            if ch in _DIGITS:
                out.append(_DIGITS[next(gen) % 10])
            elif ch in _UPPER:
                out.append(_UPPER[next(gen) % 26])
            elif ch in _LOWER:
                out.append(_LOWER[next(gen) % 26])
            elif "０" <= ch <= "９":
                out.append(chr(ord("０") + next(gen) % 10))
            else:
                out.append(ch)
        return "".join(out)

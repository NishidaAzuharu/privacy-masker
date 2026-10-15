"""鍵material用ラッパー。repr/strから鍵が漏れない（スペック§13）。"""
from __future__ import annotations

import os

from .exceptions import ConfigurationError


class SecretBytes:
    __slots__ = ("_value",)

    def __init__(self, value: bytes | bytearray) -> None:
        if not isinstance(value, (bytes, bytearray)) or len(value) == 0:
            raise ConfigurationError("secret key must be non-empty bytes")
        self._value = bytes(value)

    @classmethod
    def from_env(cls, name: str) -> "SecretBytes":
        raw = os.environ.get(name)
        if not raw:
            raise ConfigurationError(f"environment variable {name} is not set")
        return cls(raw.encode("utf-8"))

    def reveal(self) -> bytes:
        return self._value

    def __repr__(self) -> str:
        return "SecretBytes(****)"

    __str__ = __repr__

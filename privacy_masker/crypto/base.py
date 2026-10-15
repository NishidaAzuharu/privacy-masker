"""CryptoProvider Protocol（スペック§10）。vault対応表の暗号化にのみ使用。"""
from __future__ import annotations

from typing import Protocol


class CryptoProvider(Protocol):
    def encrypt(self, value: bytes) -> bytes: ...

    def decrypt(self, value: bytes) -> bytes: ...

"""Operator Protocol と TransformContext Protocol（スペック§7）。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Entity


class TransformContext(Protocol):
    """カスタムOperator実装者向けの公開API。Store/Cryptoへのアクセスを仲介する。"""

    scope_id: str | None
    text: str | None          # 変換中の文書全文（v0.9。Surrogate の「文書内の既存表層」フィルタが使う。無ければ None）

    def next_seq(self, entity_type: str) -> int: ...

    def get_token(self, entity_type: str, normalized_value: str) -> str | None: ...

    def save_token(
        self, entity_type: str, normalized_value: str, token: str,
        original_value: str | None = None,
    ) -> None: ...

    def hmac_token(
        self, entity_type: str, normalized_value: str,
        key: bytes | None = None, length: int = 8,
    ) -> str: ...


@runtime_checkable
class Operator(Protocol):
    name: str

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str: ...

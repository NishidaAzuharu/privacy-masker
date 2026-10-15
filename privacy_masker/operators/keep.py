from __future__ import annotations

from ..types import Entity
from .base import TransformContext


class Keep:
    """明示的な素通し（スペック§7 フェイルセーフ原則の唯一の例外）。"""

    name = "keep"

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        return value

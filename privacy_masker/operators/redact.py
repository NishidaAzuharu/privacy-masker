from __future__ import annotations

from ..types import Entity
from .base import TransformContext


class Redact:
    name = "redact"

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        return ""

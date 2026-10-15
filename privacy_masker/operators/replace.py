from __future__ import annotations

from ..types import Entity
from .base import TransformContext


class Replace:
    name = "replace"

    def __init__(self, text: str) -> None:
        self.text = text

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        return self.text

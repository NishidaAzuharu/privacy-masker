"""Custom: 利用者の関数で置換文字列を決める（Presidio の `custom` に相当）。

fn(value, entity) -> str。value は原文の該当部分、entity は型・位置・スコア。戻り値が str でなければ ConfigurationError。
"""
from __future__ import annotations

from typing import Callable

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext


class Custom:
    name = "custom"

    def __init__(self, fn: Callable[[str, Entity], str]) -> None:
        if not callable(fn):
            raise ConfigurationError("Custom requires a callable fn(value, entity) -> str")
        self.fn = fn

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        out = self.fn(value, entity)
        if not isinstance(out, str):
            raise ConfigurationError("Custom fn must return str")
        return out

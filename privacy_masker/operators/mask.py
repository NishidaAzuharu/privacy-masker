"""構造を解釈しない純粋な文字数ベースのマスク（スペック§7）。"""
from __future__ import annotations

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext


class Mask:
    name = "mask"

    def __init__(
        self, char: str = "*", keep_prefix: int = 0, keep_suffix: int = 0, *,
        chars_to_mask: int | None = None, masking_char: str | None = None, from_end: bool = False,
    ) -> None:
        """keep_prefix / keep_suffix は「残す文字数」。別名の chars_to_mask / masking_char / from_end
        は「マスクする文字数」（from_end=True なら末尾から）。両方指定したら chars_to_mask を優先する。"""
        if masking_char is not None:
            char = masking_char
        if len(char) != 1:
            raise ConfigurationError("mask char must be a single character")
        if keep_prefix < 0 or keep_suffix < 0 or (chars_to_mask is not None and chars_to_mask < 0):
            raise ConfigurationError("keep_prefix / keep_suffix / chars_to_mask must be >= 0")
        self.char = char
        self.keep_prefix = keep_prefix
        self.keep_suffix = keep_suffix
        self.chars_to_mask = chars_to_mask
        self.from_end = from_end

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        if self.chars_to_mask is not None:
            n = min(self.chars_to_mask, len(value))
            if n == 0:
                return self.char * len(value)  # 0 文字マスク = 素通しになる構成は全マスクに倒す
            return (value[: len(value) - n] + self.char * n) if self.from_end else (self.char * n + value[n:])
        if self.keep_prefix + self.keep_suffix >= len(value):
            return self.char * len(value)  # 実質素通しになる構成は全マスクに倒す
        head = value[: self.keep_prefix]
        tail = value[len(value) - self.keep_suffix :] if self.keep_suffix else ""
        return head + self.char * (len(value) - self.keep_prefix - self.keep_suffix) + tail

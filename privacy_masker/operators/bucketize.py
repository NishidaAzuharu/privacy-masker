"""Bucketize: 数値を幅 size の区間に一般化する（Google SDP の fixedSizeBucketingConfig に相当）。

「34歳」→「30-39歳」、「年収 4,560,000円」→「4,000,000-4,999,999円」。値の中の最初の数値を対象にし、前後の文字（歳・円・約）は残す。
数値が無い値は fallback に置き換える。AGE と組み合わせて年齢の粗視化に使うのが典型。
"""
from __future__ import annotations

import re

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext

_NUM = re.compile(r"\d[\d,]*")


class Bucketize:
    name = "bucketize"

    def __init__(self, size: int = 10, *, template: str = "{lo}-{hi}", fallback: str = "<NUMBER>") -> None:
        if size < 1 or "{lo}" not in template:
            raise ConfigurationError("Bucketize: size >= 1 and template must contain {lo}")
        self.size, self.template, self.fallback = size, template, fallback

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        m = _NUM.search(value)
        if not m:
            return self.fallback
        n = int(m.group(0).replace(",", ""))
        lo = (n // self.size) * self.size
        hi = lo + self.size - 1
        return value[: m.start()] + self.template.format(lo=lo, hi=hi) + value[m.end():]

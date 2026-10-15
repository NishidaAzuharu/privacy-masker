"""DateShift: 日付を一定日数ずらす（Google Sensitive Data Protection の dateShiftConfig に相当）。

同じ鍵（engine の secret_key か operator の key）なら常に同じ日数ずらすので、文書内・文書間で日付の順序と間隔が保たれる。
和暦（令和・平成・昭和・大正・明治）と西暦（年月日・スラッシュ・ハイフン・ドット）を読み、入力と同じ書式で出す。
読めない値は fallback（既定 "<DATE>"）に置き換える（fail-closed）。RegexDetector.default_ja(dates=True) の DATE と組み合わせて使う。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import re

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import TransformContext

_ERAS = {"令和": 2018, "平成": 1988, "昭和": 1925, "大正": 1911, "明治": 1867, "R": 2018, "H": 1988, "S": 1925, "T": 1911, "M": 1867}
_ERA_NAMES = {v: k for k, v in _ERAS.items() if len(k) > 1}
_WAREKI = re.compile(r"^(令和|平成|昭和|大正|明治|[RHSTM])\s?(元|\d{1,2})\s?年\s?(\d{1,2})\s?月\s?(\d{1,2})\s?日$")
_SEIREKI = re.compile(r"^(\d{4})\s?年\s?(\d{1,2})\s?月\s?(\d{1,2})\s?日$")
_NUMERIC = re.compile(r"^(\d{4})([/\-.])(\d{1,2})\2(\d{1,2})$")


def _parse(value: str):
    v = value.strip()
    m = _SEIREKI.match(v)
    if m:
        y, mo, d = map(int, m.groups()); return _dt.date(y, mo, d), ("seireki", None)
    m = _NUMERIC.match(v)
    if m:
        y, sep, mo, d = m.group(1), m.group(2), m.group(3), m.group(4)
        return _dt.date(int(y), int(mo), int(d)), ("numeric", (sep, len(mo) == 2, len(d) == 2))
    m = _WAREKI.match(v)
    if m:
        era, ey, mo, d = m.groups()
        y = _ERAS[era] + (1 if ey == "元" else int(ey))
        return _dt.date(y, int(mo), int(d)), ("wareki", era)
    return None


def _render(date: _dt.date, fmt) -> str:
    kind, arg = fmt
    if kind == "seireki":
        return f"{date.year}年{date.month}月{date.day}日"
    if kind == "numeric":
        sep, pad_m, pad_d = arg
        mo = f"{date.month:02d}" if pad_m else str(date.month)
        d = f"{date.day:02d}" if pad_d else str(date.day)
        return f"{date.year}{sep}{mo}{sep}{d}"
    era_start = max((s for s in _ERA_NAMES if date.year > s), default=None)
    if era_start is None:
        return f"{date.year}年{date.month}月{date.day}日"
    ey = date.year - era_start
    name = _ERA_NAMES[era_start] if len(arg) > 1 else _ERA_NAMES[era_start][0]
    return f"{name}{'元' if ey == 1 else ey}年{date.month}月{date.day}日"


class DateShift:
    name = "date_shift"

    def __init__(self, *, days: int | None = None, max_days: int = 365, key: bytes | str | None = None,
                 fallback: str = "<DATE>") -> None:
        """days を指定すれば固定シフト。指定しなければ鍵から [-max_days, +max_days]（0 を除く）の日数を決定的に決める。"""
        if max_days < 1:
            raise ConfigurationError("max_days must be >= 1")
        self.days, self.max_days, self.fallback = days, max_days, fallback
        self.key = key.encode("utf-8") if isinstance(key, str) else key

    def _shift_days(self, ctx: TransformContext) -> int:
        if self.days is not None:
            return self.days
        k = self.key
        if k is None:
            try:
                k = ctx.hmac_token("DATE_SHIFT", "", length=16).encode("ascii")   # engine の secret_key から派生
            except Exception as e:  # noqa: BLE001
                raise ConfigurationError("DateShift requires days=, key=, or an engine secret_key") from e
        digest = hmac.new(k, b"privacy-masker:date-shift", hashlib.sha256).digest()
        n = int.from_bytes(digest[:4], "big") % (2 * self.max_days) - self.max_days
        return n if n != 0 else self.max_days

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        parsed = _parse(value)
        if parsed is None:
            return self.fallback
        date, fmt = parsed
        try:
            shifted = date + _dt.timedelta(days=self._shift_days(ctx))
        except OverflowError:
            return self.fallback
        return _render(shifted, fmt)

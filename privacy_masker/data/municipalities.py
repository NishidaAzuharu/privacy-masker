"""遅延読み込みする日本郵便の市区町村一覧と表記索引。

郵便データの政令市の区から、親の政令市も導出する。rows は元の1,892件のみ。
曖昧な表記は正式名・県・親市の文脈で解決し、駅の幹だけは唯一の市を優先する。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import resources
import json
import re
from types import MappingProxyType

from .ja_pools import PREFECTURES
from ..exceptions import ConfigurationError

NAME = "ja_municipalities.tsv"


@dataclass(frozen=True, order=True)
class Municipality:
    prefecture: str
    name: str
    kind: str
    parent: str = ""

    @property
    def short(self) -> str:
        if self.parent:
            return self.name[len(self.parent):]
        return self.name.split("郡", 1)[-1] if self.kind in ("町", "村") else self.name

    @property
    def station_stem(self) -> str:
        return (self.parent or self.short)[:-1]

    @property
    def office(self) -> str:
        return self.short + ("役場" if self.kind in ("町", "村") else "役所")

    def render(self, source: Municipality, spelling: str) -> str:
        # 特別区は正式名が区単独。移行先の正式な市+区を使う。
        # 政令市の区の略記（青葉区）は区単独の形を維持する。
        if source.parent and spelling == source.short:
            return self.short
        if source.kind in ("町", "村") and "郡" not in spelling:
            return self.short
        return self.name


def _record(pref: str, name: str) -> Municipality:
    m = re.fullmatch(r"(.+市)(.+区)", name)
    if m:
        return Municipality(pref, name, "政令市の区", m[1])
    return Municipality(pref, name, "特別区" if name.endswith("区") else name[-1])


class Municipalities:
    def __init__(self, pairs):
        self.rows = tuple(_record(p, n) for p, n in pairs)
        parents = {Municipality(r.prefecture, r.parent, "政令市") for r in self.rows if r.parent}
        records = tuple(sorted(set(self.rows) | parents))
        aliases = {}
        stems = {}
        by_pref = {}
        for r in records:
            by_pref.setdefault(r.prefecture, []).append(r)
            for alias in {r.name, r.short}:
                aliases.setdefault(alias, set()).add(r)
                # 一文字幹は駅の連動対象外（中央区なども他候補との曖昧さを解決する）。
                if len(alias[:-1]) >= 2:
                    stems.setdefault(alias[:-1], set()).add(r)
        self.records = records
        self.by_prefecture = MappingProxyType({p: tuple(v) for p, v in by_pref.items()})
        self.aliases = MappingProxyType({a: tuple(sorted(v)) for a, v in aliases.items()})
        self.stems = MappingProxyType({a: tuple(sorted(v)) for a, v in stems.items()})
        self.prefectures_by_name = MappingProxyType(
            {a: tuple(sorted({r.prefecture for r in v})) for a, v in self.aliases.items()})
        self._prefixes = tuple(sorted(self.aliases, key=lambda s: (-len(s), s)))

    def resolve(self, name: str, text: str | None = None, prefecture: str | None = None,
                *, stem: bool = False) -> Municipality | None:
        candidates = self.aliases.get(name, ())
        if stem:
            candidates = tuple(sorted(set(candidates) | set(self.stems.get(name, ()))))
        if prefecture:
            candidates = tuple(r for r in candidates if r.prefecture == prefecture)
        if len(candidates) == 1:
            return candidates[0]
        if text:
            named = [r for r in candidates if r.name in text or r.short in text]
            if len(named) == 1:
                return named[0]
            contextual = [r for r in candidates if r.prefecture in text or (r.parent and r.parent in text)]
            if len(contextual) == 1:
                return contextual[0]
        if stem:
            cities = [r for r in candidates if r.kind in ("市", "政令市", "特別区")]
            if len(cities) == 1 and all(r == cities[0] or r.kind in ("町", "村") for r in candidates):
                return cities[0]
        return None

    def split(self, value: str, text: str | None = None):
        """(県の表記, 市区町村の表記, record, 残り)。最長表記が曖昧なら短い名へ逃げない。"""
        pref = next((p for p in PREFECTURES if value.startswith(p)), "")
        rest = value[len(pref):]
        spelling = next((a for a in self._prefixes if rest.startswith(a)), "")
        record = self.resolve(spelling, text, pref or None) if spelling else None
        return pref, spelling, record, rest[len(spelling):]

    def candidates(self, source: Municipality, prefecture: str, spelling: str):
        choices = self.by_prefecture[prefecture]
        kinds = {"市": ("市",), "政令市": ("政令市", "市"),
                 "政令市の区": ("政令市の区", "市"), "特別区": ("政令市の区", "市"),
                 "町": ("町",), "村": ("村", "町")}[source.kind]
        result = []
        for kind in kinds:
            selected = tuple(r for r in choices if r.kind == kind)
            if kind in ("町", "村") and "郡" in spelling:
                selected = tuple(r for r in selected if "郡" in r.name)
            result.extend(selected)
        return tuple(result)


@lru_cache(maxsize=1)
def load_municipalities() -> Municipalities:
    try:
        root = resources.files("privacy_masker.data")
        payload = root.joinpath(NAME).read_bytes()
        manifest = json.loads(root.joinpath("ja_municipalities.manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError("checksum mismatch")
        pairs = [tuple(line.split("\t")) for line in payload.decode("utf-8").splitlines()]
        if not pairs or len(pairs) != manifest["entries"] or pairs != sorted(set(pairs)):
            raise ValueError("invalid count or ordering")
        if any(len(row) != 2 or row[0] not in PREFECTURES or not row[1]
               or row[1][-1] not in "市区町村" for row in pairs):
            raise ValueError("invalid municipality row")
        return Municipalities(pairs)
    except (OSError, ValueError, KeyError, TypeError, ModuleNotFoundError) as exc:
        raise ConfigurationError("packaged municipality data is missing or corrupt; rebuild with tools/build_municipalities.py") from exc

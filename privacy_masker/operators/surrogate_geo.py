"""semantic Surrogate 専用の地名連動。旧生成器・候補列挙の集合は変えない。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from ..data.ja_pools import PREFECTURES, REGION_BLOCKS
from ..data.municipalities import load_municipalities
from ..exceptions import StoreError
from .semantic import avoid_parts, facility_shape

_INTERNAL_TYPE = "\x00surrogate_geo"
_RNG = random.SystemRandom()


@dataclass(frozen=True)
class GeoPlan:
    replacement: str
    address_prefix: bool = False
    facility_affix: bool = False


def _ok(op, candidate, original, ctx, parts=()):
    return op._restorable(candidate) and op._filter.reject_reason(
        candidate, original_norm=original, avoid_parts=parts,
        text=getattr(ctx, "text", None), check_public=False,
    ) is None


def _mapping(op, ctx, key, draw):
    """内部対応は値なし・制御文字付きトークン。置換の保存・復元とは分離する。

    実際に出力した値だけが通常の entity type と original_value 付きで保存される。
    内部エントリは復元対象にも next_seq / 変換結果の統計にもならない。
    """
    if op.strategy == "hmac":
        return draw()
    prefix = "\x00geo:" + hashlib.sha256(key.encode()).hexdigest() + ":"
    existing = ctx.get_token(_INTERNAL_TYPE, key)
    if existing is not None:
        return existing[len(prefix):] or None
    result = draw()
    try:
        ctx.save_token(_INTERNAL_TYPE, key, prefix + (result or ""))
    except StoreError:
        existing = ctx.get_token(_INTERNAL_TYPE, key)
        if existing is None:
            raise
        return existing[len(prefix):] or None
    return result


def _stored(op, ctx, key):
    token = ctx.get_token(_INTERNAL_TYPE, key)
    return token.rsplit(":", 1)[-1] if token is not None else None


def _saved_city_prefecture(source_pref, ctx):
    """県の内部対応がない旧ストアでも、一意な市対応から所属県を引き継ぐ。"""
    data = load_municipalities()
    targets = set()
    for source in data.by_prefecture[source_pref]:
        spellings = {source.prefecture + source.name}
        # 府中市・中央区など、どの元県の対応か不明な通常キーからは推測しない。
        spellings.update(s for s in (source.name, source.short)
                         if data.prefectures_by_name[s] == (source_pref,))
        for spelling in spellings:
            value = ctx.get_token("LOCATION", spelling)
            if value:
                _pref, _spelling, target, rest = data.split(value)
                if target and not rest:
                    targets.add(target.prefecture)
    return next(iter(targets)) if len(targets) == 1 else None


def _prefecture_cycle(op, block, ctx):
    choices = REGION_BLOCKS["東北"] if len(block) == 1 else block
    rng = op._hmac_rng(ctx, "LOCATION", "pref-block:" + choices[0])
    return rng.sample(choices, len(choices))


def _following_prefectures(source, ordered):
    start = (ordered.index(source) + 1) % len(ordered) if source in ordered else 0
    return ordered[start:] + ordered[:start]


def _document_prefectures(op, ctx):
    text = getattr(ctx, "text", None)
    normalized = op.normalizer(text) if text else ""
    return {p for p in PREFECTURES if p in normalized}


def _prefecture_assignments(op, present, ctx):
    """本文の県を巡回順で割り当てる。地方内の空きは地方外への退避より先に使う。"""
    result, used, pending = {}, set(), []

    def choose(source, candidates):
        parts = avoid_parts("location", source)
        for candidate in candidates:
            if candidate not in present and candidate not in used and _ok(op, candidate, source, ctx, parts):
                used.add(candidate)
                return candidate
        return None

    for block in REGION_BLOCKS.values():
        ordered = _prefecture_cycle(op, block, ctx)
        sources = block if len(block) == 1 else ordered
        for source in sources:
            if source not in present:
                continue
            candidates = _following_prefectures(source, ordered)
            # 県が一つだけの文は、禁止語などによる退避も含め従来どおり。
            result[source] = choose(source, candidates[:1] if len(present) == 1 else candidates)
            if result[source] is None and len(present) > 1:
                pending.append((source, ordered))
    for source, ordered in pending:
        other = [p for p in PREFECTURES if p not in ordered]
        rng = op._hmac_rng(ctx, "LOCATION", "pref-overflow:" + source)
        result[source] = choose(source, rng.sample(other, len(other)))
    return result


def prefecture_for(op, source, ctx):
    def draw():
        block = next(prefs for prefs in REGION_BLOCKS.values() if source in prefs)
        choices = REGION_BLOCKS["東北"] if len(block) == 1 else block
        if op.strategy == "hmac":
            present = _document_prefectures(op, ctx)
            assignments = _prefecture_assignments(op, present, ctx) if present else {}
            if source in assignments:
                return assignments[source]
            ordered = _following_prefectures(source, _prefecture_cycle(op, block, ctx))
            if present:
                # 県を明記しない市などからの参照は、本文の県の割当を変えずに選ぶ。
                other = [p for p in PREFECTURES if p not in ordered]
                rng = op._hmac_rng(ctx, "LOCATION", "pref-overflow:" + source)
                used = set(assignments.values())
                for candidate in ordered + rng.sample(other, len(other)):
                    if candidate not in used and _ok(op, candidate, source, ctx, avoid_parts("location", source)):
                        return candidate
                return None
            candidate = ordered[0]
            return candidate if _ok(op, candidate, source, ctx, avoid_parts("location", source)) else None
        existing = ctx.get_token("LOCATION", source)
        if existing is not None:
            return existing if existing in PREFECTURES else None
        saved_pref = _saved_city_prefecture(source, ctx)
        if saved_pref in PREFECTURES and saved_pref != source:
            return saved_pref
        used = {_stored(op, ctx, "pref:" + p) or ctx.get_token("LOCATION", p)
                or _saved_city_prefecture(p, ctx)
                for p in PREFECTURES if p != source}
        candidates = _RNG.sample(choices, len(choices))
        if getattr(ctx, "text", None):
            other = [p for p in PREFECTURES if p not in choices]
            candidates += _RNG.sample(other, len(other))
        for candidate in candidates:
            if candidate not in used and _ok(op, candidate, source, ctx, avoid_parts("location", source)):
                return candidate
        return None
    return _mapping(op, ctx, "pref:" + source, draw)


def _city_key(source):
    return "city:" + source.prefecture + source.name


def _city_ok(op, source, candidate, ctx):
    parts = tuple(s[:-1] for s in {source.name, source.short, source.parent} if len(s) >= 3)
    text = getattr(ctx, "text", None)
    cores = (candidate.short[:-1], candidate.parent[:-1])
    if text and any(len(core) >= 2 and core in op.normalizer(text) for core in cores):
        return False
    forms = {candidate.name, candidate.short, candidate.office, candidate.station_stem + "駅",
             candidate.station_stem + "駅前"}
    return all(_ok(op, f, source.name, ctx, parts) for f in forms)


def _ordered_candidates(data, source, pref, rng):
    # 代替種別は同種別の後ろへ。種別をまたいでシャッフルしない。
    choices = data.candidates(source, pref, source.name)
    kinds = dict.fromkeys(r.kind for r in choices)
    return [r for kind in kinds for r in rng.sample(
        [r for r in choices if r.kind == kind], sum(r.kind == kind for r in choices))]


def _city_assignments(op, source_pref, pref, ctx):
    """全元地名を鍵順で割り当て、呼出順には依存させない。

    政令市の代替先を通常市より先に予約する。候補不足時はまず他種別との
    重複を許し、同種別内の単射性を優先する。それでも足りなければ再利用。
    """
    data = load_municipalities()
    result, used = {}, set()
    for kind in ("政令市", "市", "特別区", "政令市の区", "村", "町"):
        sources = [r for r in data.by_prefecture[source_pref] if r.kind == kind]
        rng = op._hmac_rng(ctx, "LOCATION", "city-order:" + source_pref + ":" + kind)
        sources = rng.sample(sources, len(sources))
        own_used = set()
        for rank, source in enumerate(sources):
            rng = op._hmac_rng(ctx, "LOCATION", "city-target:" + source_pref + ":" + kind + ":" + pref)
            choices = _ordered_candidates(data, source, pref, rng)
            if not choices:
                result[source] = None
                continue
            start = rank % len(choices)
            choices = choices[start:] + choices[:start]
            # フィルタ除外で次へ進む場合も、先行順位の割当先は飛ばす。
            safe = None
            for reserved in (used, own_used, set()):
                for candidate in choices:
                    if candidate not in reserved and _city_ok(op, source, candidate, ctx):
                        safe = candidate
                        break
                if safe is not None:
                    break
            result[source] = safe
            if safe is not None:
                used.add(safe)
                own_used.add(safe)
    return result


def _stored_city(op, source, pref, ctx):
    data = load_municipalities()
    name = _stored(op, ctx, _city_key(source))
    if name is None:
        for spelling in (source.prefecture + source.name, source.name, source.short):
            name = ctx.get_token("LOCATION", spelling)
            if name is not None:
                break
    if name and name.startswith(pref):
        name = name[len(pref):]
    return data.resolve(name, prefecture=pref) if name else None


def municipality_for(op, source, ctx):
    data = load_municipalities()
    pref = prefecture_for(op, source.prefecture, ctx)
    if pref is None:
        return None
    if op.strategy == "hmac":
        return _city_assignments(op, source.prefecture, pref, ctx).get(source)

    def draw():
        existing = _stored_city(op, source, pref, ctx)
        if existing:
            return existing.name
        used = {_stored_city(op, r, pref, ctx) for r in data.by_prefecture[source.prefecture]
                if r != source}
        choices = _ordered_candidates(data, source, pref, _RNG)
        for candidate in choices:
            if candidate not in used and _city_ok(op, source, candidate, ctx):
                return candidate.name
        return None
    name = _mapping(op, ctx, _city_key(source), draw)
    return data.resolve(name, prefecture=pref) if name else None


def plan(op, kind, norm, ctx):
    """対象外なら None、対象だが対応不能なら空の replacement（トークン退避）。"""
    if kind not in ("location", "address", "facility"):
        return None
    data = load_municipalities()
    text = getattr(ctx, "text", None)
    if kind in ("location", "address"):
        if norm in PREFECTURES:
            return GeoPlan(prefecture_for(op, norm, ctx) or "")
        pref, spelling, source, rest = data.split(norm, text)
        if source and (kind == "address" or not rest):
            target = municipality_for(op, source, ctx)
            if target is None:
                return GeoPlan("")
            # 県なし住所に県を補う既存挙動を維持する。
            prefix = target.prefecture if pref or kind == "address" else ""
            return GeoPlan(prefix + target.render(source, spelling), kind == "address")
    if kind == "facility":
        for ending in ("役所", "役場"):
            if norm.endswith(ending):
                core = norm[:-len(ending)]
                source = data.resolve(core, text)
                if source:
                    target = municipality_for(op, source, ctx)
                    return GeoPlan(target.office if target else "")
                # 種別付き役所語（仙台市役所）は上で市名を保持して分解済み。
                return None
    core = None
    if kind == "facility":
        word, station_core, _attrs = facility_shape(norm)
        if word == "駅":
            core = station_core
    elif kind == "location" and norm.endswith("駅前"):
        core = norm[:-2]
    if core:
        source = data.resolve(core, text, stem=True)
        if source is None or len(source.station_stem) < 2:
            return None
        target = municipality_for(op, source, ctx)
        if target is None:
            return GeoPlan("")
        if len(target.station_stem) < 2:
            return None
        if kind == "facility":
            return GeoPlan(target.station_stem + "駅", facility_affix=True)
        return GeoPlan(target.station_stem + "駅前")
    return None

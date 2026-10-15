# privacy_masker/operators/surrogate_generators.py
"""カテゴリ別の偽名生成器（設計 §6）。

入力は ``Features``（閉じた語彙の表層特徴）と ``rng`` だけで、**元の値は受け取らない**。
実在の地名を使うのは LOCATION と ADDRESS の都道府県・市区町村・町域だけ（同じ地方ブロックの別の地名）。
ORG・FACILITY・PERSON・番地・建物はすべて架空。実在の公知の実体との一致は ``SurrogateFilter`` が同梱の禁止リストで弾く。
``enumerate_candidates`` はその禁止リストを作るための候補全体を列挙する（生成器が出しうる文字列はすべてここに含める）。
"""
from __future__ import annotations

import random
import string
from itertools import product
from typing import Iterator

from ..data.ja_pools import (
    BUILDING_A, BUILDING_B, COUNTRIES, DEPT_FUNCTIONS, DEPT_UNITS, DISTRICT_WORDS, FACILITY_WORDS, FOREIGN_CITIES_KATA,
    GIVEN_HIRA, GIVEN_KANJI, GIVEN_KANJI_F, GIVEN_KANJI_M, GIVEN_KATA, GIVEN_KATA_JA, GIVEN_ROMAJI, KATA_CORES,
    NATURE_WORDS, NICKNAMES_KATA, ORG_ASSOCIATION_WORDS, ORG_CORES, ORG_DIRECTIONS, ORG_GOV_WORDS, ORG_INDUSTRIES,
    ORG_INDUSTRY_WORDS, ORG_KATA_PARTS, ORG_MEDIA_WORDS, ORG_PARTY_WORDS, PREFECTURES, PREF_CITIES, PREF_CITIES_EXTRA,
    REGION_BLOCKS, SCHOOL_PREFIXES, SURNAMES_HIRA, SURNAMES_KANJI, SURNAMES_KATA, SURNAMES_KATA_JA, SURNAMES_ROMAJI, TOWNS,
)
from .semantic import Features, LEGAL_FORMS, given_gender, to_halfwidth_kana

# PREF_CITIES_EXTRA（北海道の追加市。semantic 経路専用）も選択プールに含める。
# hmac の既存出力に影響する PREF_CITIES 自体（既存タプル）は変えない
_CITIES_BY_BLOCK: dict[str, tuple[tuple[str, str], ...]] = {
    block: tuple((p, c) for p, c in PREF_CITIES + PREF_CITIES_EXTRA if p in prefs) for block, prefs in REGION_BLOCKS.items()
}
_TO_FULLWIDTH = str.maketrans("0123456789-", "０１２３４５６７８９－")
_SCHOOL_WORDS = ("小学校", "中学校", "高等学校")
_ABBR_LEN = 3
_KANJI_NUM = "一二三四五"


def _chome(f: Features, rng: random.Random) -> str:
    """f.chome_style のときだけ丁目の文字列を作る（乱数の消費順は従来と同じ）。"""
    if not f.chome_style:
        return ""
    c = rng.randint(1, 5)                       # 乱数の消費順は従来と同じ
    return f"{_KANJI_NUM[c - 1]}丁目" if f.chome_kanji else f"{c}丁目"


# "(株)"/"（株）" は _with_legal_form が常に「（株）+ body」の前置1形だけにするため、
# 列挙側でも前置/後置の両方向を作る対象からは除く（他の法人格はすべて pre/post どちらもありうる）
_LEGAL_FORMS_DIRECTIONAL = tuple(f for f in LEGAL_FORMS if f not in ("(株)", "（株）"))


def _yield_org_forms(body: str) -> Iterator[str]:
    """法人格なし・(株)/（株）・その他の法人格の前置/後置のすべてを列挙する（enumerate_candidates 用の共通処理）。"""
    yield body
    yield "（株）" + body
    for form in _LEGAL_FORMS_DIRECTIONAL:
        yield form + body
        yield body + form


def _pref_city(rng: random.Random, block: str | None) -> tuple[str, str]:
    pairs = _CITIES_BY_BLOCK.get(block or "") or PREF_CITIES
    return rng.choice(pairs)


def _given(pool: tuple[str, ...], f: Features, rng: random.Random) -> str:
    """名のプール。性別は semantic.given_gender（偶奇規約 + GENDER_OVERRIDES）で引く。"""
    if f.gender_hint in ("m", "f"):
        return rng.choice([g for g in pool if given_gender(g) == f.gender_hint])
    return rng.choice(pool)


def _given_kanji(f: Features, rng: random.Random) -> str:
    if f.gender_hint == "m":
        return rng.choice(GIVEN_KANJI_M)
    if f.gender_hint == "f":
        return rng.choice(GIVEN_KANJI_F)
    return rng.choice(GIVEN_KANJI)


def gen_person(f: Features, rng: random.Random) -> str:
    if f.script == "halfwidth":
        sei = rng.choice(SURNAMES_KATA_JA)
        s = sei if f.structure != "full" else sei + (f.separator or " ") + _given(GIVEN_KATA_JA, f, rng)
        return to_halfwidth_kana(s)
    if f.script == "kata":
        if f.kata_foreign:
            if f.structure == "full":
                return rng.choice(GIVEN_KATA) + f.separator + rng.choice(SURNAMES_KATA)
            return rng.choice(GIVEN_KATA + SURNAMES_KATA)
        if f.structure == "full":
            return rng.choice(SURNAMES_KATA_JA) + f.separator + _given(GIVEN_KATA_JA, f, rng)
        return rng.choice(SURNAMES_KATA_JA + NICKNAMES_KATA)          # 姓だけ・あだ名
    if f.script == "hira":
        if f.structure == "full":
            return rng.choice(SURNAMES_HIRA) + f.separator + _given(GIVEN_HIRA, f, rng)
        return rng.choice(SURNAMES_HIRA + GIVEN_HIRA)
    if f.script in ("romaji", "latin_abbr"):
        if f.structure == "full":
            return _given(GIVEN_ROMAJI, f, rng) + " " + rng.choice(SURNAMES_ROMAJI)
        return rng.choice(SURNAMES_ROMAJI)
    sei = rng.choice(SURNAMES_KANJI)
    if f.structure in ("sei", "mono"):
        return sei
    return sei + f.separator + _given_kanji(f, rng)


def _with_legal_form(core: str, f: Features) -> str:
    if f.legal_form is None:
        return core
    if f.legal_form in ("(株)", "（株）"):
        return "（株）" + core
    return core + f.legal_form if f.legal_form_pos == "post" else f.legal_form + core


def gen_org(f: Features, rng: random.Random) -> str:
    cat = f.category
    if cat == "department":
        return rng.choice(DEPT_FUNCTIONS) + rng.choice(DEPT_UNITS)
    if cat == "medical_corp":
        form = f.legal_form if f.legal_form in ("医療法人", "医療法人社団", "社会福祉法人") else "医療法人"
        return form + rng.choice(ORG_CORES) + "会"
    if cat == "school_corp":
        return "学校法人" + rng.choice(ORG_CORES) + "学園"
    if cat == "government":
        return rng.choice(ORG_CORES) + rng.choice(ORG_GOV_WORDS)
    if cat == "association":
        return rng.choice(ORG_CORES) + rng.choice(ORG_ASSOCIATION_WORDS)
    if cat == "media":
        return rng.choice(ORG_CORES) + rng.choice(ORG_MEDIA_WORDS)
    if cat == "party":
        return rng.choice(ORG_CORES) + rng.choice(ORG_PARTY_WORDS)
    words = ORG_INDUSTRY_WORDS.get(cat, ORG_INDUSTRIES)
    if f.script == "latin_abbr":
        core = "".join(rng.choice(string.ascii_uppercase) for _ in range(_ABBR_LEN))
    elif f.script == "kata":
        core = "".join(rng.sample(ORG_KATA_PARTS, 2)) + (rng.choice(words) if cat in ("it", "transport") else "")
    elif cat == "finance":
        core = rng.choice(ORG_DIRECTIONS) + rng.choice(ORG_CORES) + rng.choice(words)   # 実在名の空間が密なので方角で架空を明示
    else:
        core = rng.choice(ORG_CORES) + rng.choice(words)
    return _with_legal_form(core, f)


def _core(f: Features, rng: random.Random) -> str:
    return rng.choice(KATA_CORES) if f.script == "kata" else rng.choice(ORG_CORES)


def gen_facility(f: Features, rng: random.Random) -> str:
    cat = f.category if f.category in FACILITY_WORDS else "other"
    word = rng.choice(FACILITY_WORDS[cat])
    if cat == "education" and word in _SCHOOL_WORDS:
        return (f.org_prefix or rng.choice(SCHOOL_PREFIXES)) + rng.choice(ORG_CORES + TOWNS) + word
    if cat in ("commerce", "building"):
        return rng.choice(KATA_CORES) + word
    if cat in ("transport", "public"):
        return rng.choice(ORG_CORES + TOWNS) + word
    return _core(f, rng) + word


def gen_location(f: Features, rng: random.Random) -> str:
    g = f.category
    if g == "country":
        return rng.choice(COUNTRIES)
    if g == "prefecture":
        # 北海道はブロックに 1 つしか無く「同ブロックの別の都道府県」が作れないので東北で代替する（設計 §6 の例外）
        prefs = REGION_BLOCKS.get(f.region_block or "") or PREFECTURES
        if len(prefs) == 1:
            prefs = REGION_BLOCKS["東北"]
        return rng.choice(prefs)
    if g == "city":
        return _pref_city(rng, f.region_block)[1]
    if g == "town":
        return rng.choice(TOWNS) + _chome(f, rng)
    if g == "district":
        return rng.choice(TOWNS) + rng.choice(DISTRICT_WORDS)
    if g == "foreign":
        return rng.choice(FOREIGN_CITIES_KATA)
    if g == "nature":
        return rng.choice(ORG_CORES) + rng.choice(NATURE_WORDS)
    return rng.choice(TOWNS)


def _pref_only(rng: random.Random, block: str | None) -> str:
    # 北海道はブロックに 1 つしか無く「同ブロックの別の都道府県」が作れないので東北で代替する（gen_location と同じ例外）
    prefs = REGION_BLOCKS.get(block or "") or PREFECTURES
    if len(prefs) == 1:
        prefs = REGION_BLOCKS["東北"]
    return rng.choice(prefs)


def gen_address(f: Features, rng: random.Random) -> str:
    """f.granularity（pref/city/town/banchi/building）までの粒度だけを生成する。

    元の入力が都道府県だけなら偽住所も都道府県だけ（番地や町名を捏造しない）。
    市区町村までなら pref+city まで、というように「元の情報量を超えて詳細化しない」ことで、
    SurrogateFilter の avoid_parts（元の市区町村・町名）と組み合わせて元の値の混入を防ぐ。
    """
    if f.granularity == "pref":
        return _pref_only(rng, f.region_block)
    pref, city = _pref_city(rng, f.region_block)
    if f.granularity == "city":
        return pref + city
    town = rng.choice(TOWNS)
    if f.granularity == "town":
        addr = pref + city + town + _chome(f, rng)
        return addr.translate(_TO_FULLWIDTH) if f.fullwidth_digits else addr
    c, b, g = rng.randint(1, 5), rng.randint(1, 30), rng.randint(1, 20)
    c_disp = _KANJI_NUM[c - 1] if f.chome_kanji else str(c)
    num = f"{c_disp}丁目{b}番{g}号" if f.chome_style else f"{c}-{b}-{g}"
    addr = f"{pref}{city}{town}{num}"
    if f.has_building:
        addr += rng.choice(BUILDING_A) + rng.choice(BUILDING_B) + f"{rng.randint(2, 12)}0{rng.randint(1, 9)}号室"
    return addr.translate(_TO_FULLWIDTH) if f.fullwidth_digits else addr


_DISPATCH = {"person": gen_person, "org": gen_org, "facility": gen_facility, "location": gen_location, "address": gen_address}


def generate(f: Features, rng: random.Random) -> str:
    return _DISPATCH[f.kind](f, rng)


def enumerate_candidates() -> Iterator[str]:
    """禁止リスト作成用: 生成器が出しうる架空の名前をすべて列挙する（ADDRESS と実在地名は含めない）。"""
    # PERSON
    for sei in SURNAMES_KANJI:
        yield sei
        for mei in GIVEN_KANJI:
            yield sei + mei
            yield sei + " " + mei
            yield sei + "・" + mei          # gen_person は kanji full で任意の区切り（""/" "/"・"）をそのまま使う
    for g, s in product(GIVEN_KATA, SURNAMES_KATA):
        yield g + "・" + s
        yield g + " " + s                  # kata_foreign=True・full 構造で separator=" " のとき
        yield g + s                        # kata_foreign=True・full 構造で separator="" のとき（区切り無しの外国人名）
    yield from GIVEN_KATA + SURNAMES_KATA + SURNAMES_KATA_JA + NICKNAMES_KATA + SURNAMES_HIRA + GIVEN_HIRA + SURNAMES_ROMAJI
    for s, m in product(SURNAMES_KATA_JA, GIVEN_KATA_JA):
        yield s + m
        yield s + " " + m
        yield s + "・" + m                  # kata（日本語順）full で separator="・" のとき
        yield to_halfwidth_kana(s + " " + m)
        yield to_halfwidth_kana(s + "・" + m)
        yield to_halfwidth_kana(s)
    for s, m in product(SURNAMES_HIRA, GIVEN_HIRA):
        yield s + m
        yield s + " " + m
        yield s + "・" + m                  # hira full で separator="・" のとき
    for g, s in product(GIVEN_ROMAJI, SURNAMES_ROMAJI):
        yield g + " " + s
    # ORG
    words_all = sorted(set(ORG_INDUSTRIES) | {w for ws in ORG_INDUSTRY_WORDS.values() for w in ws})
    for core in ORG_CORES:
        for w in words_all:
            yield from _yield_org_forms(core + w)
        for d, w in product(ORG_DIRECTIONS, ORG_INDUSTRY_WORDS["finance"]):
            yield from _yield_org_forms(d + core + w)
        for w in ORG_ASSOCIATION_WORDS + ORG_MEDIA_WORDS + ORG_PARTY_WORDS + ORG_GOV_WORDS:
            yield core + w
        yield "医療法人" + core + "会"
        yield "医療法人社団" + core + "会"
        yield "社会福祉法人" + core + "会"
        yield "学校法人" + core + "学園"
    kata_words = ("",) + ORG_INDUSTRY_WORDS["it"] + ORG_INDUSTRY_WORDS["transport"]
    for a, b in product(ORG_KATA_PARTS, ORG_KATA_PARTS):
        if a == b:
            continue
        for w in kata_words:
            yield from _yield_org_forms(a + b + w)
    for abbr in product(string.ascii_uppercase, repeat=_ABBR_LEN):
        yield from _yield_org_forms("".join(abbr))
    for fn, u in product(DEPT_FUNCTIONS, DEPT_UNITS):
        yield fn + u
    # FACILITY
    fac_words = sorted({w for ws in FACILITY_WORDS.values() for w in ws})
    for p in ORG_CORES + TOWNS + KATA_CORES:
        for w in fac_words:
            yield p + w
    for sp, p, w in product(SCHOOL_PREFIXES, ORG_CORES + TOWNS, _SCHOOL_WORDS):
        yield sp + p + w
    for a, b in product(BUILDING_A, BUILDING_B):
        yield a + b
    # LOCATION（架空のものだけ）
    for c, w in product(ORG_CORES, NATURE_WORDS):
        yield c + w
    for t, w in product(TOWNS, DISTRICT_WORDS):
        yield t + w
        yield t

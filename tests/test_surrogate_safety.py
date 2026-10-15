# tests/test_surrogate_safety.py
"""設計 §11 の検証。A: 逆算不能、B: 公知の実体、C: 粒度、D: 一貫性、生成器の隔離。表層はすべて架空。"""
import dataclasses
import random

import pytest

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.data import ja_pools as jp
from privacy_masker.operators import Surrogate
from privacy_masker.operators import semantic as sm
from privacy_masker.operators.surrogate_generators import generate
from privacy_masker.types import Entity

from .conftest import SurrogateCtx as Ctx

E_ORG = Entity("ORG", 0, 8, 1.0)
E_FAC = Entity("FACILITY", 0, 6, 1.0)
E_PER = Entity("PERSON", 0, 4, 1.0)
E_LOC = Entity("LOCATION", 0, 3, 1.0)

# 架空の原文（プールの語と重ならない造語）
CORES = ("大江戸", "神楽坂", "百舌鳥", "勘解由", "月影", "朧月", "雷鳴", "夕凪", "雲雀", "翠嵐")
ORIG_ORG = [f"株式会社{c}{w}" for c in CORES for w in ("精密", "電子", "重工", "薬品")][:200]
ORIG_FAC = [f"{c}{w}" for c in CORES for w in ("記念病院", "中央クリニック", "第一小学校", "総合体育館")][:200]
ORIG_PER = [f"{s}{g}" for s in ("勘解由小路", "朧月", "白雨", "雲雀", "翠嵐", "月影", "雷鳴", "百舌鳥", "神楽坂", "大江戸")
            for g in ("麗華", "宗一郎", "美琴", "颯馬", "琥珀", "紬希", "蒼真", "凛音", "悠仁", "陽葵")][:100]
ALLOWED_SHARED = set(sm.LEGAL_FORMS) | {w for ws in jp.ORG_INDUSTRY_WORDS.values() for w in ws} | set(jp.ORG_INDUSTRIES) \
    | {w for ws in jp.FACILITY_WORDS.values() for w in ws} | set(jp.PREFECTURES) | set(jp.ORG_DIRECTIONS) | set(jp.SCHOOL_PREFIXES) \
    | {"総合", "中央", "第一", "記念", "会社", "式会", "株式"}


def shared_substrings(a: str, b: str, min_len: int = 2) -> set[str]:
    out = set()
    for i in range(len(a) - min_len + 1):
        s = a[i:i + min_len]
        if s in b:
            out.add(s)
    return out


def _surrogate(kind, key):
    return Surrogate(kind=kind, strategy="hmac", key=key)


@pytest.mark.parametrize("kind, originals, ent", [("org", ORIG_ORG, E_ORG), ("facility", ORIG_FAC, E_FAC), ("person", ORIG_PER, E_PER)])
def test_A1_no_original_substrings_leak(kind, originals, ent):
    op = _surrogate(kind, b"key-A")
    ctx = Ctx(b"key-A")
    for o in originals:
        fake = op.apply(o, ent, ctx)
        leaked = {s for s in shared_substrings(o, fake) if not any(s in w for w in ALLOWED_SHARED)}
        assert not leaked, (o, fake, leaked)


@pytest.mark.parametrize("kind, originals, ent", [("org", ORIG_ORG, E_ORG), ("person", ORIG_PER, E_PER)])
def test_A2_different_keys_give_unrelated_surrogates(kind, originals, ent):
    a = [_surrogate(kind, b"key-1").apply(o, ent, Ctx(b"key-1")) for o in originals]
    b = [_surrogate(kind, b"key-2").apply(o, ent, Ctx(b"key-2")) for o in originals]
    same = sum(x == y for x, y in zip(a, b))
    assert same / len(originals) <= 0.05, same


def test_A3_bruteforce_inversion_without_key_is_at_chance():
    """攻撃者はコード・プール・原文の候補集合を持つが鍵を持たない。自分の鍵で写像を作って逆引きしても当たらない。"""
    truth = {o: _surrogate("org", b"secret").apply(o, E_ORG, Ctx(b"secret")) for o in ORIG_ORG}
    attacker = {}
    for o in ORIG_ORG:
        attacker.setdefault(_surrogate("org", b"guess").apply(o, E_ORG, Ctx(b"guess")), o)
    correct = sum(1 for o, fake in truth.items() if attacker.get(fake) == o)
    assert correct / len(ORIG_ORG) <= 2 / 40 + 0.02      # 製造カテゴリの候補空間（コア 15 × 語 8 = 120 通り以上）に対する偶然一致の上限


def test_B_public_entities_are_rejected_via_knowledge():
    famous = PublicKnowledge.from_iterable(["曙電機", "株式会社曙電機", "若葉学園", "八雲病院", "青葉総合病院", "汐見クリニック"])
    op = Surrogate(kind="org", key=b"k", knowledge=famous)
    for o in ORIG_ORG[:50]:
        assert op.apply(o, E_ORG, Ctx(b"k")) not in famous
    fac = Surrogate(kind="facility", key=b"k", knowledge=famous)
    for o in ORIG_FAC[:50]:
        assert fac.apply(o, E_FAC, Ctx(b"k")) not in famous


def test_B_packaged_denylist_is_applied_by_default_facility():
    from privacy_masker.operators.surrogate_filter import load_packaged_denylist
    deny = load_packaged_denylist()
    op2 = Surrogate(kind="facility", key=b"k")
    assert not any(op2.apply(o, E_FAC, Ctx(b"k")) in deny for o in ORIG_FAC)


def test_B_packaged_denylist_is_applied_by_default_person():
    from privacy_masker.operators.surrogate_filter import load_packaged_denylist
    deny = load_packaged_denylist()
    op = Surrogate(kind="person", key=b"k")
    outs = [op.apply(o, E_PER, Ctx(b"k")) for o in ORIG_PER]
    assert not any(o in deny for o in outs)


def test_public_check_applies_person_is_structural_not_length_based():
    """Fix round 1: _public_check_applies は候補の文字数ではなく構造（姓だけ/名だけ か フルネームか）で
    判定する。姓だけ・名だけは短くても公知照合を掛けず、フルネームは短くても（3字でも）掛ける。"""
    from privacy_masker.operators.surrogate import _public_check_applies

    bare_sei = sm.Features(kind="person", category="person", script="kanji", length_class="short", structure="sei")
    bare_mono = sm.Features(kind="person", category="person", script="kanji", length_class="short", structure="mono")
    full_short = sm.Features(kind="person", category="person", script="kanji", length_class="short", structure="full")
    assert _public_check_applies("person", bare_sei.structure) is False
    assert _public_check_applies("person", bare_mono.structure) is False
    assert _public_check_applies("person", full_short.structure) is True   # 3字のフルネームでも文字数に関係なく掛かる
    assert _public_check_applies("org") is True and _public_check_applies("facility") is True
    assert _public_check_applies("location") is False and _public_check_applies("address") is False


def test_C_granularity_and_region_block_preserved():
    loc = Surrogate(kind="location", key=b"k")
    for pref in jp.PREFECTURES:
        out = loc.apply(pref, E_LOC, Ctx(b"k"))
        expected_block = "東北" if pref == "北海道" else sm.region_block_of(pref)      # 北海道だけ東北で代替
        assert out != pref and sm.region_block_of(out) == expected_block, (pref, out)
    # PREF_CITIES_EXTRA（北海道の追加市。semantic 経路専用）も選択肢に入る
    known_cities = {c for _, c in jp.PREF_CITIES + jp.PREF_CITIES_EXTRA}
    for pref, city in jp.PREF_CITIES:
        out = loc.apply(city, Entity("LOCATION", 0, len(city), 1.0), Ctx(b"k"))
        assert out != city and out in known_cities and sm.region_block_of(out) == sm.region_block_of(pref), (city, out)
    fac = Surrogate(kind="facility", key=b"k")
    assert fac.apply("大江戸記念病院", E_FAC, Ctx(b"k")).endswith(jp.FACILITY_WORDS["medical"])
    assert fac.apply("神楽坂第一小学校", E_FAC, Ctx(b"k")).endswith(jp.FACILITY_WORDS["education"])
    org = Surrogate(kind="org", key=b"k")
    assert org.apply("株式会社月影電子", E_ORG, Ctx(b"k")).endswith(jp.ORG_INDUSTRY_WORDS["manufacturing"])
    assert org.apply("総務企画部", Entity("ORG", 0, 5, 1.0), Ctx(b"k")).endswith(jp.DEPT_UNITS)


def test_D_hmac_is_consistent_within_a_scope_across_documents():
    """README（『同一key(+scope)で常に同値』）どおり、hmac 戦略の一貫性は scope_id 込みで決まる
    （scope_id は Tokenize(strategy="hmac") と同じ _HmacMixin.hmac_token を経由してハッシュ入力に
    含まれる）。同一 scope なら文書が変わっても同じ偽名、scope が変われば偽名が変わってよい
    （変わらないことまでは保証しない。両方とも架空の偽名であることだけ見る）。"""
    op = Surrogate(kind="org", key=b"k")
    a = op.apply("株式会社月影電子", E_ORG, Ctx(b"k", scope_id="s1", text="月影電子の議事録。"))
    b = op.apply("株式会社月影電子", E_ORG, Ctx(b"k", scope_id="s1", text="別の文書。"))
    assert a == b                            # 同一 scope・同一鍵・別文書 → 同じ偽名

    # scope_id が違えば偽名が変わってよい（一致しないことは保証しない。ここでは架空の偽名で
    # あることだけ確認する: legal_form・カテゴリを保ち、元のコア語を含まない）
    c = op.apply("株式会社月影電子", E_ORG, Ctx(b"k", scope_id="s2", text="別の文書。"))
    assert c.startswith("株式会社") and c.endswith(jp.ORG_INDUSTRY_WORDS["manufacturing"]) and "月影" not in c

    # 文書に偽名が既出のときだけ変わる（同一 scope 内でも present_in_document で引き直す。設計 §7）
    d = op.apply("株式会社月影電子", E_ORG, Ctx(b"k", scope_id="s1", text=f"{a}は別会社。"))
    assert d != a and "月影" not in d


def test_D_random_strategy_is_consistent_within_scope():
    rnd = Surrogate(kind="org", strategy="random")
    ctx = Ctx(b"k")
    assert rnd.apply("株式会社月影電子", E_ORG, ctx) == rnd.apply("株式会社月影電子", E_ORG, ctx)
    assert rnd.apply("株式会社月影電子", E_ORG, ctx) != rnd.apply("株式会社雷鳴電子", E_ORG, ctx)


def test_generator_isolation_features_have_no_original_and_fallback_never_returns_original():
    f = sm.extract_features("org", "株式会社月影電子", "株式会社月影電子", "manufacturing")
    for fld in dataclasses.fields(f):
        v = getattr(f, fld.name)
        assert not (isinstance(v, str) and "月影" in v)
    assert "月影" not in generate(f, random.Random(0))

    class RejectAll:
        def reject_reason(self, candidate, **kw):
            return "denylist"

    op = Surrogate(kind="auto", key=b"k")
    op._filter = RejectAll()
    for value, ent in [("株式会社月影電子", E_ORG), ("勘解由小路麗華", Entity("PERSON", 0, 7, 1.0)), ("大江戸記念病院", E_FAC)]:
        out = op.apply(value, ent, Ctx(b"k"))
        assert out.startswith(f"<{ent.entity_type}_") and value not in out and value[:2] not in out

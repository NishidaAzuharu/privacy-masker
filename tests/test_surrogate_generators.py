# tests/test_surrogate_generators.py
"""カテゴリ別の生成器（privacy_masker/operators/surrogate_generators.py）。出力がカテゴリ・表層特徴に従うこと。"""
import random
import re

import pytest

from privacy_masker.data import ja_pools as jp
from privacy_masker.operators import semantic as sm
from privacy_masker.operators import surrogate_generators as sg

R = lambda seed=0: random.Random(seed)  # noqa: E731


def F(kind, category, **kw):
    kw.setdefault("script", "kanji"); kw.setdefault("length_class", "medium")
    return sm.Features(kind, category, kw.pop("script"), kw.pop("length_class"), **kw)


@pytest.mark.parametrize("cat, words", [
    ("manufacturing", jp.ORG_INDUSTRY_WORDS["manufacturing"]), ("finance", jp.ORG_INDUSTRY_WORDS["finance"]),
    ("association", jp.ORG_ASSOCIATION_WORDS), ("media", jp.ORG_MEDIA_WORDS), ("party", jp.ORG_PARTY_WORDS),
    ("government", jp.ORG_GOV_WORDS), ("service", jp.ORG_INDUSTRIES),
])
def test_org_output_ends_with_category_word(cat, words):
    for seed in range(20):
        out = sg.generate(F("org", cat), R(seed))
        assert out.endswith(words), (cat, out)


def test_org_legal_form_position_and_special_categories():
    assert sg.generate(F("org", "manufacturing", legal_form="株式会社", legal_form_pos="pre"), R(1)).startswith("株式会社")
    assert sg.generate(F("org", "manufacturing", legal_form="株式会社", legal_form_pos="post"), R(1)).endswith("株式会社")
    assert sg.generate(F("org", "retail", legal_form="(株)", legal_form_pos="pre"), R(1)).startswith("（株）")
    assert sg.generate(F("org", "medical_corp", legal_form="医療法人"), R(2)).startswith("医療法人") and sg.generate(F("org", "medical_corp"), R(2)).endswith("会")
    assert sg.generate(F("org", "school_corp"), R(2)).startswith("学校法人")
    dept = sg.generate(F("org", "department"), R(3))
    assert dept.endswith(jp.DEPT_UNITS) and dept[:-1].rstrip("グループチーム") != ""
    fin = sg.generate(F("org", "finance"), R(4))
    assert fin.startswith(jp.ORG_DIRECTIONS)                         # 金融は方角プレフィックスで架空を明示
    kata = sg.generate(F("org", "it", script="kata"), R(5))
    assert re.fullmatch(r"[ァ-ヶー・]+", kata), kata
    abbr = sg.generate(F("org", "service", script="latin_abbr"), R(6))
    assert re.fullmatch(r"[A-Z]{3}", abbr), abbr


@pytest.mark.parametrize("cat", list(jp.FACILITY_WORDS))
def test_facility_output_ends_with_category_word(cat):
    for seed in range(10):
        out = sg.generate(F("facility", cat), R(seed))
        assert out.endswith(jp.FACILITY_WORDS[cat]), (cat, out)


def test_facility_school_prefix_and_place_prefix():
    out = sg.generate(F("facility", "education", org_prefix="県立"), R(7))
    if out.endswith(("小学校", "中学校", "高等学校")):
        assert out.startswith("県立")
    for seed in range(10):
        out = sg.generate(F("facility", "transport", region_block="関東"), R(seed))
        assert out.endswith(jp.FACILITY_WORDS["transport"]) and out.startswith(jp.ORG_CORES + jp.TOWNS)


def test_person_preserves_script_structure_separator_and_gender():
    full_m = sg.generate(F("person", "person", structure="full", gender_hint="m", separator=" "), R(1))
    sei, mei = full_m.split(" ")
    assert sei in jp.SURNAMES_KANJI and mei in jp.GIVEN_KANJI_M
    full_f = sg.generate(F("person", "person", structure="full", gender_hint="f"), R(1))
    assert any(full_f.endswith(m) for m in jp.GIVEN_KANJI_F)
    assert sg.generate(F("person", "person", structure="sei"), R(2)) in jp.SURNAMES_KANJI
    kf = sg.generate(F("person", "person", script="kata", kata_foreign=True, structure="full", separator="・"), R(3))
    g, s = kf.split("・")
    assert g in jp.GIVEN_KATA and s in jp.SURNAMES_KATA
    assert sg.generate(F("person", "person", script="kata", structure="mono"), R(4)) in jp.SURNAMES_KATA_JA + jp.NICKNAMES_KATA
    hira = sg.generate(F("person", "person", script="hira", structure="full"), R(5))
    assert re.fullmatch(r"[ぁ-ゖー]+", hira)
    rom = sg.generate(F("person", "person", script="romaji", structure="full", separator=" "), R(6))
    g, s = rom.split(" ")
    assert g in jp.GIVEN_ROMAJI and s in jp.SURNAMES_ROMAJI
    hw = sg.generate(F("person", "person", script="halfwidth", structure="full", separator=" "), R(7))
    assert re.fullmatch(r"[ｦ-ﾟ ]+", hw), hw


def test_given_uses_gender_overrides_not_parity():
    f_m = F("person", "person", script="kata", structure="full", separator="", gender_hint="m")
    f_f = F("person", "person", script="kata", structure="full", separator="", gender_hint="f")
    outs_f = {sg.generate(f_f, R(s)) for s in range(200)}
    assert not any(o.endswith("カイト") for o in outs_f)                      # 女性寄りの抽選にカイトが出ない
    assert any(o.endswith("カイト") for o in (sg.generate(f_m, R(s)) for s in range(200)))


def test_kata_foreign_full_without_separator():
    f = F("person", "person", script="kata", structure="full", separator="", kata_foreign=True)
    out = sg.generate(f, R(3))
    assert "・" not in out and " " not in out and re.fullmatch(r"[ァ-ヶー]+", out)
    assert out in set(sg.enumerate_candidates())                              # 閉包に含まれる


def test_location_preserves_granularity_and_region_block():
    assert sg.generate(F("location", "country"), R(1)) in jp.COUNTRIES
    for seed in range(20):
        p = sg.generate(F("location", "prefecture", region_block="東北"), R(seed))
        assert p in jp.REGION_BLOCKS["東北"]
        c = sg.generate(F("location", "city", region_block="九州・沖縄"), R(seed))
        assert any(c == city and pref in jp.REGION_BLOCKS["九州・沖縄"] for pref, city in jp.PREF_CITIES)
    assert sg.generate(F("location", "city", region_block=None), R(1)) in {c for _, c in jp.PREF_CITIES}
    assert sg.generate(F("location", "town", chome_style=True), R(2)).endswith("丁目")
    assert sg.generate(F("location", "district"), R(3)).endswith(jp.DISTRICT_WORDS)
    assert sg.generate(F("location", "foreign"), R(4)) in jp.FOREIGN_CITIES_KATA
    assert sg.generate(F("location", "nature"), R(5)).endswith(jp.NATURE_WORDS)
    assert sg.generate(F("location", "other"), R(6)) in jp.TOWNS


def test_address_granularity_pref_stays_prefecture_only():
    # Critical #1: 都道府県だけの入力は都道府県だけの偽住所にする（番地・市区町村を捏造しない）
    f = F("address", "address", granularity="pref", region_block="関東")
    for seed in range(20):
        out = sg.generate(f, R(seed))
        assert out in jp.REGION_BLOCKS["関東"]


def test_address_granularity_city_stays_pref_plus_city_only():
    f = F("address", "address", granularity="city", region_block="九州・沖縄")
    for seed in range(20):
        out = sg.generate(f, R(seed))
        assert any(out == pref + city for pref, city in jp.PREF_CITIES if pref in jp.REGION_BLOCKS["九州・沖縄"])
        assert not re.search(r"[0-9０-９]", out)


def test_address_granularity_town_has_no_banchi_or_building():
    f = F("address", "address", granularity="town", region_block="中部", chome_style=False)
    for seed in range(20):
        out = sg.generate(f, R(seed))
        assert out.endswith(jp.TOWNS)
        assert not any(ch in out for ch in "番号-") and not re.search(r"[0-9０-９]", out)


def test_address_preserves_block_chome_building_and_fullwidth():
    f = F("address", "address", granularity="building", region_block="近畿", has_building=True, chome_style=True)
    out = sg.generate(f, R(1))
    assert out.startswith(jp.REGION_BLOCKS["近畿"]) and "丁目" in out and "号室" in out
    f2 = F("address", "address", granularity="banchi", region_block="中国", fullwidth_digits=True)
    out2 = sg.generate(f2, R(2))
    assert out2.startswith(jp.REGION_BLOCKS["中国"]) and re.search(r"[０-９]", out2) and not re.search(r"[0-9]", out2)


def test_enumerate_candidates_covers_generator_outputs():
    universe = set(sg.enumerate_candidates())
    assert len(universe) >= 20000
    checked = 0
    for seed in range(300):
        rng = R(seed)
        for f in (F("org", "manufacturing", legal_form="株式会社", legal_form_pos="pre"), F("org", "association"), F("org", "it", script="kata"),
                  F("org", "service", script="latin_abbr"), F("facility", "medical"), F("facility", "education", org_prefix="市立"),
                  F("person", "person", structure="full"), F("person", "person", script="hira", structure="full"),
                  F("location", "nature"), F("org", "department")):
            out = sg.generate(f, rng)
            assert out in universe, (f.kind, f.category, out)
            checked += 1
    assert checked == 3000


def test_enumerate_candidates_is_exact_closure_of_generate_outputs():
    """enumerate_candidates() は generate() が出しうる文字列の全体を覆う必要がある（Task 6 review Important #1）。

    Task 7/8 はこの集合から禁止リストを作るため、ここに無い出力は安全性チェックを一切通らずに
    そのまま使われてしまう。PERSON は script×structure×separator×gender_hint×kata_foreign の全組、
    ORG は category×legal_form×legal_form_pos×script の全組、FACILITY は category×org_prefix の全組、
    LOCATION は実在地名を含まないカテゴリ（nature）を総当たりし、生成結果が必ず候補集合に含まれることを見る。
    """
    cands = set(sg.enumerate_candidates())
    assert len(cands) >= 400_000          # 法人格の全展開により Task 6 時点の 20,000 から大きく増える

    # PERSON
    for script in sm.PERSON_SCRIPTS:
        for structure in sm.STRUCTURES:
            for separator in ("", " ", "・"):
                for gender_hint in sm.GENDER_HINTS:
                    for kata_foreign in (False, True):
                        f = F("person", "person", script=script, structure=structure, separator=separator,
                              gender_hint=gender_hint, kata_foreign=kata_foreign)
                        for seed in range(5):
                            out = sg.generate(f, R(seed))
                            assert out in cands, (script, structure, separator, gender_hint, kata_foreign, out)

    # ORG（法人格は None + LEGAL_FORMS 全種、位置は pre/post、スクリプトは分岐のある3種）
    for cat in sm.ORG_CATEGORIES:
        for legal_form in (None,) + sm.LEGAL_FORMS:
            for legal_form_pos in ("pre", "post"):
                for script in ("kanji", "kata", "latin_abbr"):
                    f = F("org", cat, script=script, legal_form=legal_form, legal_form_pos=legal_form_pos)
                    for seed in range(5):
                        out = sg.generate(f, R(seed))
                        assert out in cands, (cat, legal_form, legal_form_pos, script, out)

    # FACILITY（org_prefix の指定あり・なし、_core() が script で KATA_CORES/ORG_CORES を切り替える）
    for cat in sm.FACILITY_CATEGORIES:
        for org_prefix in (None,) + jp.SCHOOL_PREFIXES:
            for script in ("kanji", "kata"):
                f = F("facility", cat, org_prefix=org_prefix, script=script)
                for seed in range(5):
                    out = sg.generate(f, R(seed))
                    assert out in cands, (cat, org_prefix, script, out)

    # LOCATION（実在地名を含まない架空カテゴリのみ。country/prefecture/city/town/district/foreign は対象外）
    for seed in range(5):
        out = sg.generate(F("location", "nature"), R(seed))
        assert out in cands, out


def test_hokkaido_city_fakes_have_five_more_choices():
    """北海道は PREF_CITIES に 2 市しか無く都市レベルの偽名が事実上 1 択だった。
    PREF_CITIES_EXTRA（semantic 経路専用）を足した分だけ選択肢が増える。"""
    outs = {sg.generate(F("location", "city", region_block="北海道"), R(s)) for s in range(300)}
    assert {"函館市", "小樽市", "帯広市", "釧路市", "北見市"} <= outs


def test_chome_kanji_numeral_is_preserved():
    f = F("address", "address", granularity="town", region_block="関東", chome_style=True, chome_kanji=True)
    assert re.search(r"[一二三四五]丁目$", sg.generate(f, R(1)))
    g = F("address", "address", granularity="town", region_block="関東", chome_style=True)
    assert re.search(r"[1-5]丁目$", sg.generate(g, R(1)))

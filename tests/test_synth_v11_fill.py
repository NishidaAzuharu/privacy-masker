"""値の差し込み（tools/synth_v11_fill.py）。位置の一致・同一人物の一貫性・対の負例・ラベルなしスロット。"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import synth_v11_fill as sf  # noqa: E402
import synth_v11_values as vals  # noqa: E402


@pytest.fixture
def pools():
    personas = [{"uuid": "u1", "sei": "雪見沢", "mei": "拓真", "age": 34, "sex": "男", "occupation": "看護師", "prefecture": "大分県", "hobby": "釣り"}]
    ken = [("大分県", "大分市", "府内町"), ("東京都", "新宿区", "西新宿"), ("北海道", "札幌市中央区", "北一条西")]
    return vals.build_pools(personas=personas, ken_rows=ken, excl_person=set(), excl_place=set())


def _spec(slot, type_, num, *, required=True, script="kanji", first="full", rementions=(), form=None, suffix=None, bound=False):
    return {"slot": slot, "type": type_, "num": num, "required": required, "role": "担当者", "script": script, "first": first,
            "rementions": list(rementions), "form": form, "compound_suffix": suffix, "persona_bound": bound}


T = "{PERSON1}様\n\n{ORG1}の{PERSON2}です。{PERSON2_SEI}まで {PHONE1} にご連絡ください。\n住所: {ADDRESS1}\n会場は{LOCATION1}大会の{FACILITY1}です。"
SPECS = [_spec("PERSON1", "PERSON", 1, bound=True), _spec("PERSON2", "PERSON", 2, rementions=["SEI"]),
         _spec("ORG1", "ORG", 1, form="kata_brand"), _spec("PHONE1", "PHONE", 1),
         _spec("ADDRESS1", "ADDRESS", 1, form="pref_first", bound=True), _spec("LOCATION1", "LOCATION", 1, form="compound", suffix="大会"),
         _spec("FACILITY1", "FACILITY", 1)]


def test_parse_slots_positions_and_mods():
    slots = sf.parse_slots(T)
    assert [s.raw for s in slots][:3] == ["{PERSON1}", "{ORG1}", "{PERSON2}"]
    assert slots[3].type == "PERSON" and slots[3].mod == "SEI" and T[slots[3].start:slots[3].end] == "{PERSON2_SEI}"


def test_fill_offsets_match_text_and_labels(pools):
    doc = sf.fill_template(T, SPECS, random.Random(1), pools, persona=pools.personas[0], role_ok=False)
    text, ents = doc["text"], doc["entities"]
    assert "{" not in text and sf.validate_filled(doc, pools) == []
    labels = [l for _, _, l in ents]
    assert labels.count("PERSON") == 3 and "ORG" in labels and "ADDRESS" in labels and "LOCATION" in labels and "FACILITY" in labels
    for s, e, l in ents:
        assert text[s:e] == text[s:e].strip() and s < e
    # ペルソナ束縛: PERSON1 はペルソナの名前、ADDRESS1 は同じ都道府県
    p1 = next(text[s:e] for s, e, l in ents if l == "PERSON" and s == 0)
    assert p1.replace(" ", "").replace("　", "") == "雪見沢拓真"
    addr = next(text[s:e] for s, e, l in ents if l == "ADDRESS")
    assert addr.startswith("大分県")
    # 複合語: ラベルは地名部分だけで、直後に「大会」が続く
    loc_s, loc_e = next((s, e) for s, e, l in ents if l == "LOCATION")
    assert text[loc_e:loc_e + 2] == "大会"
    # 再言及は同じ人物の姓
    p2_full = [text[s:e] for s, e, l in ents if l == "PERSON"][1]
    p2_sei = [text[s:e] for s, e, l in ents if l == "PERSON"][2]
    assert p2_full.replace(" ", "").replace("　", "").startswith(p2_sei)
    # 電話番号はラベルなし
    assert not any(text[s:e].startswith("0") and "-" in text[s:e] for s, e, _ in ents)
    assert doc["persona_uuid"] == "u1" and doc["source"] == sf.SOURCE_NAME


def test_role_substitution_removes_labels_consistently(pools):
    rng = random.Random(0)
    docs = [sf.fill_template(T, SPECS, rng, pools, persona=None, role_ok=True, role_rate=1.0) for _ in range(3)]
    for d in docs:
        assert d["role_substituted"] == 2                     # PERSON1 と PERSON2（再言及込みで同じ役割語）
        assert not any(l == "PERSON" for _, _, l in d["entities"])
        assert any(w in d["text"] for w in vals.ROLE_WORDS)
    no_role = sf.fill_template(T, SPECS, random.Random(0), pools, role_ok=False, role_rate=1.0)
    assert no_role["role_substituted"] == 0


def test_fill_all_is_deterministic_and_dedups(pools):
    templates = [{"template_id": "syn_v11_00001", "template": T, "genre_key": "business", "subgenre_key": "x", "format_key": "prose",
                  "style": "polite", "length_key": "short", "slots": SPECS, "persona": pools.personas[0], "role_ok": False, "prompt_sha256": "a"}]
    a = sf.fill_all(templates, pools, seed=42, fills=2)
    b = sf.fill_all(templates, pools, seed=42, fills=2)
    assert [d["text"] for d in a] == [d["text"] for d in b] and len(a) == 2
    assert a[0]["text"] != a[1]["text"] and {d["fill_index"] for d in a} == {0, 1}
    assert all(d["variant"] == "business/x/prose" for d in a)


def test_validate_filled_flags_leftover_brace_and_edge_space(pools):
    bad = {"text": "山田 さんは{ORG1}へ", "entities": [[0, 3, "PERSON"]], "source": sf.SOURCE_NAME}
    reasons = sf.validate_filled(bad, pools)
    assert "placeholder_left" in reasons and "span_edge_space" in reasons

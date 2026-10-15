"""網羅行列と計画（tools/synth_v11_plan.py）。Drive・ネットワークは読まない。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import synth_v11_plan as sp  # noqa: E402
import synth_v11_values as vals  # noqa: E402

PERSONAS = [{"uuid": f"u{i}", "sei": "雪見沢", "mei": "拓真", "age": 30 + i, "sex": "男", "occupation": "看護師",
             "prefecture": "大分県", "hobby": "釣り"} for i in range(5)]


def test_plan_prefix_keeps_genre_ratio_and_ids_unique():
    rows = sp.build_plan(1000, seed=42, personas=PERSONAS)
    assert len({r["request_id"] for r in rows}) == 1000
    for k in (200, 1000):
        c = Counter(r["genre_key"] for r in rows[:k])
        for g in sp.GENRES:
            assert abs(c[g.key] - k * g.weight) <= 1, (k, g.key, c[g.key])


def test_plan_is_deterministic_and_rows_have_required_keys():
    a = sp.build_plan(60, seed=7, personas=PERSONAS)
    b = sp.build_plan(60, seed=7, personas=PERSONAS)
    assert [r["prompt_sha256"] for r in a] == [r["prompt_sha256"] for r in b]
    required = ("request_id", "index", "genre_key", "subgenre_key", "format_key", "length_key", "max_tokens", "prompt",
                "prompt_sha256", "slots", "persona", "role_ok", "topic", "style", "opening")
    for r in a:
        for k in required:
            assert k in r, k
        assert r["format_key"] in sp.FORMATS and r["style"] in sp.STYLES
        assert r["length_key"] in sp.LENGTH_KEYS and r["max_tokens"] == sp.MAX_TOKENS[r["length_key"]]
        assert r["topic"] in r["prompt"]


def test_slots_are_described_in_prompt_and_persona_name_is_not():
    rows = sp.build_plan(300, seed=3, personas=PERSONAS)
    for r in rows:
        labeled = [s for s in r["slots"] if s["type"] in sp.LABELED_TYPES]
        assert labeled, "ラベル付きスロットが 1 つも無い計画行"
        for s in r["slots"]:
            assert s["slot"] in r["prompt"], (s["slot"], r["prompt"])
            for mod in s.get("rementions", []):
                assert f"{{{s['type']}{s['num']}_{mod}}}" in r["prompt"]
            if s.get("compound_suffix"):
                assert f"{s['slot']}{s['compound_suffix']}" in r["prompt"]
        if r["persona"]:
            assert "雪見沢" not in r["prompt"] and "拓真" not in r["prompt"]
            assert "看護師" in r["prompt"] and "大分県" in r["prompt"]
            p1 = next(s for s in r["slots"] if s["slot"] == "PERSON1")
            assert p1["script"] == "kanji" and p1["persona_bound"]
    assert any(r["persona"] for r in rows) and any(not r["persona"] for r in rows)


def test_person_script_and_mention_ratios_follow_weights():
    rows = sp.build_plan(4000, seed=11, personas=None)
    specs = [s for r in rows for s in r["slots"] if s["type"] == "PERSON"]
    scripts = Counter(s["script"] for s in specs)
    for k, w in vals.PERSON_SCRIPTS.items():
        assert abs(scripts[k] / len(specs) - w) < 0.05, (k, scripts[k] / len(specs), w)
    firsts = Counter(s["first"] for s in specs)
    for k, w in sp.FIRST_MENTION.items():
        assert abs(firsts[k] / len(specs) - w) < 0.05, (k, firsts[k] / len(specs), w)
    locs = [s for r in rows for s in r["slots"] if s["type"] == "LOCATION"]
    forms = Counter(s["form"] for s in locs)
    assert forms["foreign_kata"] / len(locs) >= 0.15 and forms["compound"] / len(locs) >= 0.10
    orgs = [s for r in rows for s in r["slots"] if s["type"] == "ORG"]
    assert sum(1 for s in orgs if s["form"] in ("kata_brand", "quoted_brand", "abbr_latin")) / len(orgs) >= 0.40
    addrs = [s for r in rows for s in r["slots"] if s["type"] == "ADDRESS"]
    assert sum(1 for s in addrs if s["form"] == "banchi_first") / len(addrs) >= 0.20
    assert sum(1 for r in rows if any(s["type"] in sp.UNLABELED_TYPES for s in r["slots"])) / len(rows) >= 0.30


def test_literary_subgenres_do_not_ask_for_romaji_or_halfwidth():
    rows = [r for r in sp.build_plan(3000, seed=5) if r["genre_key"] == "literary"]
    assert rows
    for r in rows:
        for s in r["slots"]:
            if s["type"] == "PERSON":
                assert s["script"] in ("kanji", "hira", "historic", "kata_ja"), s


def test_manifest_carries_prompt_schema_and_realized_ratios():
    rows = sp.build_plan(40, seed=3, personas=PERSONAS)
    m = sp.plan_manifest(rows, 3, "deadbeef", personas_sha256="cafe")
    assert m["system_prompt"] == sp.SYSTEM_PROMPT and m["output_schema"] == sp.OUTPUT_SCHEMA
    assert m["plan_version"] == sp.PLAN_VERSION and m["personas_sha256"] == "cafe"
    assert set(m["person_script_counts"]) <= set(vals.PERSON_SCRIPTS) and m["n"] == 40
    assert m["genre_counts"]["business"] >= 1


def test_slot_regex_accepts_grammar_and_rejects_particles_inside_braces():
    assert sp.SLOT_RE.fullmatch("{PERSON1}") and sp.SLOT_RE.fullmatch("{PERSON12_SEI}") and sp.SLOT_RE.fullmatch("{PHONE1}")
    assert not sp.SLOT_RE.fullmatch("{PERSON1さん}") and not sp.SLOT_RE.fullmatch("{ORG1_SEI}") and not sp.SLOT_RE.fullmatch("{NAME1}")

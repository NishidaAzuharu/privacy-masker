"""テンプレートの受け入れ検査（tools/synth_v11_templates.py）。Drive・NER モデルは読まない。"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import synth_v11_plan as sp  # noqa: E402
import synth_v11_templates as st  # noqa: E402


def _row(slots):
    """テスト用の最小の計画行。slots は build_slot_specs と同じ形の dict の列。"""
    return {"request_id": "syn_v11_00000", "index": 0, "genre_key": "business", "subgenre_key": "meeting_minutes",
            "format_key": "prose", "style": "polite", "length_key": "short", "length_min": 60, "length_max": 150,
            "slots": slots, "persona": None, "role_ok": True, "prompt_sha256": "abc"}


def _spec(slot, type_, num, required=True, rementions=()):
    return {"slot": slot, "type": type_, "num": num, "required": required, "role": "担当者", "script": "kanji",
            "first": "full", "rementions": list(rementions), "form": None, "compound_suffix": None, "persona_bound": False}


GOOD = ("お世話になっております。{ORG1}の{PERSON1}です。\n先日の件、{PERSON1_SEI}までご連絡ください。\n"
        "会場は{FACILITY1}、担当は{PERSON2}さんです。連絡先は{PHONE1}。よろしくお願いいたします。担当者一同、心よりお待ちしております。")
SLOTS = [_spec("PERSON1", "PERSON", 1, rementions=["SEI"]), _spec("PERSON2", "PERSON", 2, required=False),
         _spec("ORG1", "ORG", 1), _spec("FACILITY1", "FACILITY", 1, required=False), _spec("PHONE1", "PHONE", 1, required=False)]


def test_good_template_is_accepted():
    assert st.validate_template(GOOD, _row(SLOTS)) == ([], [])


def test_malformed_and_unknown_slots_are_rejected():
    reasons, _ = st.validate_template(GOOD.replace("{PERSON1}", "{PERSON1さん}"), _row(SLOTS))
    assert "malformed_slot" in reasons
    reasons, _ = st.validate_template(GOOD + " {PERSON9} も出席。", _row(SLOTS))
    assert "unknown_slot" in reasons
    reasons, _ = st.validate_template(GOOD.replace("{PHONE1}", "{PHONE1_SEI}"), _row(SLOTS))
    assert "malformed_slot" in reasons


def test_missing_required_is_soft_and_modifier_only_person_is_allowed():
    """必須スロットの欠落と、姓だけ・あだ名だけで呼ばれる人物は棄却しない（slot_usage で数える）。"""
    no_org = GOOD.replace("{ORG1}の", "")
    assert st.validate_template(no_org, _row(SLOTS)) == ([], [])
    assert st.slot_usage(no_org, SLOTS)["omitted_required"] == ["ORG1"]
    swapped = "{PERSON1_SEI}です。" + GOOD
    assert st.validate_template(swapped, _row(SLOTS)) == ([], [])
    only_sei = GOOD.replace("{PERSON1}", "{PERSON1_SEI}")
    assert st.validate_template(only_sei, _row(SLOTS)) == ([], [])
    assert st.slot_usage(only_sei, SLOTS)["modifier_only_person"] == ["PERSON1"]


def test_augment_specs_adopts_unplanned_slots():
    import random
    t = GOOD + "\n日付: {DATE1}\n住所: {ADDRESS1}\n同席: {PERSON3}（{PERSON3_SEI}）"
    specs, added = st.augment_specs(t, _row(SLOTS), random.Random(0))
    assert added == ["DATE1", "ADDRESS1", "PERSON3"]
    by = {x["slot"]: x for x in specs}
    assert by["ADDRESS1"]["form"] in ("pref_first", "city_first", "banchi_first", "with_building", "fullwidth")
    assert by["PERSON3"]["script"] and by["PERSON3"]["first"] and by["PERSON3"]["role"] and by["PERSON3"]["origin"] == "extra"
    assert not by["DATE1"]["required"]
    assert st.validate_template(t, {**_row(SLOTS), "slots": specs}) == ([], [])


def test_repair_replaces_leaked_values_with_new_slots_but_not_titles():
    import random
    leaked = GOOD + "\n会場は愛知県、窓口は総務部、電話は 03-1234-5678、詳細は https://example.com/x まで。"
    t2, specs, repairs = st.repair_template(leaked, SLOTS, random.Random(0), _row(SLOTS))
    assert "愛知県" not in t2 and "総務部" not in t2 and "03-1234-5678" not in t2 and "example.com" not in t2
    assert "{LOCATION1}" in t2 and "{ORG2}" in t2 and "{PHONE2}" in t2 and "{URL1}" in t2      # 番号は計画の続き
    by = {x["slot"]: x for x in specs}
    assert by["LOCATION1"]["form"] == "prefecture" and by["ORG2"]["form"] == "dept" and by["ORG2"]["origin"] == "repair:department_name"
    assert st.validate_template(t2, {**_row(SLOTS), "slots": specs}) == ([], [])
    assert len(repairs) == 4
    # 肩書きの一部（営業部長）は置き換えない → 棄却のまま
    t3, specs3, repairs3 = st.repair_template(GOOD + " 営業部長の判断です。", SLOTS, random.Random(0), _row(SLOTS))
    assert repairs3 == [] and "pii_outside_slots" in st.validate_template(t3, {**_row(SLOTS), "slots": specs3})[0]


def test_pii_outside_slots_is_rejected():
    reasons, evidence = st.validate_template(GOOD + " 東京都から山田さんも来ます。", _row(SLOTS))
    assert "pii_outside_slots" in reasons and evidence
    reasons, _ = st.validate_template(GOOD + " 連絡は 03-1234-5678 まで。", _row(SLOTS))
    assert "pii_outside_slots" in reasons


def test_length_and_labeled_slot_requirements():
    reasons, _ = st.validate_template("{PERSON1}です。", _row(SLOTS))
    assert "too_short" in reasons
    reasons, _ = st.validate_template("こちらは連絡です。" * 5 + "{PHONE1}", _row([_spec("PHONE1", "PHONE", 1)]))
    assert "no_labeled_slot" in reasons


_FILLERS = [
    "進捗の確認と次回の予定について話し合いました。担当は各自で確認してください。詳細は追って連絡します。",
    "設備点検の結果、異常はありませんでした。次回の点検日程は改めて案内します。報告は以上です。",
    "納期遅れの原因は部材の入荷遅延でした。対策として発注時期を前倒しします。関係者への周知をお願いします。",
    "新規取引の可否について意見交換を行いました。与信の確認後に再度協議します。結論は保留とします。",
    "予算の見直し案を共有しました。各部門は修正案を来週までに提出してください。質問は窓口まで。",
    "安全点検の結果、手すりの緩みが1か所見つかりました。修繕を依頼済みです。完了後に再確認します。",
]


def _body(plan_row, filler):
    """計画行の全スロットを使う最小のテンプレート本文（行の配列）。"""
    slot_line = "、".join("{%s}" % s["slot"] for s in plan_row["slots"]) + "が出席しました。本日の議題を共有します。"
    return ["お世話になっております。", slot_line, "", filler]


def test_run_filter_dedups_and_writes_outputs(tmp_path):
    rows = sp.build_plan(6, seed=1)
    for r in rows:                      # 長さの検査は本文の短さに合わせる（short 相当）
        r["length_min"], r["length_max"] = 60, 900
    raw = {}
    for i, r in enumerate(rows):
        body = _body(r, _FILLERS[0 if i == 1 else i])   # 1 番目は 0 番目と本文が同じ → 近重複で落ちる
        raw[r["request_id"]] = {"request_id": r["request_id"], "prompt_sha256": r["prompt_sha256"], "parsed": {"lines": body}}
    res = st.run_template_filter(raw, rows, quota_by_genre=False)
    ids = {a["template_id"] for a in res["accepted"]}
    assert "syn_v11_00000" in ids and "syn_v11_00001" not in ids, res["counts"]
    assert res["counts"]["near_duplicate"] >= 1 and len(ids) >= 4
    assert res["accepted"][0]["slots"] and "usage" in res["accepted"][0] and "repairs" in res["accepted"][0]
    st.write_template_outputs(res, str(tmp_path), plan_sha256="p", raw_sha256={"x.jsonl": "r"})
    assert (tmp_path / "templates.jsonl").exists() and (tmp_path / "report.md").exists()
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["plan_sha256"] == "p" and m["rule_version"] == st.RULE_VERSION and m["counts"]["accepted"] == len(ids)
    with open(tmp_path / "rejected.csv", encoding="utf-8") as f:
        assert {"request_id", "reasons", "evidence", "head"} <= set(next(csv.DictReader(f)).keys())


def test_load_raw_templates_reads_dir_or_file_last_wins(tmp_path):
    d = tmp_path / "run1"; d.mkdir()
    (d / sp.RAW_FILE).write_text('{"request_id": "a", "parsed": {"lines": ["x"]}}\n{"request_id": "b", "parsed": {"lines": ["y"]}}\n', encoding="utf-8")
    f = tmp_path / "extra.jsonl"
    f.write_text('{"request_id": "b", "parsed": {"lines": ["z"]}}\n', encoding="utf-8")
    rows = st.load_raw_templates([str(d), str(f)])
    assert set(rows) == {"a", "b"} and rows["b"]["parsed"]["lines"] == ["z"]


def test_bare_slot_names_get_braces_and_too_many_repairs_rejects():
    fixed, n = st.fix_bare_slot_names("「ORG1」の PERSON1さん、{PERSON2}、PERSON2_SEI と ORGANIZATION や ID10。")
    assert fixed == "「{ORG1}」の {PERSON1}さん、{PERSON2}、{PERSON2_SEI} と ORGANIZATION や {ID10}。" and n == 4
    rows = sp.build_plan(1, seed=1)
    for r in rows:
        r["length_min"], r["length_max"] = 60, 900
    body = ["、".join("{%s}" % x["slot"] for x in rows[0]["slots"]) + "が出席。連絡先は 03-1234-5678 と 045-1111-2222、",
            "住所は東京都と神奈川県と大阪府、担当は総務部です。会議は予定どおり終了しました。次回も同じ場所で行います。"]
    raw = {rows[0]["request_id"]: {"request_id": rows[0]["request_id"], "prompt_sha256": rows[0]["prompt_sha256"], "parsed": {"lines": body}}}
    res = st.run_template_filter(raw, rows, quota_by_genre=False)
    assert res["accepted"] == [] and res["counts"]["too_many_repairs"] == 1


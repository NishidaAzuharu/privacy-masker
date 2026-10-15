"""分類ヘッド v2: Qwen ラベル付けの計画・監査計画・選別（tools/category_labels.py）。Qwen は使わない。"""
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import category_labels as cl  # noqa: E402
from privacy_masker.normalize import default_normalize  # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES, RuleCategorizer  # noqa: E402


def _doc(text, ents, source="out_ner"):
    return {"text": text, "entities": ents, "source": source}


def test_extract_requests_marks_spans_and_splits_groups_of_eight():
    text = "".join(f"青葉{i:02d}社と" for i in range(10)) + "の話。"      # 10 個の ORG、各 5 字（青葉00社）+「と」
    ents = [[i * 6, i * 6 + 5, "ORG"] for i in range(10)]
    reqs = cl.extract_requests([_doc(text, ents)], "train")
    assert [len(r["spans"]) for r in reqs] == [8, 2]
    assert reqs[0]["doc_id"] == "train:0" and reqs[1]["doc_id"] == "train:0"
    assert reqs[0]["spans"][0] == {"index": 1, "kind": "org", "start": 0, "end": 5}
    assert reqs[0]["marked_text"].startswith("『青葉00社』(#1)と『青葉01社』(#2)と")
    assert reqs[1]["marked_text"].count("』(#") == 2 and "(#1)" in reqs[1]["marked_text"]   # 2 件目の番号は 1 から


def test_extract_requests_window_and_truncation_flags():
    text = "あ" * 400 + "青葉電機" + "い" * 400
    reqs = cl.extract_requests([_doc(text, [[400, 404, "ORG"]])], "valid", context_chars=300)
    (r,) = reqs
    assert r["window"] == [100, 704] and r["truncated"] == [True, True]
    assert r["marked_text"] == "あ" * 300 + "『青葉電機』(#1)" + "い" * 300
    r2 = cl.extract_requests([_doc("青葉電機の件", [[0, 4, "ORG"]])], "valid")[0]
    assert r2["window"] == [0, 6] and r2["truncated"] == [False, False]


def test_extract_requests_skips_synthetic_sources_and_non_target_types():
    docs = [_doc("青葉電機", [[0, 4, "ORG"]], source="qwen38_synth_v11"),
            _doc("山田太郎", [[0, 4, "PERSON"]]),
            _doc("札幌市の汐見病院", [[0, 3, "LOCATION"], [4, 8, "FACILITY"]], source="gsd_train"),
            _doc("札幌市の汐見病院", [[0, 3, "LOCATION"], [4, 8, "FACILITY"]], source="concat")]
    reqs = cl.extract_requests(docs, "train")
    assert len(reqs) == 1 and reqs[0]["doc_id"] == "train:2"                 # concat（doc 3）も実文書ではないので skip
    assert [s["kind"] for s in reqs[0]["spans"]] == ["location", "facility"]


def test_label_schema_enumerates_every_category_of_the_three_kinds():
    enum = cl.LABEL_SCHEMA["properties"]["labels"]["items"]["properties"]["category"]["enum"]
    for kind in ("org", "facility", "location"):
        assert set(CATEGORIES[kind]) <= set(enum)
        assert set(cl.CATEGORY_DEFS[kind]) == set(CATEGORIES[kind])          # 定義は語彙と 1:1
    assert cl.LABEL_SCHEMA["properties"]["labels"]["maxItems"] == cl.MAX_SPANS
    assert cl.LABEL_SCHEMA["properties"]["labels"]["items"]["properties"]["reason"]["maxLength"] == 120


def test_build_label_plan_rows_are_indexed_deterministic_and_source_balanced():
    docs = {"train": [_doc(f"青葉{i}社の件", [[0, 4, "ORG"]], source=("out_ner" if i % 4 else "gsd_train")) for i in range(80)],
            "valid": [_doc("汐見病院へ", [[0, 4, "FACILITY"]], source="gsd_train")]}
    rows = cl.build_label_plan(docs, seed=42)
    rows2 = cl.build_label_plan(docs, seed=42)
    assert rows == rows2 and len(rows) == 81
    assert [r["index"] for r in rows] == list(range(81)) and len({r["request_id"] for r in rows}) == 81
    assert rows[0]["request_id"] == "lab_000000" and rows[0]["stage"] == "label"
    assert all(r["max_tokens"] == cl.max_tokens_for(len(r["spans"])) for r in rows) and cl.max_tokens_for(8) == 600
    assert all(r["prompt_sha256"] == cl.sha256_short(cl.LABEL_SYSTEM_PROMPT + "\n" + r["prompt"]) for r in rows)
    head = Counter(r["source"] for r in rows[:40])                              # 先頭を取っても比率が保たれる
    assert 26 <= head["out_ner"] <= 34 and 6 <= head["gsd_train"] <= 14
    p = rows[0]["prompt"]
    assert "## ORG（組織）のカテゴリ" in p and "- finance:" in p and "## 本文" in p and "『" in p
    assert "## FACILITY（施設）のカテゴリ" not in p and "## LOCATION（地名）のカテゴリ" not in p   # 定義は出現する kind だけ


def test_label_plan_manifest_records_prompt_schema_and_counts(tmp_path):
    docs = {"train": [_doc("青葉電機の件", [[0, 4, "ORG"]])], "valid": []}
    rows = cl.build_label_plan(docs)
    m = cl.label_plan_manifest(rows, 42, "abc", {"train": "t", "valid": "v"})
    assert m["system_prompt"] == cl.LABEL_SYSTEM_PROMPT and m["output_schema"] == cl.LABEL_SCHEMA
    assert m["plan_tag"] == "plan_v1" and m["n"] == 1 and m["spans_by_kind"] == {"org": 1}
    assert m["sampling_recommended"]["temperature"] == 0.2 and m["plan_sha256"] == "abc"


def test_cli_plan_writes_plan_and_manifest(tmp_path):
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps(_doc("青葉電機の件", [[0, 4, "ORG"]]), ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text(json.dumps(_doc("汐見病院へ", [[0, 4, "FACILITY"]], source="gsd_train"), ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "labels"
    assert cl.main(["plan", "--data", str(data), "--out-dir", str(out)]) == 0
    rows = [json.loads(l) for l in (out / cl.LABEL_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    m = json.loads((out / cl.LABEL_PLAN_MANIFEST).read_text(encoding="utf-8"))
    assert len(rows) == 2 and m["n"] == 2 and m["plan_sha256"] == cl.file_sha256(str(out / cl.LABEL_PLAN_FILE))
    assert set(m["inputs"]) == {"train", "valid"}


# ---------------------------------------------------------------------------
# audit-plan / filter（Task 3）
# ---------------------------------------------------------------------------

def _raw(req, items, key="labels", error=""):
    return {"request_id": req["request_id"], "prompt_sha256": req["prompt_sha256"], "attempt": 1, "structured_error": error,
            "parsed": None if error else {key: items}}


def _label(index, category, confidence="high"):
    return {"index": index, "category": category, "confidence": confidence, "reason": "r"}


def _audit(index, verdict="ok", suggested=None):
    return {"index": index, "verdict": verdict, "suggested_category": suggested, "reason": "r"}


def test_parsed_items_requires_exactly_the_expected_indices():
    raw = {"parsed": {"labels": [_label(1, "it"), _label(2, "finance")]}}
    assert set(cl.parsed_items(raw, "labels", [1, 2])) == {1, 2}
    assert cl.parsed_items(raw, "labels", [1, 2, 3]) is None                      # 欠落
    assert cl.parsed_items({"parsed": {"labels": [_label(1, "it"), _label(1, "it")]}}, "labels", [1]) is None   # 重複
    assert cl.parsed_items({"parsed": None}, "labels", [1]) is None and cl.parsed_items(None, "labels", [1]) is None


def test_build_audit_plan_joins_labels_and_skips_structured_failures():
    docs = {"train": [_doc("アオバとミナトヤの件", [[0, 3, "ORG"], [4, 8, "ORG"]]), _doc("汐見病院へ", [[0, 4, "FACILITY"]])], "valid": []}
    label_rows = cl.build_label_plan(docs)
    r0 = next(r for r in label_rows if r["doc_id"] == "train:0"); r1 = next(r for r in label_rows if r["doc_id"] == "train:1")
    raw = [_raw(r0, [_label(1, "it"), _label(2, "retail", "medium")]), _raw(r1, [], error="schema_error")]
    rows = cl.build_audit_plan(label_rows, raw)
    assert len(rows) == 1 and rows[0]["request_id"] == "aud_000000" and rows[0]["label_request_id"] == r0["request_id"]
    assert rows[0]["stage"] == "audit" and rows[0]["labels"] == {"1": {"category": "it", "confidence": "high"}, "2": {"category": "retail", "confidence": "medium"}}
    p = rows[0]["prompt"]
    assert "- #1: ORG（組織） → it" in p and "- #2: ORG（組織） → retail" in p and "『アオバ』(#1)" in p
    assert rows[0]["prompt_sha256"] == cl.sha256_short(cl.AUDIT_SYSTEM_PROMPT + "\n" + p)
    m = cl.audit_plan_manifest(rows, "sha", "labelsha")
    assert m["stage"] == "audit" and m["output_schema"] == cl.AUDIT_SCHEMA and m["sampling_recommended"]["temperature"] == 0.5
    assert m["prompt_version"] == cl.AUDIT_PROMPT_VERSION and m["n"] == 1 and m["label_raw_sha256"] == "labelsha"


def _one_doc_env():
    # 規則で決まる 1 件（青葉電機 → manufacturing）と、規則で決まらない 5 件（カタカナ）を 1 文書に置く
    text = "青葉電機とアオバとミナトヤとヒカリノとサクラノとコハクの件"
    ents = [[0, 4, "ORG"], [5, 8, "ORG"], [9, 13, "ORG"], [14, 18, "ORG"], [19, 23, "ORG"], [24, 27, "ORG"]]
    docs = {"train": [_doc(text, ents)], "valid": []}
    label_rows = cl.build_label_plan(docs)
    (req,) = label_rows
    return docs, label_rows, req


def test_collect_spans_assigns_each_drop_reason_and_keeps_rule_rows():
    docs, label_rows, req = _one_doc_env()
    labels = [_label(1, "it"),                       # 規則(manufacturing)と不一致だが規則ラベルで残す
              _label(2, "it"),                       # 残る
              _label(3, "retail", "low"),            # low_confidence
              _label(4, "medical"),                  # wrong_kind_category（facility のカテゴリ）
              _label(5, "finance"),                  # audit_not_ok
              _label(6, "it")]                       # forbidden_surface（テスト用の禁止集合）
    audit_rows = cl.build_audit_plan(label_rows, [_raw(req, labels)])
    (areq,) = audit_rows
    audits = [_audit(1), _audit(2), _audit(3), _audit(4), _audit(5, "wrong", "retail"), _audit(6)]
    spans = cl.collect_spans(label_rows, [_raw(req, labels)], audit_rows, [_raw(areq, audits, key="audits")], docs, forbidden={"コハク"})
    by_start = {s["start"]: s for s in spans}
    assert by_start[0]["rule"] == "manufacturing" and by_start[0]["qwen"] == "it" and by_start[0]["drop"] is None
    assert by_start[5]["drop"] is None and by_start[5]["rule"] is None and by_start[5]["qwen"] == "it"
    assert by_start[9]["drop"] == "low_confidence" and by_start[14]["drop"] == "wrong_kind_category"
    assert by_start[19]["drop"] == "audit_not_ok" and by_start[19]["suggested"] == "retail"
    assert by_start[24]["drop"] == "forbidden_surface"
    assert all("surface" not in s for s in spans) and by_start[5]["surface_sha"] == cl.sha256_short("アオバ")


def test_collect_spans_marks_structured_failures_and_missing_audit():
    docs, label_rows, req = _one_doc_env()
    spans = cl.collect_spans(label_rows, [_raw(req, [], error="json_decode_error")], [], [], docs, forbidden=set())
    assert {s["drop"] for s in spans if s["rule"] is None} == {"label_structured"}
    assert next(s for s in spans if s["rule"]) ["drop"] is None                  # 規則ラベル付きは Qwen が失敗しても残す
    labels = [_label(i, "it") for i in range(1, 7)]
    spans = cl.collect_spans(label_rows, [_raw(req, labels)], [], [], docs, forbidden=set())
    assert {s["drop"] for s in spans if s["rule"] is None} == {"audit_missing"}


def test_collect_spans_treats_prompt_sha_mismatch_as_prompt_mismatch():
    docs, label_rows, req = _one_doc_env()
    labels = [_label(i, "it") for i in range(1, 7)]
    bad_raw = dict(_raw(req, labels)); bad_raw["prompt_sha256"] = "wrong-hash"
    spans = cl.collect_spans(label_rows, [bad_raw], [], [], docs, forbidden=set())
    assert {s["drop"] for s in spans if s["rule"] is None} == {"prompt_mismatch"}
    assert next(s for s in spans if s["rule"])["drop"] is None                   # 規則ラベル付きは影響を受けない


def test_build_audit_plan_skips_label_raw_with_mismatched_prompt_sha():
    docs = {"train": [_doc("アオバとミナトヤの件", [[0, 3, "ORG"], [4, 8, "ORG"]])], "valid": []}
    label_rows = cl.build_label_plan(docs)
    (r0,) = label_rows
    raw = _raw(r0, [_label(1, "it"), _label(2, "retail")])
    raw["prompt_sha256"] = "wrong-hash"
    assert cl.build_audit_plan(label_rows, [raw]) == []


def test_collect_spans_marks_audit_structured_failure_distinct_from_missing():
    docs, label_rows, req = _one_doc_env()
    labels = [_label(i, "it") for i in range(1, 7)]
    audit_rows = cl.build_audit_plan(label_rows, [_raw(req, labels)])
    (areq,) = audit_rows
    # 監査の依頼自体はあるが raw が構造化に失敗している → audit_structured（audit_missing とは別の理由）
    spans = cl.collect_spans(label_rows, [_raw(req, labels)], audit_rows, [_raw(areq, [], error="json_decode_error")], docs, forbidden=set())
    assert {s["drop"] for s in spans if s["rule"] is None} == {"audit_structured"}


def test_collect_spans_marks_audit_stale_when_label_category_changed_since_audit_plan():
    docs, label_rows, req = _one_doc_env()
    original_labels = [_label(i, "it") for i in range(1, 7)]
    audit_rows = cl.build_audit_plan(label_rows, [_raw(req, original_labels)])
    (areq,) = audit_rows
    audits = [_audit(i) for i in range(1, 7)]                        # 元のラベルに対する監査（全部 ok）
    changed_labels = [_label(1, "it"), _label(2, "finance")] + [_label(i, "it") for i in range(3, 7)]   # index2 だけ変更
    spans = cl.collect_spans(label_rows, [_raw(req, changed_labels)], audit_rows, [_raw(areq, audits, key="audits")], docs, forbidden=set())
    by_start = {s["start"]: s for s in spans}
    assert by_start[5]["drop"] == "audit_stale"                      # index2 = アオバ（start=5）
    assert by_start[0]["drop"] is None                                # index1 は規則ラベル付きなので影響なし


def test_forbidden_surfaces_is_the_eval_surfaces_file():
    from build_data_v7 import EVAL_SURFACES
    if not EVAL_SURFACES:
        pytest.skip("盲検表層ファイル（データフォルダの blind_eval_v1/eval_surfaces.json）が無い環境")
    assert cl._forbidden_surfaces() == set(EVAL_SURFACES)


def test_apply_consistency_drops_minority_and_drops_split_surfaces():
    def row(norm, qwen):
        return {"kind": "org", "norm": norm, "rule": None, "qwen": qwen, "drop": None}
    spans = [row("a", "it"), row("a", "it"), row("a", "finance"),            # 2/3 → 少数派（finance の 1 件）だけ落とす
             row("b", "it"), row("b", "finance"),                            # 1/2 < 2/3 → 全部落とす
             row("c", "retail"), {"kind": "org", "norm": "a", "rule": "manufacturing", "qwen": "it", "drop": None}]
    n = cl.apply_consistency(spans)
    assert n == 3                                                            # 少数派 1 件（a）+ 2 件（b）
    assert spans[0]["drop"] is None and spans[1]["drop"] is None             # 多数派はカテゴリを変えずそのまま残る
    assert spans[2]["drop"] == "inconsistent_surface" and spans[2]["qwen"] == "finance"   # 書き換えない
    assert spans[3]["drop"] == spans[4]["drop"] == "inconsistent_surface" and spans[5]["drop"] is None
    assert spans[6]["qwen"] == "it" and spans[6]["drop"] is None               # 規則ラベル付きは対象外


def test_agreement_rates_per_kind_and_excluding_low():
    def row(kind, rule, qwen, conf, verdict=None, suggested=None):
        return {"kind": kind, "rule": rule, "qwen": qwen, "confidence": conf, "verdict": verdict, "suggested": suggested, "drop": None}
    spans = [row("org", "it", "it", "high"), row("org", "it", "finance", "low"), row("org", "finance", "finance", "medium"),
             row("org", "retail", "it", "high", "wrong", "retail"), row("facility", "medical", "medical", "high"), row("org", None, "it", "high")]
    a = cl.agreement(spans)
    assert a["org"]["n_rule_total"] == 4 and a["org"]["n"] == 4 and abs(a["org"]["rate"] - 0.5) < 1e-9
    assert a["org"]["n_excl_low"] == 3 and abs(a["org"]["rate_excl_low"] - 2 / 3) < 1e-9
    assert a["org"]["audit_suggested_n"] == 1 and a["org"]["audit_suggested_rate"] == 1.0
    assert a["facility"]["n"] == 1 and a["facility"]["rate"] == 1.0 and "location" not in a


def test_agreement_reports_rule_total_even_when_no_qwen_label_is_usable():
    spans = [{"kind": "org", "rule": "it", "qwen": None, "confidence": None, "verdict": None, "suggested": None, "drop": None}
             for _ in range(5)]
    a = cl.agreement(spans)
    assert a["org"]["n_rule_total"] == 5 and a["org"]["n"] == 0
    assert a["org"]["rate"] is None and a["org"]["rate_excl_low"] is None
    assert a["org"]["error_rate"] is None and a["org"]["error_rate_upper"] is None


def test_agreement_computes_adjudicated_error_rate_and_breakdown():
    # ユーザー決定 2026-09-29 案 1: 門は「監査で裁定した Qwen の誤り率」。
    # 不一致 4 件（監査 ok 1・unsure 1・wrong→規則 1・wrong→他 1）+ 未裁定（監査なし）1 件 + 一致 5 件 → n 10。
    def row(qwen, rule="manufacturing", verdict=None, suggested=None):
        return {"kind": "org", "rule": rule, "qwen": qwen, "confidence": "high", "verdict": verdict, "suggested": suggested, "drop": None}
    spans = ([row("manufacturing") for _ in range(5)]                              # 一致 5 件
             + [row("it", verdict="ok")]                                          # 不一致・監査 ok（規則の癖）
             + [row("it", verdict="unsure")]                                      # 不一致・監査 unsure
             + [row("it", verdict="wrong", suggested="manufacturing")]            # 不一致・監査 wrong→規則（Qwen の誤り）
             + [row("it", verdict="wrong", suggested="retail")]                   # 不一致・監査 wrong→他
             + [row("it")])                                                       # 不一致・未裁定（監査なし）
    a = cl.agreement(spans)["org"]
    assert a["n_rule_total"] == 10 and a["n"] == 10
    assert a["disagree"] == 5
    assert a["adjudicated_errors"] == 1 and a["unadjudicated"] == 1
    assert a["disagree_audit_ok"] == 1 and a["disagree_unsure"] == 1 and a["disagree_wrong_other"] == 1
    assert abs(a["error_rate"] - 0.1) < 1e-9
    assert abs(a["error_rate_upper"] - 0.2) < 1e-9


def test_stats_gate_is_false_not_none_when_rule_rows_exist_but_no_usable_qwen_label():
    # 「Qwen の出力が 0 件でも判定が通ってしまう」バグの再現・修正確認。規則ラベル付きスパンは十分あるが
    # Qwen ラベルが 1 件も無いとき、判定不能（None）ではなく不合格（False）にする。
    spans = [{"kind": "org", "rule": "it", "qwen": None, "confidence": None, "verdict": None, "suggested": None,
              "drop": None, "norm": "x", "doc_id": "train:0"} for _ in range(5)]
    agree = cl.agreement(spans)
    st = cl._stats(spans, [], agree, max_error=0.10, min_n=3)
    assert st["agreement_gate"]["org"] is False and st["agreement_ok"] is False


def test_stats_min_agreement_n_zero_does_not_crash_comparing_none():
    spans = [{"kind": "org", "rule": "it", "qwen": None, "confidence": None, "verdict": None, "suggested": None,
              "drop": None, "norm": "x", "doc_id": "train:0"}]
    agree = cl.agreement(spans)
    st = cl._stats(spans, [], agree, max_error=0.10, min_n=0)          # min_n=0 でも n=0（error_rate_upper=None）で例外にならない
    assert st["agreement_gate"]["org"] is False


def test_stats_gate_thresholds_on_error_rate_upper_against_max_error():
    # 門は一致率ではなく「監査で裁定した Qwen の誤り率（上限）」。agreement() を経由せず、
    # _stats に渡す agreement 辞書を直接組み立てて 4 通りを確認する。
    spans = [{"kind": "org", "rule": "it", "qwen": None, "confidence": None, "verdict": None, "suggested": None,
              "drop": None, "norm": "x", "doc_id": "train:0"} for _ in range(10)]

    def st_for(n_rule_total, n, error_rate_upper):
        agree = {"org": {"n_rule_total": n_rule_total, "n": n, "error_rate_upper": error_rate_upper}}
        return cl._stats(spans, [], agree, max_error=0.10, min_n=5)

    assert st_for(10, 10, 0.05)["agreement_gate"]["org"] is True
    assert st_for(10, 10, 0.15)["agreement_gate"]["org"] is False
    assert st_for(3, 3, 0.05)["agreement_gate"]["org"] is None            # n_rule_total < min_n → 判定不能
    assert st_for(10, 0, None)["agreement_gate"]["org"] is False          # n == 0（Qwen ラベルが 1 件も無い）→ 不合格


def test_assign_splits_is_deterministic_stratified_and_judge_is_valid():
    docs = {"train": [_doc("x", [], source=("out_ner" if i % 5 else "gsd_train")) for i in range(100)],
            "valid": [_doc("y", [], source="gsd_train") for _ in range(3)] + [_doc("y", [], source="concat")]}
    s1 = cl.assign_splits(docs, seed=42, dev_fraction=0.10)
    s2 = cl.assign_splits(docs, seed=42, dev_fraction=0.10)
    assert s1 == s2 and all(s1[f"valid:{i}"] == "head_judge" for i in range(3))
    assert "valid:3" not in s1                                              # concat は実文書ではないので skip
    dev = [d for d, v in s1.items() if v == "head_dev"]
    assert len(dev) == 10 and sum(1 for d in dev if docs["train"][int(d.split(":")[1])]["source"] == "gsd_train") == 2   # 20 件の 10%
    assert cl.assign_splits(docs, seed=7, dev_fraction=0.10) != s1


def test_cli_filter_end_to_end_writes_outputs_and_gates_on_agreement(tmp_path, capsys):
    data = tmp_path / "data_v11"; data.mkdir()
    train = [_doc("青葉電機とアオバの件", [[0, 4, "ORG"], [5, 8, "ORG"]]), _doc("汐見病院とヒカリノへ", [[0, 4, "FACILITY"], [5, 9, "FACILITY"]], source="gsd_train")]
    valid = [_doc("札幌市の会議", [[0, 3, "LOCATION"]], source="gsd_train")]
    for name, rows in (("train_v11.jsonl", train), ("valid_v11.jsonl", valid)):
        (data / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    plan_dir = tmp_path / "labels"
    assert cl.main(["plan", "--data", str(data), "--out-dir", str(plan_dir)]) == 0
    label_rows = [json.loads(l) for l in (plan_dir / cl.LABEL_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    good = {"train:0": [_label(1, "manufacturing"), _label(2, "it")], "train:1": [_label(1, "medical"), _label(2, "medical", "medium")], "valid:0": [_label(1, "city")]}
    raw_dir = tmp_path / "qwen_label_plan_v1_full_seed42"; raw_dir.mkdir()
    (raw_dir / cl.LABEL_RAW_FILE).write_text("".join(json.dumps(_raw(r, good[r["doc_id"]]), ensure_ascii=False) + "\n" for r in label_rows), encoding="utf-8")
    assert cl.main(["audit-plan", "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir)]) == 0
    audit_rows = [json.loads(l) for l in (plan_dir / cl.AUDIT_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    araw_dir = tmp_path / "qwen_audit_plan_v1_full_seed42"; araw_dir.mkdir()
    (araw_dir / cl.AUDIT_RAW_FILE).write_text("".join(json.dumps(_raw(a, [_audit(s["index"]) for s in a["spans"]], key="audits"), ensure_ascii=False) + "\n" for a in audit_rows), encoding="utf-8")
    # --min-agreement-n 1 にして、規則ラベル付きスパンが各 kind 1 件しか無くても実際にゲートを評価させる
    rc = cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir), "--audit-raw", str(araw_dir),
                 "--max-error", "0.10", "--min-agreement-n", "1"])
    assert rc == 0
    out = [json.loads(l) for l in (plan_dir / cl.LABELS_FILE).read_text(encoding="utf-8").splitlines()]
    assert len(out) == 5 and {r["label_source"] for r in out} == {"rule", "qwen"}
    judge = [r for r in out if r["split"] == "head_judge"]
    assert len(judge) == 1 and judge[0]["category"] == "city" and judge[0]["label_source"] == "rule" and judge[0]["confidence"] is None
    qwen_rows = [r for r in out if r["label_source"] == "qwen"]
    assert {r["category"] for r in qwen_rows} == {"it", "medical"} and all(r["confidence"] in ("high", "medium") for r in qwen_rows)
    assert all("norm" not in r and "surface" not in r and "text" not in r for r in out)
    report = (plan_dir / cl.REPORT_FILE).read_text(encoding="utf-8")
    assert "青葉" not in report and "アオバ" not in report and "一致率" in report and "head_train" in report
    m = json.loads((plan_dir / cl.LABELS_MANIFEST).read_text(encoding="utf-8"))
    assert m["agreement_ok"] is True and m["counts"]["kept"] == 5 and m["files"]["labels"]["sha256"] == cl.file_sha256(str(plan_dir / cl.LABELS_FILE))
    assert m["agreement_gate"] == {"org": True, "facility": True, "location": True}     # ゲートが実際に評価されたことの確認
    assert m["max_error"] == 0.10 and "min_agreement" not in m
    # 規則と違う Qwen ラベルを付け、監査が wrong かつ規則を提案した（= 裁定された Qwen の誤り）ことにして exit code 2 にする
    bad = dict(good, **{"train:0": [_label(1, "it"), _label(2, "it")], "train:1": [_label(1, "education"), _label(2, "medical")]})
    (raw_dir / cl.LABEL_RAW_FILE).write_text("".join(json.dumps(_raw(r, bad[r["doc_id"]]), ensure_ascii=False) + "\n" for r in label_rows), encoding="utf-8")
    assert cl.main(["audit-plan", "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir)]) == 0     # bad の内容で監査計画を作り直す
    audit_rows2 = [json.loads(l) for l in (plan_dir / cl.AUDIT_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    wrong_map = {"train:0": {1: "manufacturing"}, "train:1": {1: "medical"}}      # 規則ラベル付きの index だけ wrong→規則
    audits_by_req = {a["request_id"]: [(_audit(s["index"], "wrong", wrong_map[a["doc_id"]][s["index"]]) if s["index"] in wrong_map.get(a["doc_id"], {})
                                        else _audit(s["index"])) for s in a["spans"]] for a in audit_rows2}
    (araw_dir / cl.AUDIT_RAW_FILE).write_text("".join(json.dumps(_raw(a, audits_by_req[a["request_id"]], key="audits"), ensure_ascii=False) + "\n" for a in audit_rows2),
                                              encoding="utf-8")
    capsys.readouterr()
    bad_out = tmp_path / "bad_out"                                          # 正式名が以前の実行から残っていない出力先
    rc = cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir), "--audit-raw", str(araw_dir),
                  "--out-dir", str(bad_out), "--max-error", "0.10", "--min-agreement-n", "1"])
    err = capsys.readouterr().err
    assert rc == 2 and "0.10" in err and cl.LABELS_REJECTED_FILE in err
    # 不合格: 正式名では書かず REJECTED に書く（manifest と report は書き、manifest に実際の名前と SHA を入れる）
    assert not (bad_out / cl.LABELS_FILE).exists() and (bad_out / cl.LABELS_REJECTED_FILE).exists()
    assert (bad_out / cl.REPORT_FILE).exists()
    m2 = json.loads((bad_out / cl.LABELS_MANIFEST).read_text(encoding="utf-8"))
    assert m2["files"]["labels"]["name"] == cl.LABELS_REJECTED_FILE
    assert m2["files"]["labels"]["sha256"] == cl.file_sha256(str(bad_out / cl.LABELS_REJECTED_FILE))
    assert m2["agreement_ok"] is False and m2["agreement_gate"]["org"] is False and m2["agreement_gate"]["facility"] is False
    assert m2["agreement"]["org"]["adjudicated_errors"] == 1 and m2["agreement"]["facility"]["adjudicated_errors"] == 1
    # 規則ラベルの保持: 規則で決まるスパンは Qwen が規則と違うカテゴリを付けても規則のカテゴリで残る
    # （finalize が Qwen の値を優先する退行 `s["qwen"] or s["rule"]` をここで落とす）
    docs = {"train": train, "valid": valid}
    qwen_at = {(r["doc_id"], sp["start"]): bad[r["doc_id"]][sp["index"] - 1]["category"] for r in label_rows for sp in r["spans"]}
    out2 = [json.loads(l) for l in (bad_out / cl.LABELS_REJECTED_FILE).read_text(encoding="utf-8").splitlines()]
    rule_rows = [r for r in out2 if r["label_source"] == "rule"]
    differ = 0
    for r in rule_rows:
        split, di = r["doc_id"].split(":")
        rule = RuleCategorizer.categorize(r["kind"], default_normalize(docs[split][int(di)]["text"][r["start"]:r["end"]]))
        assert r["category"] == rule and r["confidence"] is None
        differ += qwen_at[(r["doc_id"], r["start"])] != rule
    assert len(rule_rows) == 3 and differ == 2                               # org・facility の規則行は Qwen の値と違う
    # 以前の合格した実行の正式名が同じ出力先に残っていても書き換えない（manifest が今回の不合格を記録し、下流は拒否する）
    good_sha = cl.file_sha256(str(plan_dir / cl.LABELS_FILE))
    rc = cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir), "--audit-raw", str(araw_dir),
                  "--max-error", "0.10", "--min-agreement-n", "1"])
    err = capsys.readouterr().err
    assert rc == 2 and cl.file_sha256(str(plan_dir / cl.LABELS_FILE)) == good_sha and "agreement_ok=false" in err
    m3 = json.loads((plan_dir / cl.LABELS_MANIFEST).read_text(encoding="utf-8"))
    assert m3["agreement_ok"] is False and m3["files"]["labels"]["name"] == cl.LABELS_REJECTED_FILE


def test_cli_filter_without_audit_raw_all_disagreements_are_unadjudicated_and_gate_fails(tmp_path):
    # 監査 raw を渡さない実行では、規則と違う Qwen ラベルの不一致が全部 unadjudicated（fail-closed で誤り扱い）になり、門が落ちる。
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps(_doc("青葉電機とアオバの件", [[0, 4, "ORG"], [5, 8, "ORG"]]), ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text("", encoding="utf-8")
    plan_dir = tmp_path / "labels"
    assert cl.main(["plan", "--data", str(data), "--out-dir", str(plan_dir)]) == 0
    label_rows = [json.loads(l) for l in (plan_dir / cl.LABEL_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    bad = {"train:0": [_label(1, "it"), _label(2, "it")]}                    # idx1 = 青葉電機（規則は manufacturing）と不一致
    (raw_dir / cl.LABEL_RAW_FILE).write_text("".join(json.dumps(_raw(r, bad[r["doc_id"]]), ensure_ascii=False) + "\n" for r in label_rows), encoding="utf-8")
    # audit-plan を作らず --audit-raw も渡さない
    rc = cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir), "--min-agreement-n", "1"])
    assert rc == 2
    assert not (plan_dir / cl.LABELS_FILE).exists() and (plan_dir / cl.LABELS_REJECTED_FILE).exists()   # 不合格は正式名で書かない
    m = json.loads((plan_dir / cl.LABELS_MANIFEST).read_text(encoding="utf-8"))
    assert m["agreement"]["org"]["disagree"] == 1 and m["agreement"]["org"]["unadjudicated"] == 1 and m["agreement"]["org"]["adjudicated_errors"] == 0
    assert abs(m["agreement"]["org"]["error_rate_upper"] - 1.0) < 1e-9
    assert m["agreement_gate"]["org"] is False


def test_cmd_filter_exits_without_writing_when_no_label_raw_rows(tmp_path):
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps(_doc("青葉電機の件", [[0, 4, "ORG"]]), ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text("", encoding="utf-8")
    plan_dir = tmp_path / "labels"
    assert cl.main(["plan", "--data", str(data), "--out-dir", str(plan_dir)]) == 0
    empty_raw_dir = tmp_path / "empty_raw"; empty_raw_dir.mkdir()          # *_raw.jsonl が 1 つも無いフォルダ
    with pytest.raises(SystemExit):
        cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(empty_raw_dir)])
    assert not (plan_dir / cl.LABELS_FILE).exists()                        # 出力を書かずに終了する


def test_cmd_filter_exits_when_audit_raw_given_without_audit_plan(tmp_path):
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps(_doc("青葉電機の件", [[0, 4, "ORG"]]), ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text("", encoding="utf-8")
    plan_dir = tmp_path / "labels"
    assert cl.main(["plan", "--data", str(data), "--out-dir", str(plan_dir)]) == 0
    label_rows = [json.loads(l) for l in (plan_dir / cl.LABEL_PLAN_FILE).read_text(encoding="utf-8").splitlines()]
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    (raw_dir / cl.LABEL_RAW_FILE).write_text("".join(json.dumps(_raw(r, [_label(1, "manufacturing")]), ensure_ascii=False) + "\n" for r in label_rows),
                                              encoding="utf-8")
    araw_dir = tmp_path / "araw"; araw_dir.mkdir()                         # plan_audit.jsonl は一度も作っていない
    with pytest.raises(SystemExit):
        cl.main(["filter", "--data", str(data), "--plan-dir", str(plan_dir), "--label-raw", str(raw_dir), "--audit-raw", str(araw_dir)])
    assert not (plan_dir / cl.LABELS_FILE).exists()

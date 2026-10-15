"""負例生成ノート（make_negatives_notebook）のセルを置換して、分類ヘッド v2 のラベル付け・監査ノート
`qwen3_8_27b_category_labels.ipynb` を作る。vLLM 初期化・構造化出力・チェックポイント再開は負例ノートと同じ。
違うのは計画の場所（段で切り替え）・出力名・サンプリング・簡易検査だけ。

usage:
  PYTHONPATH=. .venv/bin/python tools/make_category_labels_notebook.py            # データフォルダ直下と notebooks_dir に書く
  PYTHONPATH=. .venv/bin/python tools/make_category_labels_notebook.py --dryrun   # RUN_LLM=False で全コードセルを実行（label と audit）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
import category_labels as cl  # noqa: E402
import make_negatives_notebook as nnb  # noqa: E402
from make_synth_notebook_v11 import _replace_all  # noqa: E402
from negatives_v8 import file_sha256, write_jsonl  # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES  # noqa: E402

NOTEBOOK_NAME = "qwen3_8_27b_category_labels.ipynb"

CELL_HEADER = """# Qwen3.8-27B-FP8 による実文書スパンのカテゴリ注釈（分類ヘッド v2）

> **これは分類ヘッド v2 の学習・判定データを作るノートです。** 実文書（data_v11 の out_ner / gsd_train）の
> ORG / FACILITY / LOCATION スパンに、`RUN_STAGE = "label"` でカテゴリを付け、`RUN_STAGE = "audit"` で同じ Qwen が
> 別の依頼文で監査します。依頼文・スキーマ・機械検査はローカル（`tools/category_labels.py`）にあり、ここでは計画を読んで生成するだけです。
>
> - 計画: `<データフォルダ>/category_labels_v1/plan_label.jsonl`（`tools/category_labels.py plan`）と `plan_audit.jsonl`（label の raw から `audit-plan`）
> - 出力: `<データフォルダ>/category_labels_v1/qwen_{label|audit}___PLAN_TAG___{pilot|full}_seed42/qwen_{labels|audits}_raw.jsonl`（バッチごとに保存、RESUME 対応）
> - 手順: label pilot → ローカルで `audit-plan` → audit pilot → ローカルで `filter`（監査で裁定した Qwen 誤り率（上限）が 0.10 を超える kind があれば依頼文を直して pilot から）→ label full → `audit-plan` → audit full → `filter`
> - 設計: 開発記録（非公開）
"""

CELL_STAGES = '''RUN_STAGE = "label"  # @param ["label", "audit"]
STAGES = {
    "label": {"plan_file": "plan_label.jsonl", "manifest_file": "plan_label.manifest.json", "temperature": 0.2,
              "raw_file": "qwen_labels_raw.jsonl", "manifest_out": "qwen_labels_manifest.json"},
    "audit": {"plan_file": "plan_audit.jsonl", "manifest_file": "plan_audit.manifest.json", "temperature": 0.5,
              "raw_file": "qwen_audits_raw.jsonl", "manifest_out": "qwen_audits_manifest.json"},
}
if RUN_STAGE not in STAGES:
    raise ValueError(f"未知のRUN_STAGE: {RUN_STAGE}")
RUN_PROFILES = {"pilot": {"max_docs": 200}, "full": {"max_docs": None}}'''

CELL_SUMMARY = r'''# 集計と manifest。簡易検査は明らかな崩れを数えるだけで、正式な選別はローカルの tools/category_labels.py filter。
KIND_CATEGORIES = __KIND_CATEGORIES__
ITEMS_KEY = "labels" if RUN_STAGE == "label" else "audits"
plan_by_id = {p["request_id"]: p for p in PLAN}
items = {r["request_id"]: (r["parsed"].get(ITEMS_KEY) or []) for r in ok_rows}
def _index_ok(rid, its):
    want = [s["index"] for s in plan_by_id[rid]["spans"]]
    return sorted(int(x.get("index", -1)) for x in its) == want
quick_counts = {
    "index_mismatch_docs": sum(1 for rid, its in items.items() if not _index_ok(rid, its)),
    "wrong_kind_category_docs": 0, "confidence_or_verdict": {},
}
if RUN_STAGE == "label":
    kind_by = {(rid, s["index"]): s["kind"] for rid in items for s in plan_by_id[rid]["spans"]}
    quick_counts["wrong_kind_category_docs"] = sum(
        1 for rid, its in items.items()
        if any(x.get("category") not in KIND_CATEGORIES.get(kind_by.get((rid, int(x.get("index", -1))), ""), ()) for x in its))
    quick_counts["confidence_or_verdict"] = dict(Counter(x.get("confidence") for its in items.values() for x in its))
else:
    quick_counts["confidence_or_verdict"] = dict(Counter(x.get("verdict") for its in items.values() for x in its))
errors = Counter((r.get("structured_error") or "")[:40] for r in ROWS if not isinstance(r.get("parsed"), dict))
total_seconds = sum(t["seconds"] for t in LLM_BATCH_TIMINGS)
total_requests = sum(t["requests"] for t in LLM_BATCH_TIMINGS)
total_tokens = sum(t.get("generated_tokens", 0) for t in LLM_BATCH_TIMINGS)
req_per_min = round(total_requests * 60 / total_seconds, 1) if total_seconds else None
projection_full_min = round(len(FULL_PLAN) / req_per_min) if req_per_min else None

MANIFEST = {
    "pipeline_version": CFG.pipeline_version, "stage": RUN_STAGE, "run_profile": CFG.run_profile, "run_signature": RUN_SIGNATURE,
    "status": "complete" if len(ROWS) == len(PLAN) else "partial",
    "model": {"id": CFG.model_id, "revision": CFG.model_revision, "vllm_expected": CFG.vllm_version,
              "vllm_runtime": VLLM_RUNTIME_VERSION, "torch_cuda_runtime": TORCH_CUDA_RUNTIME_VERSION,
              "gpu": GPU_ROWS, "init_seconds": INIT_SECONDS,
              "sampling": {"temperature": CFG.temperature, "top_p": CFG.top_p, "top_k": CFG.top_k, "min_p": CFG.min_p,
                           "presence_penalty": CFG.presence_penalty, "repetition_penalty": CFG.repetition_penalty},
              "max_model_len": CFG.max_model_len, "max_num_seqs": CFG.max_num_seqs},
    "plan": {"path": str(PLAN_PATH), "sha256": PLAN_SHA256, "plan_version": PLAN_VERSION, "prompt_version": PROMPT_VERSION,
             "full_rows": len(FULL_PLAN), "target_rows": len(PLAN)},
    "counts": {"planned": len(PLAN), "reused_from_checkpoint": reused, "generated_now": len(pending_plans),
               "rows": len(ROWS), "json_ok": len(ok_rows), "structured_failed": len(ROWS) - len(ok_rows),
               "spans_planned": sum(len(p["spans"]) for p in PLAN), "items_ok": sum(len(v) for v in items.values()),
               "by_kind_planned": dict(Counter(s["kind"] for p in PLAN for s in p["spans"]))},
    "structured_errors": dict(errors.most_common(10)),
    "quick_checks": quick_counts,
    "timing": {"generation_seconds": GENERATION_SECONDS, "llm_seconds": round(total_seconds, 1), "requests": total_requests,
               "generated_tokens": total_tokens, "req_per_min": req_per_min,
               "tok_per_sec": round(total_tokens / total_seconds, 1) if total_seconds else None,
               "projected_full_plan_minutes": projection_full_min},
    "llm_batch_timings": LLM_BATCH_TIMINGS,
    "files": {"raw": str(RAW_PATH), "raw_sha256": file_sha256(RAW_PATH) if RAW_PATH.exists() else None},
    "promote_final": False,
}
write_json(MANIFEST_PATH, MANIFEST)
print(json.dumps({k: MANIFEST[k] for k in ("stage", "status", "counts", "structured_errors", "quick_checks", "timing")},
                 ensure_ascii=False, indent=2))
print("manifest:", MANIFEST_PATH)
'''

CELL_AFTER = """## 実行後の手順（ローカル CPU）

1. pilot（200 リクエスト）の `quick_checks` を見る。`index_mismatch_docs` か `wrong_kind_category_docs` が 1 割を超えるなら依頼文を直して計画を作り直す。
2. `RUN_STAGE = "label"` の raw から監査の計画を作る: `PYTHONPATH=. .venv/bin/python tools/category_labels.py audit-plan --plan-dir <データフォルダ>/category_labels_v1 --label-raw <OUT_DIR>`
3. `RUN_STAGE = "audit"` で同じ profile を回す。
4. 選別: `PYTHONPATH=. .venv/bin/python tools/category_labels.py filter --data <データフォルダ>/data_v11 --plan-dir <データフォルダ>/category_labels_v1 --label-raw <label OUT_DIR> --audit-raw <audit OUT_DIR>`
   監査で裁定した Qwen 誤り率（上限、`filter_report.md` の kind 別）が 0.10 を超える kind があれば学習に進まない（exit code 2。選別結果は `category_labels_v1.REJECTED.jsonl` に書かれる）。依頼文（`tools/category_labels.py`）を直して pilot からやり直す。
5. `timing.projected_full_plan_minutes` を見てから `RUN_PROFILE = "full"`（label → audit-plan → audit → filter）。
"""


def cells() -> list[tuple[str, str]]:
    config = _replace_all(nnb.CELL_CONFIG, (
        ("# @title 負例生成の設定（Qwen3.8-27B-FP8 / Colab G4）", "# @title カテゴリ注釈の設定（Qwen3.8-27B-FP8 / Colab G4）"),
        ('RUN_PROFILES = {"pilot": {"max_docs": 200}, "full": {"max_docs": None}}', CELL_STAGES),
        ('PLAN_DIR = DRIVE_BASE / "negatives_v8"', 'PLAN_DIR = DRIVE_BASE / "category_labels_v1"'),
        ('PLAN_PATH = PLAN_DIR / "plan.jsonl"', 'PLAN_PATH = PLAN_DIR / STAGES[RUN_STAGE]["plan_file"]'),
        ('PLAN_MANIFEST_PATH = PLAN_DIR / "plan.manifest.json"', 'PLAN_MANIFEST_PATH = PLAN_DIR / STAGES[RUN_STAGE]["manifest_file"]'),
        ('pipeline_version: str = "qwen3_8_negatives_v1"', 'pipeline_version: str = "qwen3_8_category_labels_v1"'),
        ("temperature: float = 0.7", 'temperature: float = STAGES[RUN_STAGE]["temperature"]'),
        ("top_p: float = 0.8", "top_p: float = 0.9"),
        ("top_k: int = 20", "top_k: int = 40"),
        ("presence_penalty: float = 1.5", "presence_penalty: float = 0.0"),
        ("__MODEL_ID__", nnb.MODEL_ID), ("__MODEL_REVISION__", nnb.MODEL_REVISION),
    ))
    plan = _replace_all(nnb.CELL_PLAN, (
        ("tools/negatives_v8.py plan", "tools/category_labels.py plan（監査は audit-plan）"),
        ('("request_id", "index", "genre_key", "subgenre_key", "format_key", "length_key", "max_tokens", "prompt", "prompt_sha256")',
         '("request_id", "index", "doc_id", "stage", "spans", "max_tokens", "prompt", "prompt_sha256")'),
        ('OUT_DIR = DRIVE_BASE / f"qwen3_8_27b_negatives_v8_{PLAN_TAG}_{CFG.run_profile}_seed{CFG.seed}"',
         'OUT_DIR = PLAN_DIR / f"qwen_{RUN_STAGE}_{PLAN_TAG}_{CFG.run_profile}_seed{CFG.seed}"'),
        ('RAW_PATH = OUT_DIR / "qwen_negatives_raw.jsonl"', 'RAW_PATH = OUT_DIR / STAGES[RUN_STAGE]["raw_file"]'),
        ('MANIFEST_PATH = OUT_DIR / "qwen_negatives_manifest.json"', 'MANIFEST_PATH = OUT_DIR / STAGES[RUN_STAGE]["manifest_out"]'),
        ('DRIVE_BASE / f"qwen3_8_27b_negatives_v8_{PLAN_TAG}_pilot_seed{CFG.seed}" / "qwen_negatives_raw.jsonl"',
         'PLAN_DIR / f"qwen_{RUN_STAGE}_{PLAN_TAG}_pilot_seed{CFG.seed}" / STAGES[RUN_STAGE]["raw_file"]'),
        ('"prompt_version": PROMPT_VERSION, "pipeline_version": CFG.pipeline_version,',
         '"prompt_version": PROMPT_VERSION, "pipeline_version": CFG.pipeline_version, "stage": RUN_STAGE,'),
        ('print("ジャンル:", dict(Counter(r["genre_key"] for r in PLAN)))', 'print("kind:", dict(Counter(s["kind"] for r in PLAN for s in r["spans"])))'),
        ('print("形式:", dict(Counter(r["format_key"] for r in PLAN)))', 'print("source:", dict(Counter(r["source"] for r in PLAN)))'),
        ('print("長さ:", dict(Counter(r["length_key"] for r in PLAN)))', 'print("split:", dict(Counter(r["split"] for r in PLAN)))'),
    ))
    assert "negatives_v8" not in plan and "genre_key" not in plan
    generate = _replace_all(nnb.CELL_GENERATE, (
        ('stage="negatives"', "stage=RUN_STAGE"),
        ('"genre_key": plan["genre_key"], "subgenre_key": plan["subgenre_key"],\n                "format_key": plan["format_key"], "length_key": plan["length_key"],',
         '"doc_id": plan["doc_id"], "stage": plan["stage"], "spans": plan["spans"],'),
        ('"parsed": parsed, "chars": len(text) if isinstance(text, str) else None,',
         '"parsed": parsed, "n_items": (len(parsed.get("labels") or parsed.get("audits") or []) if isinstance(parsed, dict) else None),'),
    ))
    assert "genre_key" not in generate
    summary = CELL_SUMMARY.replace("__KIND_CATEGORIES__", json.dumps({k: list(CATEGORIES[k]) for k in ("org", "facility", "location")}, ensure_ascii=False))
    header = CELL_HEADER.replace("__PLAN_TAG__", cl.PLAN_TAG)
    return [
        ("markdown", header),
        ("code", nnb.CELL_DEPS),
        ("code", nnb.CELL_IMPORTS),
        ("code", nnb.CELL_DRIVE),
        ("markdown", "## 1. 設定（RUN_STAGE で label / audit を切り替える）"),
        ("code", config),
        ("markdown", "## 2. 計画の読み込み（依頼文・system prompt・出力 schema は計画側）"),
        ("code", plan),
        ("markdown", "## 3. vLLM 初期化（G4 対応・密結合 FP8）"),
        ("code", nnb.CELL_VLLM),
        ("code", nnb.CELL_BATCH),
        ("markdown", "## 4. 生成（canary → バッチ、バッチごとにデータフォルダへ保存）"),
        ("code", generate),
        ("markdown", "## 5. 集計と manifest"),
        ("code", summary),
        ("markdown", CELL_AFTER),
    ]


def build_notebook() -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells()):
        cell = {"cell_type": kind, "id": f"cat-lab-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {"cells": nb_cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "RTX_PRO_6000", "machine_shape": "hm"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def write_notebook(paths: list[str]) -> None:
    nb = build_notebook()
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", path)


def _fake_docs(n: int) -> dict[str, list[dict]]:
    """dry-run 用の架空文書（実文書の表層は使わない）。train/valid で重複させず、合計が n 件になるよう分ける
    （build_label_plan は train と valid の両方から計画行を作るため、重複させると planned が n を超える）。"""
    docs = [{"text": f"青葉{i:03d}商会は汐見{i:03d}病院の近くの若葉{i:03d}町にある。", "source": ("out_ner", "gsd_train")[i % 2],
             "entities": [[0, 7, "ORG"], [8, 15, "FACILITY"], [19, 25, "LOCATION"]]} for i in range(n)]
    return {"train": docs[2:], "valid": docs[:2]}


def dryrun(stage: str = "label", plan_rows: int = 24) -> None:
    """一時ディレクトリをデータフォルダに見立て、依存関係セル以外の全コードセルを RUN_LLM=False で実行する。
    audit 段は、label の raw を架空に作ってから `category_labels.build_audit_plan` で計画を作る。"""
    tmp = tempfile.mkdtemp(prefix="privacy-masker-category-labels-nb-dryrun-")
    plan_dir = os.path.join(tmp, cl.OUT_DIR_NAME)
    docs = _fake_docs(plan_rows)
    label_rows = cl.build_label_plan(docs, seed=42)
    label_path = os.path.join(plan_dir, cl.LABEL_PLAN_FILE)
    write_jsonl(label_path, label_rows)
    with open(os.path.join(plan_dir, cl.LABEL_PLAN_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(cl.label_plan_manifest(label_rows, 42, file_sha256(label_path), {}), f, ensure_ascii=False, indent=2)
    if stage == "audit":
        raw = [{"request_id": r["request_id"], "prompt_sha256": r["prompt_sha256"], "attempt": 1, "structured_error": "",
                "parsed": {"labels": [{"index": s["index"], "category": cl.OTHER[s["kind"]], "confidence": "medium", "reason": "x"}
                                      for s in r["spans"]]}} for r in label_rows]
        audit_rows = cl.build_audit_plan(label_rows, raw)
        audit_path = os.path.join(plan_dir, cl.AUDIT_PLAN_FILE)
        write_jsonl(audit_path, audit_rows)
        with open(os.path.join(plan_dir, cl.AUDIT_PLAN_MANIFEST), "w", encoding="utf-8") as f:
            json.dump(cl.audit_plan_manifest(audit_rows, file_sha256(audit_path), "labelraw"), f, ensure_ascii=False, indent=2)
    os.environ["PRIVACY_MASKER_DRIVE_BASE"] = tmp
    g = {"__name__": "__main__", "display": lambda x: print(str(x)[:300])}
    ok = True
    try:
        for i, (kind, src) in enumerate(cells()):
            if kind != "code" or "# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in src:
                continue
            src = src.replace("RUN_LLM = True", "RUN_LLM = False").replace('RUN_STAGE = "label"', f'RUN_STAGE = "{stage}"')
            exec(compile(src, f"<cell {i}>", "exec"), g)
        m = json.loads(g["MANIFEST_PATH"].read_text(encoding="utf-8"))
        assert m["stage"] == stage and m["counts"]["planned"] == plan_rows and m["counts"]["json_ok"] == 0, m["counts"]
        assert not g["RAW_PATH"].exists(), "RUN_LLM=False で raw が書かれた"
        assert f"qwen_{stage}_" in str(g["OUT_DIR"]) and cl.OUT_DIR_NAME in str(g["OUT_DIR"]), g["OUT_DIR"]
        print("\nDRYRUN OK:", g["OUT_DIR"])
    except Exception:
        ok = False
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=None)
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args(argv)
    if args.dryrun:
        dryrun("label"); dryrun("audit"); return
    write_notebook(args.out or [os.path.join(drive_dir(), NOTEBOOK_NAME), os.path.join(notebooks_dir(), NOTEBOOK_NAME)])


if __name__ == "__main__":
    main()

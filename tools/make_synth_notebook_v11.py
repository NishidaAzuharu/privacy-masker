"""負例生成ノート（make_negatives_notebook）のセルを置換して、v11 テンプレート生成ノート `qwen3_8_27b_synth_v11.ipynb` を作る。
vLLM 初期化・構造化出力・チェックポイント再開は負例ノートと同じ。違うのは計画の場所・出力名・サンプリング・簡易検査だけ。

usage:
  PYTHONPATH=. .venv/bin/python tools/make_synth_notebook_v11.py            # データフォルダ直下と notebooks_dir に書く
  PYTHONPATH=. .venv/bin/python tools/make_synth_notebook_v11.py --dryrun   # RUN_LLM=False で全コードセルを実行
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
import make_negatives_notebook as nnb  # noqa: E402
from negatives_v8 import file_sha256, write_jsonl  # noqa: E402

NOTEBOOK_NAME = "qwen3_8_27b_synth_v11.ipynb"

CELL_HEADER = """# Qwen3.8-27B-FP8 によるスロット付きテンプレートの生成（学習データ v11）

> **これは v11 の合成層の材料を作るノートです。** 値（人名・住所・組織）は書かせません。`{PERSON1}` などのスロットを含む
> 文書テンプレートを 1 リクエスト 1 件生成し、ローカルで受け入れ検査（`tools/synth_v11_templates.py filter`）と
> 値の差し込み（`tools/synth_v11_fill.py fill`）を行います。
>
> - 計画: `<データフォルダ>/synth_v11/plan.jsonl`（`tools/synth_v11_plan.py plan` で生成。ジャンル均衡順なので pilot は先頭 200 件。PLAN_TAG は計画の manifest から読む）
> - 出力: `<データフォルダ>/qwen3_8_27b_synth_v11___PLAN_TAG___{pilot|full}_seed42/qwen_templates_raw.jsonl`（バッチごとに保存、RESUME 対応）
> - 依存関係・vLLM 初期化・構造化出力は負例ノート（qwen3_8_27b_negatives_v1）と同じ。先頭セルの自動再起動後に全セルを再実行する
> - 設計: 開発記録（非公開）
"""

CELL_SUMMARY = r'''# 集計と manifest。簡易検査は明らかな崩れを数えるだけで、正式な受け入れ検査はローカルの tools/synth_v11_templates.py filter。
SLOT_RE = re.compile(r"\{(?!(?:ADDRESS|LOCATION|ORG|FACILITY|PHONE|EMAIL|URL|POSTAL|DATE|ID)\d+_)"
                     r"(PERSON|ADDRESS|LOCATION|ORG|FACILITY|PHONE|EMAIL|URL|POSTAL|DATE|ID)(\d+)(?:_(SEI|MEI|NICK))?\}")
ANY_BRACE_RE = re.compile(r"\{[^{}]*\}")
PREFECTURES = ("北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県",
               "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県",
               "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県",
               "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県", "熊本県", "大分県",
               "宮崎県", "鹿児島県", "沖縄県")
QUICK_CHECKS = {
    "prefecture_outside_slots": re.compile("|".join(PREFECTURES)),
    "phone_email_url_outside_slots": re.compile(r"0\d{1,4}-\d{1,4}-\d{3,4}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|https?://|www\."),
    "kanji_name_with_title_outside_slots": re.compile(r"[一-龥]{2,3}(?:さん|様|氏|部長|課長|先生)"),
    "foreign_script": re.compile(r"[Ѐ-ӿ가-힯؀-ۿ฀-๿这个们说时为对还现样关从发电让过种经]"),
}
texts = {r["request_id"]: parsed_text(r["parsed"]) or "" for r in ok_rows}
plan_by_id = {p["request_id"]: p for p in PLAN}
stripped = {rid: SLOT_RE.sub("", t) for rid, t in texts.items()}
quick_counts = {name: sum(1 for t in stripped.values() if rx.search(unicodedata.normalize("NFKC", t))) for name, rx in QUICK_CHECKS.items()}
quick_counts["slot_docs"] = sum(1 for t in texts.values() if SLOT_RE.search(t))
quick_counts["malformed_brace_docs"] = sum(1 for t in texts.values() if any(not SLOT_RE.fullmatch(m.group(0)) for m in ANY_BRACE_RE.finditer(t)))
quick_counts["missing_required_docs"] = sum(
    1 for rid, t in texts.items()
    if any(s["required"] and s["slot"] not in t for s in plan_by_id[rid]["slots"]))
quick_counts["newline_docs"] = sum(1 for t in texts.values() if "\n" in t)
lens = sorted(len(t) for t in stripped.values())
quantile = (lambda p: lens[int(p * (len(lens) - 1))]) if lens else (lambda p: 0)
length_ok = sum(1 for rid, t in stripped.items() if plan_by_id[rid]["length_min"] * 0.4 <= len(t) <= plan_by_id[rid]["length_max"] * 1.6)
by_genre_ok = Counter(plan_by_id[rid]["genre_key"] for rid in texts)
by_genre_all = Counter(p["genre_key"] for p in PLAN)
errors = Counter((r.get("structured_error") or "")[:40] for r in ROWS if not isinstance(r.get("parsed"), dict))
total_seconds = sum(t["seconds"] for t in LLM_BATCH_TIMINGS)
total_requests = sum(t["requests"] for t in LLM_BATCH_TIMINGS)
total_tokens = sum(t.get("generated_tokens", 0) for t in LLM_BATCH_TIMINGS)
req_per_min = round(total_requests * 60 / total_seconds, 1) if total_seconds else None
projection_full_min = round(len(FULL_PLAN) / req_per_min) if req_per_min else None

MANIFEST = {
    "pipeline_version": CFG.pipeline_version, "run_profile": CFG.run_profile, "run_signature": RUN_SIGNATURE,
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
               "length_within_plan": length_ok, "by_genre_planned": dict(by_genre_all), "by_genre_ok": dict(by_genre_ok)},
    "structured_errors": dict(errors.most_common(10)),
    "chars_without_slots": {"p10": quantile(.1), "p50": quantile(.5), "p90": quantile(.9), "mean": (sum(lens) // len(lens)) if lens else 0},
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
print(json.dumps({k: MANIFEST[k] for k in ("status", "counts", "structured_errors", "chars_without_slots", "quick_checks", "timing")},
                 ensure_ascii=False, indent=2))
print("manifest:", MANIFEST_PATH)
print("\n--- 先頭 5 件（先頭 160 字） ---")
for rid in list(texts)[:5]:
    p = plan_by_id[rid]
    print(f"[{p['genre_key']}/{p['subgenre_key']}/{p['format_key']}] {texts[rid][:160].replace(chr(10), '⏎')}")
'''

CELL_AFTER = """## 実行後の手順（ローカル CPU）

1. pilot（200 件）の `quick_checks` を見る。`malformed_brace_docs` と `missing_required_docs` が 1 割を超えるなら system prompt を直して計画を作り直す
   （`tools/synth_v11_plan.py plan` → 計画の SHA が変わるので出力フォルダ名も変わる）。
2. `timing.projected_full_plan_minutes` が Colab のセッション時間に収まるか確認してから `RUN_PROFILE = "full"`。
3. 受け入れ検査 → 差し込み → 構築はローカルで:

```bash
PYTHONPATH=. .venv/bin/python tools/synth_v11_templates.py filter --raw "<データフォルダ>/qwen3_8_27b_synth_v11___PLAN_TAG___full_seed42" --target 7000
PYTHONPATH=. .venv/bin/python tools/synth_v11_fill.py fill
PYTHONPATH=. .venv/bin/python tools/gsd_train_to_ner.py
PYTHONPATH=. .venv/bin/python tools/build_data_v11.py
```
"""


def _replace_all(src: str, pairs: tuple[tuple[str, str], ...]) -> str:
    for old, new in pairs:
        assert old in src, f"置換元が見つからない: {old!r}"
        src = src.replace(old, new)
    return src


def cells() -> list[tuple[str, str]]:
    config = _replace_all(nnb.CELL_CONFIG, (
        ("# @title 負例生成の設定（Qwen3.8-27B-FP8 / Colab G4）", "# @title テンプレート生成の設定（Qwen3.8-27B-FP8 / Colab G4）"),
        ('PLAN_DIR = DRIVE_BASE / "negatives_v8"', 'PLAN_DIR = DRIVE_BASE / "synth_v11"'),
        ('pipeline_version: str = "qwen3_8_negatives_v1"', 'pipeline_version: str = "qwen3_8_synth_v11"'),
        ("temperature: float = 0.7", "temperature: float = 0.85"),
        ("top_p: float = 0.8", "top_p: float = 0.9"),
        ("top_k: int = 20", "top_k: int = 40"),
        ("__MODEL_ID__", nnb.MODEL_ID), ("__MODEL_REVISION__", nnb.MODEL_REVISION),
    ))
    plan = _replace_all(nnb.CELL_PLAN, (("tools/negatives_v8.py plan", "tools/synth_v11_plan.py plan"),))
    plan = plan.replace("qwen3_8_27b_negatives_v8_", "qwen3_8_27b_synth_v11_").replace("qwen_negatives_", "qwen_templates_")
    assert "qwen3_8_27b_synth_v11_" in plan and "qwen_templates_raw.jsonl" in plan and "negatives_v8" not in plan
    generate = _replace_all(nnb.CELL_GENERATE, (('stage="negatives"', 'stage="templates"'),))
    from synth_v11_plan import PLAN_TAG
    header = CELL_HEADER.replace("__PLAN_TAG__", PLAN_TAG)
    after = CELL_AFTER.replace("__PLAN_TAG__", PLAN_TAG)
    return [
        ("markdown", header),
        ("code", nnb.CELL_DEPS),
        ("code", nnb.CELL_IMPORTS),
        ("code", nnb.CELL_DRIVE),
        ("markdown", "## 1. 設定"),
        ("code", config),
        ("markdown", "## 2. 計画の読み込み（依頼文・system prompt・出力 schema は計画側）"),
        ("code", plan),
        ("markdown", "## 3. vLLM 初期化（G4 対応・密結合 FP8）"),
        ("code", nnb.CELL_VLLM),
        ("code", nnb.CELL_BATCH),
        ("markdown", "## 4. 生成（canary → バッチ、バッチごとにデータフォルダへ保存）"),
        ("code", generate),
        ("markdown", "## 5. 集計と manifest"),
        ("code", CELL_SUMMARY),
        ("markdown", after),
    ]


def build_notebook() -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells()):
        cell = {"cell_type": kind, "id": f"syn-v11-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
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


def dryrun(plan_rows: int = 40) -> None:
    """一時ディレクトリをデータフォルダに見立て、依存関係セル以外の全コードセルを RUN_LLM=False で実行する。"""
    import synth_v11_plan as sp

    tmp = tempfile.mkdtemp(prefix="privacy-masker-synth-v11-nb-dryrun-")
    plan_dir = os.path.join(tmp, sp.PLAN_DIR_NAME)
    rows = sp.build_plan(plan_rows, seed=42, personas=None)
    plan_path = os.path.join(plan_dir, sp.PLAN_FILE)
    write_jsonl(plan_path, rows)
    manifest = sp.plan_manifest(rows, 42, file_sha256(plan_path))
    with open(os.path.join(plan_dir, sp.PLAN_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.environ["PRIVACY_MASKER_DRIVE_BASE"] = tmp
    g = {"__name__": "__main__", "display": lambda x: print(str(x)[:300])}
    ok = True
    try:
        for i, (kind, src) in enumerate(cells()):
            if kind != "code" or "# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in src:
                continue
            exec(compile(src.replace("RUN_LLM = True", "RUN_LLM = False"), f"<cell {i}>", "exec"), g)
        m = json.loads(g["MANIFEST_PATH"].read_text(encoding="utf-8"))
        assert m["counts"]["planned"] == plan_rows and m["counts"]["json_ok"] == 0, m["counts"]
        assert not g["RAW_PATH"].exists(), "RUN_LLM=False で raw が書かれた"
        assert "synth_v11" in str(g["OUT_DIR"]), g["OUT_DIR"]
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
        dryrun(); return
    write_notebook(args.out or [os.path.join(drive_dir(), NOTEBOOK_NAME), os.path.join(notebooks_dir(), NOTEBOOK_NAME)])


if __name__ == "__main__":
    main()

"""ラベル付けノートブックの組み立て（tools/make_category_labels_notebook.py）。vLLM は使わず RUN_LLM=False で全セルを通す。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import make_category_labels_notebook as mk  # noqa: E402


def test_cells_are_rewritten_for_category_labels():
    code = "\n".join(src for kind, src in mk.cells() if kind == "code")
    assert 'RUN_STAGE = "label"' in code and '"audit"' in code
    assert 'PLAN_DIR = DRIVE_BASE / "category_labels_v1"' in code
    assert 'STAGES[RUN_STAGE]["plan_file"]' in code and "qwen_labels_raw.jsonl" in code and "qwen_audits_raw.jsonl" in code
    assert 'stage=RUN_STAGE' in code and '"stage": RUN_STAGE' in code
    assert 'temperature: float = STAGES[RUN_STAGE]["temperature"]' in code and '"temperature": 0.2' in code and '"temperature": 0.5' in code
    assert "presence_penalty: float = 0.0" in code and "top_p: float = 0.9" in code and "top_k: int = 40" in code
    assert "negatives_v8" not in code and "qwen_negatives_" not in code and "genre_key" not in code
    assert "Qwen/Qwen3.8-27B-FP8" in code and "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a" in code
    assert "index_mismatch_docs" in code and "wrong_kind_category_docs" in code      # ラベル向けの簡易検査
    assert '"doc_id": plan["doc_id"]' in code
    # 門は「監査で裁定した Qwen 誤り率（上限）≤ 0.10」（2026-09-29 に一致率 0.90 の門から変更）。旧記述を残さない
    markdown = "\n".join(src for kind, src in mk.cells() if kind == "markdown")
    assert "0.90" not in markdown and "一致率" not in markdown
    assert "誤り率（上限）" in markdown and "0.10" in markdown


def test_notebook_json_shape():
    nb = mk.build_notebook()
    assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
    assert any("# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in "".join(c["source"]) for c in nb["cells"])


def test_dryrun_label_stage_runs_all_code_cells_without_llm():
    pytest.importorskip("pandas")     # ノートブック本体が pandas を使う（最小構成の CI では skip）
    mk.dryrun(stage="label", plan_rows=24)


def test_dryrun_audit_stage_runs_all_code_cells_without_llm():
    pytest.importorskip("pandas")
    mk.dryrun(stage="audit", plan_rows=24)

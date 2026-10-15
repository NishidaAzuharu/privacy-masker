"""生成ノートブックの組み立て（tools/make_synth_notebook_v11.py）。vLLM は使わず RUN_LLM=False で全セルを通す。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import make_synth_notebook_v11 as mk  # noqa: E402


def test_cells_are_rewritten_for_synth_v11():
    code = "\n".join(src for kind, src in mk.cells() if kind == "code")
    assert 'PLAN_DIR = DRIVE_BASE / "synth_v11"' in code
    assert "qwen3_8_27b_synth_v11_" in code and "qwen_templates_raw.jsonl" in code and "qwen_templates_manifest.json" in code
    assert "negatives_v8" not in code and "qwen_negatives_" not in code and "qwen3_8_27b_negatives_v8_" not in code
    assert 'stage="templates"' in code and "temperature: float = 0.85" in code
    assert "Qwen/Qwen3.8-27B-FP8" in code and "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a" in code
    assert "slot_docs" in code                      # テンプレート向けの簡易検査


def test_notebook_json_shape():
    nb = mk.build_notebook()
    assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
    assert any("# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in "".join(c["source"]) for c in nb["cells"])


def test_dryrun_runs_all_code_cells_without_llm():
    pytest.importorskip("pandas")     # ノートブック本体が pandas を使う（requirements-dev.txt。最小構成の CI では skip）
    mk.dryrun(plan_rows=24)      # 例外が出なければ OK（内部で manifest の counts を assert している）

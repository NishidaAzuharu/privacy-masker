"""分類ヘッド（計画 B）のモジュールが torch / safetensors / numpy 無しで import できること。

``tests/test_category_head.py`` は重みの書き出しに numpy / safetensors を使うため CI の minimal profile
（``.[dev]`` のみ）では skip される。ここは追加の依存を一切使わないので、minimal profile でも走る。
"""
import subprocess
import sys

import pytest

from privacy_masker.category_head import CATEGORY_HEAD_CONFIG, CATEGORY_HEAD_WEIGHTS, HeadCategorizer


def test_import_does_not_pull_in_torch_or_safetensors_or_numpy():
    # 基本パッケージが [transformers] extra 無しで import できることを保証する
    # （torch/numpy/safetensors をトップレベルで import しない）。同一プロセスだと他のテストが
    # 先に torch 等を import してしまい判定できないため、サブプロセスで確認する。
    code = (
        "import sys\n"
        "import privacy_masker.category_head\n"
        "assert 'torch' not in sys.modules, 'torch should not be imported at module level'\n"
        "assert 'safetensors' not in sys.modules, 'safetensors should not be imported at module level'\n"
        "assert 'numpy' not in sys.modules, 'numpy should not be imported at module level'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr


def test_load_without_safetensors_raises_import_error_not_configuration_error(tmp_path, monkeypatch):
    # safetensors が入っていない（[transformers] extra 無し）ことを「ヘッドのファイルが壊れている」
    # （ConfigurationError）に丸め込まない。minimal profile では本当に無く、full profile では sys.modules で塞ぐ
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text("{}", encoding="utf-8")
    (tmp_path / CATEGORY_HEAD_WEIGHTS).write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
    with pytest.raises(ImportError):
        HeadCategorizer.load(str(tmp_path))

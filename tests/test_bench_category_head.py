"""1,870 件の確認スクリプトは Drive と本番モデルが要るので、ここでは import と引数解析、集計関数だけを見る。"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("script", ["tools/bench_category_head.py", "tools/bench_category_head_realcheck.py"])
def test_help_runs_without_drive(script):
    env = dict(os.environ, PYTHONPATH=ROOT, PRIVACY_MASKER_DRIVE_DIR="/nonexistent")
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "--head-dir" in r.stdout, r.stderr


@pytest.mark.parametrize("script", ["tools/bench_category_head.py", "tools/bench_category_head_realcheck.py"])
def test_help_lists_model_option(script):
    env = dict(os.environ, PYTHONPATH=ROOT, PRIVACY_MASKER_DRIVE_DIR="/nonexistent")
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "--model" in r.stdout


def test_realcheck_precision_table_is_pure():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import bench_category_head_realcheck as rc
    items = [("org", "it", True), ("org", "finance", True), ("org", "it", True)]         # (kind, 規則ラベル, 種別語を削ったか)
    probs = [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]                                         # cats = [it, finance]
    table = rc.precision_table(items, probs, ["it", "finance"], margins=(0.15, 0.5))
    assert table[0.15] == {"n": 3, "correct": 1, "precision": 1 / 3} and table[0.5] == {"n": 2, "correct": 1, "precision": 0.5}

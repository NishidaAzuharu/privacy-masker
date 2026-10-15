"""学習ノート v11 の生成（tools/make_train_notebook_v11.py）。v10 ノートからの置換が全部当たり、neutral 対応が入ること。
元になる学習ノートブック（notebooks_dir。PRIVACY_MASKER_NOTEBOOKS）が無ければ skip。"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import make_train_notebook_v11 as mk  # noqa: E402
from _paths import notebooks_dir  # noqa: E402

_SRC = os.path.join(notebooks_dir(), "train_ja_mask_charhead_v7_scratch.ipynb")   # 生成の連鎖（v7 → v8 → … → v11）の起点
pytestmark = pytest.mark.skipif(not os.path.exists(_SRC), reason="元ノートブックが無い（別マシン）")


def test_generated_notebook_has_v11_data_neutral_and_revision(tmp_path):
    out = tmp_path / "v11.ipynb"
    mk.main(["--out", str(out)])
    nb = json.loads(out.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "data_v11/train_v11.jsonl" in code and "data_v11/train_v11_plus_old.jsonl" in code and "data_v11/valid_v11.jsonl" in code
    assert "data_v10" not in code and "v10-noneg" not in code
    # ローカルディレクトリから始めるときは revision を付けない（HF の id のときだけ commit 固定）
    assert f'else "{mk.ENCODER_REVISION}"' in code and "ENCODER_REVISION = None if os.path.isdir(ENCODER_INIT)" in code
    assert code.count("**_rev_kwargs()") >= 3            # tokenizer / encoder / 凍結確認の 3 か所
    assert '"neutral": neutral' in code                     # normalize_records が neutral を保持
    assert "neutral_docs" in code                          # mask_metrics が neutral を過剰マスクから除く
    assert code.count("ch.IGNORE_INDEX") >= 3               # 境界・neutral の両方で使う
    assert '"encoder_revision": ENCODER_REVISION' in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":                       # 構文が壊れていない（!pip / %magic 行は除く）
            src = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
            compile(src, "<cell>", "exec")


def test_70m_variant_pins_its_own_revision_and_defaults_to_plus_old(tmp_path):
    out = tmp_path / "v11_70m.ipynb"
    mk.main(["--out", str(out), "--encoder", "70m"])
    nb = json.loads(out.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert '"cl-nagoya/ruri-v3-70m")' in code and mk.ENCODERS["70m"]["revision"] in code
    assert 'DATA_VARIANT_DEFAULT = "plus_old"' in code
    assert "ruri-v3-30m\")" not in code            # encoder の既定に 30m が残っていない
    assert "70m" in "".join(nb["cells"][0]["source"])


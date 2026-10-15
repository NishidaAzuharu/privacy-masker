"""多タスク版ノートブックの生成（tools/make_multitask_notebook.py）。元の v11 ノートブックが notebooks_dir に無ければ skip。"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import make_multitask_notebook as mk  # noqa: E402
from _paths import notebooks_dir  # noqa: E402

SRC = os.path.join(notebooks_dir(), mk.SOURCE_NAME)
pytestmark = pytest.mark.skipif(not os.path.exists(SRC), reason="元ノートブックが無い（別マシン）")


def _code(nb):
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def test_transform_rewrites_the_expected_cells():
    nb = json.load(open(SRC, encoding="utf-8"))
    out = mk.transform(nb)
    assert len(out["cells"]) == len(nb["cells"]) + 2                      # markdown + code の結合セル
    code = "\n".join(_code(out))
    assert 'DATA_VARIANT_DEFAULT = "plus_old"' in code and 'RUN_TAG = RUN_TAG + "-multitask"' in code
    assert "CAT_LAMBDA = 0.3" in code and "CAT_LR = 1e-3" in code and "category_labels_v1/category_labels_v1.jsonl" in code
    assert "ch.category_spans_by_text(_cat_rows, _text_by_doc)" in code and "ch.sqrt_class_weights(" in code
    assert "category_spans=local_cats" in code and "n_cat_cut" in code
    assert "aux = ch.CategoryAuxHead(model.config.hidden_size).to(device)" in code
    assert '("aux",' in code and "ch.category_loss(aux, out[\"token_hidden\"], batch, class_weights=CAT_CLASS_WEIGHTS_DEV)" in code
    assert "loss = out[\"loss\"] + CAT_LAMBDA * cat_loss" in code
    assert code.count("list(model.parameters()) + list(aux.parameters())") == 2
    assert "category_aux.safetensors" in code and 'manifest["category_aux"]' in code
    assert "v12 多タスク版" in "".join(out["cells"][0]["source"])
    # 教師が空・本文の取り違え・SMOKE_LIMIT での切り詰めを黙って通さない（レビュー指摘）
    assert 'assert CAT_STATS["n_rows"] > 0 and CAT_STATS["n_used"] == CAT_STATS["n_rows"]' in code
    assert 'CAT_STATS["n_surface_sha_mismatch"] = _n_bad_sha' in code and "assert _n_bad_sha == 0" in code
    assert "assert not SMOKE_LIMIT" in code
    assert '"aux_weights_epoch": history[-1]["epoch"]' in code and '"train_v11_sha256": CAT_TRAIN_V11_SHA' in code
    header = "".join(out["cells"][0]["source"])
    assert "`plus_old`" in header and "outputs_ruri-v3-30m-charhead-v11-plus-old-frompre-multitask" in header
    # 元の NER レシピは変えない
    assert "EPOCHS        = 1 if SMOKE_LIMIT else 10" in code and "LR_ENCODER    = 2e-5 if FROM_FINETUNED else 3e-5" in code
    assert "BATCH_SIZE    = 32" in code and "LR_HEAD       = 1e-3" in code


def test_join_cell_follows_data_cell_and_drops_duplicate_spans():
    out = mk.transform(json.load(open(SRC, encoding="utf-8")))
    cells = out["cells"]
    i = next(k for k, c in enumerate(cells) if "all_train, all_valid, SPLIT_MANIFEST = load_split(" in "".join(c["source"]))
    assert cells[i + 1]["cell_type"] == "markdown" and "3b. カテゴリ教師の結合" in "".join(cells[i + 1]["source"])
    join = "".join(cells[i + 2]["source"])
    assert cells[i + 2]["cell_type"] == "code" and "ch.category_spans_by_text(" in join
    assert join.index("assert not SMOKE_LIMIT") < join.index("read_jsonl(")              # 切り詰めて読む前に止める
    assert join.index("_n_bad_sha = ") < join.index("ch.category_spans_by_text(")       # head_train/head_dev 全行を照合
    _call = join.index("CAT_BY_TEXT, CAT_STATS = ch.category_spans_by_text(")
    assert join.index('assert CAT_STATS["n_rows"] > 0') == join.index("\n", _call) + 1   # 結合の直後で空を止める
    # 同じ本文を持つ train:<行> が複数あると同じスパンが重複して入る → 本文ごとに sorted(set(...)) で落として数える
    assert "CAT_BY_TEXT = {t: sorted(set(v)) for t, v in CAT_BY_TEXT.items()}" in join
    assert 'CAT_STATS["n_duplicate_spans_dropped"]' in join
    assert join.index("sorted(set(v))") < join.index("_counts = {k: _Counter()")   # class weight は重複除去後に数える


def test_every_code_cell_compiles():
    out = mk.transform(json.load(open(SRC, encoding="utf-8")))
    for i, src in enumerate(_code(out)):
        # Colab のシェル行（!pip）と magic（%）は Python の構文ではないので除く
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        compile(src, f"<cell {i}>", "exec")


def test_transform_is_idempotent_guard():
    nb = json.load(open(SRC, encoding="utf-8"))
    once = mk.transform(nb)
    with pytest.raises(AssertionError):
        mk.transform(once)                                                 # 二重適用は置換元が無く止まる


def test_main_writes_every_out_path(tmp_path):
    a, b = tmp_path / "a" / mk.NOTEBOOK_NAME, tmp_path / "b" / mk.NOTEBOOK_NAME
    mk.main(["--out", str(a), "--out", str(b)])
    for p in (a, b):
        nb = json.loads(p.read_text(encoding="utf-8"))
        assert "v12 多タスク版" in "".join(nb["cells"][0]["source"])
    assert a.read_bytes() == b.read_bytes()

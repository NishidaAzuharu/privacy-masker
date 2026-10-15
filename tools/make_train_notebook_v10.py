"""v9 の学習ノートブック生成器を使って v10 版 `train_ja_mask_charhead_v10_scratch.ipynb` を作る（データ指定・タグ・ヘッダの置換だけ）。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_train_notebook_v10.py [--default-variant neg|noneg]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
import make_train_notebook_v9 as v9nb  # noqa: E402

DST_NAME = "train_ja_mask_charhead_v10_scratch.ipynb"

HEADER = """# 日本語PIIマスキング 文字単位ヘッド — データ v10 / 素の事前学習モデルから

> **この Notebook は 1 回の全セル実行で 1 モデルだけ学習します。**
>
> - `ENCODER_INIT = "cl-nagoya/ruri-v3-30m"`（素の事前学習モデル）
> - データ: 設定セル先頭の `DATA_VARIANT_DEFAULT`。`neg` = `data_v10/train_v10.jsonl`（LLM 負例 2,000 あり、**既定**）、
>   `noneg` = `data_v10/train_v10_noneg.jsonl`。valid は共通の `data_v10/valid_v10.jsonl`
> - `EPOCHS = 10` / `PATIENCE = 3` / エポックごとに `epochNN/` へ保存。**エポック選択は 4 評価セットで行い、
>   後半エポックの平均（`tools/average_checkpoints.py`、範囲は「best 以降全部」と「後半 3 エポック」に固定）も候補に入れる**
> - 保存先: `outputs_ruri-v3-30m-charhead-v10-frompre/`

## v9 からの変更（設計は開発記録、非公開）

v9 は英数字の誤検出とカタカナ著名人を直したが、帳票の珍しい姓（`宅地建物取引士: 雪見沢` `記録者：夜勤 朱鷺野`）を落とした。
v10 は帳票の比率調整だけ: フレーム行で名が入る割合 70 → 85%、帳票文書 1,100 → 1,600、頻度の低い姓の重み 3、字間空白の姓 5%。
Qwen によるテンプレート生成は使わない（2026-09-17 決定）。**本番の NER は 1 つ**の方針なので、判定は単体で行う。

**最終判定は盲検セットと 4 評価セット（`tools/eval_ood_f1.py` の被覆ゼロ列）。**
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--default-variant", default="neg", choices=("neg", "noneg"))
    ap.add_argument("--out", action="append", default=None)
    args = ap.parse_args(argv)
    # v9 版を一時ファイルに作り、v9 → v10 の置換を掛ける
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "v9.ipynb")
        v9nb.main(["--default-variant", args.default_variant, "--out", tmp_path])
        nb = json.load(open(tmp_path, encoding="utf-8"))
    cells = nb["cells"]
    cells[0]["source"] = HEADER.splitlines(keepends=True)
    src5 = "".join(cells[5]["source"])
    for old, new in (('"v9": [_d("data_v9/train_v9.jsonl")]', '"v10": [_d("data_v10/train_v10.jsonl")]'),
                     ('"v9-noneg": [_d("data_v9/train_v9_noneg.jsonl")]', '"v10-noneg": [_d("data_v10/train_v10_noneg.jsonl")]'),
                     ('DATA_TAG    = "v9" if DATA_VARIANT == "neg" else "v9-noneg"', 'DATA_TAG    = "v10" if DATA_VARIANT == "neg" else "v10-noneg"'),
                     ('_d("data_v9/valid_v9.jsonl")', '_d("data_v10/valid_v10.jsonl")'),
                     ("# tools/build_data_v9.py で生成。テンプレート上限 5", "# tools/build_data_v10.py で生成（v9 + 帳票の比率調整）。テンプレート上限 5")):
        assert old in src5, old
        src5 = src5.replace(old, new)
    cells[5]["source"] = src5.splitlines(keepends=True)
    tail = "".join(cells[22]["source"]).replace("data_v9/valid_v9.jsonl", "data_v10/valid_v10.jsonl").replace("文字ヘッドモデル ∪ v9", "（併用は方針外。単体で判定）")
    cells[22]["source"] = tail.splitlines(keepends=True)
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            left = [l for l in s.splitlines() if "data_v9" in l or "train_v9" in l or "valid_v9" in l]
            assert not left, (i, left)
    outs = args.out or [os.path.join(notebooks_dir(), DST_NAME), os.path.join(drive_dir(), DST_NAME)]
    for p in outs:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()

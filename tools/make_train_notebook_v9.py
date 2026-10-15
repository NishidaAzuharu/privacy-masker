"""v8 の学習ノートブック（`make_train_notebook_v8.py` の出力）から v9 版を作る。データ指定・タグ・ヘッダだけを置換する。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_train_notebook_v9.py [--default-variant neg|noneg]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
import make_train_notebook_v8 as v8  # noqa: E402

DST_NAME = "train_ja_mask_charhead_v9_scratch.ipynb"

HEADER = """# 日本語PIIマスキング 文字単位ヘッド — データ v9 / 素の事前学習モデルから

> **この Notebook は 1 回の全セル実行で 1 モデルだけ学習します。**
>
> - `ENCODER_INIT = "cl-nagoya/ruri-v3-30m"`（素の事前学習モデル）
> - データ: 設定セル先頭の `DATA_VARIANT_DEFAULT`。`neg` = `data_v9/train_v9.jsonl`（LLM 負例 2,000 あり、**既定**）、
>   `noneg` = `data_v9/train_v9_noneg.jsonl`。valid は共通の `data_v9/valid_v9.jsonl`
> - `EPOCHS = 10` / `PATIENCE = 3` / エポックごとに `epochNN/` へ保存。**エポック選択は 4 評価セットで行い、
>   後半エポックの平均（`tools/average_checkpoints.py`、ローカル CPU）も候補に入れる**
> - 保存先: `outputs_ruri-v3-30m-charhead-v9-frompre/`（noneg は `…-v9-noneg-frompre/`）

## v8 からの変更（設計は開発記録、非公開）

v8 の切り分け（開発記録、非公開）: LLM 負例は負例文書への過剰マスクを 5〜10 分の 1 にし
OOD の一般語の誤検出を 4 割減らすが、17% では OpenPII の裸の町名と GSD のカタカナ著名人を落とす。英数字（メールの一部）の
誤検出は帳票フレームの `TO: {k}` `cc: {k}` が原因で負例のせいではない。

| 変更 | v8 | v9 |
|---|---|---|
| テンプレート上限 | 6 | **5** |
| LLM 負例 | 4,000（17%） | **2,000（約 10%）**。ハード負例 423 を優先 |
| 帳票フレーム | `TO: {k}` `cc: {k}` | 外す。**メールアドレス・URL を非エンティティの値として追加**、`From: "{k}" <{email}>` |
| カタカナ人名の文書 | なし | **600** |
| 裸の町名（LOCATION）の文書 | なし | **500** |

**最終判定は 4 評価セット（固定OOD / 盲検v1 / GSD / OpenPII）と `tools/eval_ood_f1.py`（被覆ゼロ列）・`tools/eval_all.py`。**
"""

TAIL = """## 9. 学習後にやること（ローカル CPU）

1. `tools/average_checkpoints.py --out <OUTPUT_DIR>/<avg_dir> <OUTPUT_DIR>/epochNN ...` で後半エポックの平均を作る
2. `tools/eval_ood_f1.py` で best・各エポック・平均を 4 評価セット + `data_v9/valid_v9.jsonl` で採点し、
   **被覆ゼロ（`uncovered` 列）** を当時の本番モデル・文字ヘッドモデル・v7・v8 と並べる
3. 採用条件は「PERSON / ADDRESS の完全見逃しが 4 セット全部で本番比悪化ゼロ」。過剰マスクは `tools/eval_all.py` の本番構成で
4. 併用（文字ヘッドモデル ∪ v9）は v8 のときの `union_coverage` スクリプトで
5. 70m は同じデータ・同じ設定で `MASK_NER_ENCODER_INIT=cl-nagoya/ruri-v3-70m` を指定して別実行
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(notebooks_dir(), v8.SRC_NAME), help="v7 のノートブック（v8 と同じ元）")
    ap.add_argument("--default-variant", default="neg", choices=("neg", "noneg"))
    ap.add_argument("--out", action="append", default=None)
    args = ap.parse_args(argv)
    nb = v8.convert(json.load(open(args.src, encoding="utf-8")))
    cells = nb["cells"]
    cells[0]["source"] = HEADER.splitlines(keepends=True)
    cells[22]["source"] = TAIL.splitlines(keepends=True)
    src5 = "".join(cells[5]["source"])
    for old, new in (('DATA_VARIANT_DEFAULT = "noneg"', f'DATA_VARIANT_DEFAULT = "{args.default_variant}"'),
                     ('"v8": [_d("data_v8/train_v8.jsonl")]', '"v9": [_d("data_v9/train_v9.jsonl")]'),
                     ('"v8-noneg": [_d("data_v8/train_v8_noneg.jsonl")]', '"v9-noneg": [_d("data_v9/train_v9_noneg.jsonl")]'),
                     ('DATA_TAG    = "v8" if DATA_VARIANT == "neg" else "v8-noneg"', 'DATA_TAG    = "v9" if DATA_VARIANT == "neg" else "v9-noneg"'),
                     ('_d("data_v8/valid_v8.jsonl")', '_d("data_v9/valid_v9.jsonl")'),
                     ("# tools/build_data_v8.py で生成。テンプレート上限 6", "# tools/build_data_v9.py で生成。テンプレート上限 5"),
                     ("/ 帳票・ログの裸の姓 / 長文 / テンプレート分離 / LLM 負例 4,000（Qwen3.8-27B-FP8）",
                      "/ 帳票・ログの裸の姓（メール・URL は非エンティティ）/ カタカナ人名 / 裸の町名 / 長文 / テンプレート分離 / LLM 負例 2,000")):
        assert old in src5, old
        src5 = src5.replace(old, new)
    cells[5]["source"] = src5.splitlines(keepends=True)
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            left = [l for l in s.splitlines() if "data_v8" in l or "train_v8" in l or "valid_v8" in l]
            assert not left, (i, left)
    outs = args.out or [os.path.join(notebooks_dir(), DST_NAME), os.path.join(drive_dir(), DST_NAME)]
    for p in outs:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()

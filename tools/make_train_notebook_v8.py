"""v7 の学習ノートブックから v8 版 `train_ja_mask_charhead_v8_scratch.ipynb` を作る。

変えるのはデータの指定（`data_v8/`）、負例あり／なしの切り替え（環境変数 `MASK_NER_DATA_VARIANT=noneg`）、
出力先タグ（`-v8` / `-v8-noneg`）、ヘッダと末尾の手順だけ。学習コードは v7 と同一。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_train_notebook_v8.py            # notebooks_dir とデータフォルダ直下に書く
  PYTHONPATH=. .venv/bin/python tools/make_train_notebook_v8.py --check    # 置換対象が v7 に存在するかだけ確認
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402

SRC_NAME = "train_ja_mask_charhead_v7_scratch.ipynb"
DST_NAME = "train_ja_mask_charhead_v8_scratch.ipynb"

HEADER = """# 日本語PIIマスキング 文字単位ヘッド — データ v8 / 素の事前学習モデルから

> **この Notebook は 1 回の全セル実行で 1 モデルだけ学習します。**
>
> - `ENCODER_INIT = "cl-nagoya/ruri-v3-30m"`（素の事前学習モデル。当時の本番モデルからの追加学習ではない）
> - データ: 設定セル先頭の `DATA_VARIANT_DEFAULT` で選ぶ。`neg` = `data_v8/train_v8.jsonl`（LLM 負例あり、1 回目・済）、
>   `noneg` = `data_v8/train_v8_noneg.jsonl`（LLM 負例なし、**2 回目の既定**）。環境変数 `MASK_NER_DATA_VARIANT` があれば優先。
>   valid は共通の `data_v8/valid_v8.jsonl`
> - `EPOCHS = 10` / `PATIENCE = 3` / エポックごとに `epochNN/` へ保存
> - best の選択基準は valid の **PERSON + ADDRESS の未被覆文字数**（同点なら F1）。
>   ただし v7 ではこの指標が OOD の順序を追えなかったので、**エポック選択は 4 評価セットで行う**
> - 保存先: `outputs_ruri-v3-30m-charhead-v8-frompre/`（noneg は `outputs_ruri-v3-30m-charhead-v8-noneg-frompre/`）

## v7 からの変更（設計は開発記録、非公開）

v7 は盲検 v1 のハンドル名・短名・半角カナを直したが、固定OOD 危険 14 → 24、GSD 21 → 32 と悪化した。
落ちたのは公知の LOCATION / ORG と、帳票・ログの裸の姓（`operator=蒼波` `記録者：夜勤 朱鷺野`）。
負例は 1,035 件（3.6%）しかなく、併用構成の過剰マスクは本番比 +14〜31%。

| 変更 | v7 | v8 |
|---|---|---|
| テンプレート上限 | 8 | **6**（実テキスト out_ner の比重を上げる） |
| チャット / 会話 / 半角カナの新規文書 | 1,500 / 1,500 / 500 | **750 / 750 / 250** |
| 帳票・ログのフレーム（`key=value` `ラベル: 値` の裸の姓、同フレームの役割語・状態語が対の負例） | なし | **1,100 文書** |
| LLM 生成の負例（Qwen3.8-27B-FP8、PII を含まない文書） | なし | **4,000 文書**（train 3,600 / valid 400） |

**手順**: 1 回目（`neg`、2026-09-15 済、結果は開発記録、非公開）は単体で採用条件を満たさず、
GSD・OpenPII のカタカナ・外国名・平仮名名が v7 から後退した。2 回目は `noneg` で同じ設定を回し、
その後退と負例の効き目（v8 valid の負例への過剰マスク 247 → 25 字）が負例由来かを切り分ける。

**最終判定は 4 評価セット（固定OOD / 盲検v1 / GSD / OpenPII）と `tools/eval_ood_f1.py`
（被覆ゼロ列）・`tools/eval_all.py` で行う。** 学習後にローカルで回す。
"""

OLD_DATA = '''DATA_CONFIGS = {
    # tools/build_data_v7.py で生成。テンプレート上限 8 / 住所・人名の書式変種 / チャット・会話・半角カナ / 長文 / テンプレート分離
    "v7": [_d("data_v7/train_v7.jsonl")],
}
DATA_TAG    = next(iter(DATA_CONFIGS))
TRAIN_PATHS = DATA_CONFIGS[DATA_TAG]
VALID_PATH  = _d("data_v7/valid_v7.jsonl")   # 事前分割（train とテンプレート重複 0）。None にしないこと
'''
NEW_DATA = '''# tools/build_data_v8.py で生成。テンプレート上限 6 / 住所・人名の書式変種 / チャット・会話・半角カナ（v7 の半分）
# / 帳票・ログの裸の姓 / 長文 / テンプレート分離 / LLM 負例 4,000（Qwen3.8-27B-FP8）
# LLM 負例あり = "neg"（1 回目、済）/ なし = "noneg"（2 回目、切り分け用）。環境変数 MASK_NER_DATA_VARIANT が優先
DATA_VARIANT_DEFAULT = "noneg"  # @param ["neg", "noneg"]
DATA_VARIANT = os.environ.get("MASK_NER_DATA_VARIANT", DATA_VARIANT_DEFAULT).strip().lower()
if DATA_VARIANT not in ("neg", "noneg"):
    raise ValueError(f"MASK_NER_DATA_VARIANT は neg か noneg（受け取った値: {DATA_VARIANT!r}）")
DATA_CONFIGS = {
    "v8": [_d("data_v8/train_v8.jsonl")],
    "v8-noneg": [_d("data_v8/train_v8_noneg.jsonl")],
}
DATA_TAG    = "v8" if DATA_VARIANT == "neg" else "v8-noneg"
TRAIN_PATHS = DATA_CONFIGS[DATA_TAG]
VALID_PATH  = _d("data_v8/valid_v8.jsonl")   # 事前分割（train とテンプレート重複 0）。None にしないこと
print("DATA_TAG     :", DATA_TAG, "(LLM 負例", "あり" if DATA_VARIANT == "neg" else "なし", ")")
'''

TAIL = """## 9. 学習後にやること（ローカル CPU）

1. `tools/eval_ood_f1.py` で 4 評価セット + `data_v8/valid_v8.jsonl` を採点し、
   **被覆ゼロ（`uncovered_gold.csv` / summary の `uncovered` 列）** を当時の本番モデル・文字ヘッドモデル・v7 と並べる。
   F1 だけで判断しない。`false_negatives` は 1 対 1 照合の未マッチで安全性の指標ではない。
2. `epochNN/` を全部同じ手順で採点し、OOD で最良のエポックを選ぶ（valid の選択指標は v7 で OOD を追えなかった）。
3. 採用条件は「PERSON / ADDRESS の完全見逃しが 4 セット全部で本番比悪化ゼロ」。過剰マスクは
   `tools/eval_all.py` の本番構成で本番・char 8ep ∪ v7 と比べる（負例の効果はここに出る）。
4. 採用条件を満たさなければ `MASK_NER_DATA_VARIANT=noneg` で再走し、負例の有無で切り分ける。
5. 70m は同じデータ・同じ設定で `MASK_NER_ENCODER_INIT=cl-nagoya/ruri-v3-70m` を指定して別実行。
"""


def convert(nb: dict) -> dict:
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown"
    cells[0]["source"] = HEADER.splitlines(keepends=True)
    src5 = "".join(cells[5]["source"])
    assert OLD_DATA in src5, "v7 の DATA_CONFIGS ブロックが見つかりません"
    src5 = src5.replace(OLD_DATA, NEW_DATA)
    cells[5]["source"] = src5.splitlines(keepends=True)
    assert cells[22]["cell_type"] == "markdown"
    cells[22]["source"] = TAIL.splitlines(keepends=True)
    # 学習コード側に v7 固有の参照が残っていないか
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        leftovers = [l for l in s.splitlines() if "data_v7" in l or "train_v7" in l or "valid_v7" in l]
        assert not leftovers, (i, leftovers)
        c["outputs"] = []
        c["execution_count"] = None
    return nb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(notebooks_dir(), SRC_NAME))
    ap.add_argument("--out", action="append", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    nb = json.load(open(args.src, encoding="utf-8"))
    nb = convert(nb)
    if args.check:
        print("OK: 置換対象あり、v7 固有の参照なし")
        return
    outs = args.out or [os.path.join(notebooks_dir(), DST_NAME), os.path.join(drive_dir(), DST_NAME)]
    for p in outs:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()

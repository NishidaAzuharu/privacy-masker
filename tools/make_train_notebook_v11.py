"""v10 の学習ノート生成器を使って v11 版 `train_ja_mask_charhead_v11_scratch.ipynb` を作る。
変更点: データ指定（v11 / v11-plus-old）、`neutral` を損失と過剰マスク計算から外す、ベースモデルのリビジョン固定、manifest への記録。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_train_notebook_v11.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
import make_train_notebook_v10 as v10nb  # noqa: E402

DST_NAME = "train_ja_mask_charhead_v11_scratch.ipynb"
ENCODER_REVISION = "24899e5de370b56d179604a007c0d727bf144504"
# 同じデータ・同じ手順で encoder だけ替える版（2026-09-23 ユーザー依頼: 30m の容量限界かを見る）
ENCODERS = {
    "30m": {"id": "cl-nagoya/ruri-v3-30m", "revision": ENCODER_REVISION, "dst": DST_NAME, "default_variant": "v11"},
    "70m": {"id": "cl-nagoya/ruri-v3-70m", "revision": "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc",   # 2025-04-17 の main
            "dst": "train_ja_mask_charhead_v11_70m_scratch.ipynb", "default_variant": "plus_old"},
}

HEADER = """# 日本語PIIマスキング 文字単位ヘッド — データ v11 / 素の事前学習モデルから

> **この Notebook は 1 回の全セル実行で 1 モデルだけ学習します。**
>
> - `ENCODER_INIT = "cl-nagoya/ruri-v3-30m"`（リビジョン `24899e5de370b56d179604a007c0d727bf144504` に固定）
> - データ: 設定セル先頭の `DATA_VARIANT_DEFAULT`。`v11` = `data_v11/train_v11.jsonl`（**既定**。Qwen テンプレート合成 + GSD 学習分割 + Wikipedia + LLM 負例）、
>   `plus_old` = `data_v11/train_v11_plus_old.jsonl`（v10 の合成層を足した切り分け用）。valid は共通の `data_v11/valid_v11.jsonl`
> - `neutral` 範囲（GSD の写像しない型）は損失を計算せず、valid の過剰マスク字数からも除く
> - `EPOCHS = 10` / `PATIENCE = 3` / エポックごとに `epochNN/` へ保存。学習後に `tools/average_checkpoints.py` で「best 以降全部」「後半 3」の平均を作る
> - 保存先: `outputs_ruri-v3-30m-charhead-v11-frompre/` / `outputs_ruri-v3-30m-charhead-v11-plus-old-frompre/`

設計は開発記録（非公開）。**判定は盲検セット（悪化ゼロ）を主、固定OOD・OpenPII・盲検 v1 は完全見逃し +1 まで。GSD テストは分布内として参考。**
"""

NEUTRAL_NORMALIZE = '''        neutral = []
        for it in (rec.get("neutral", []) or []):
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                try:
                    ns, ne = max(0, int(it[0])), min(len(text), int(it[1]))
                except (TypeError, ValueError):
                    continue
                if ns < ne:
                    neutral.append([ns, ne])
'''
NEUTRAL_LABELS = '''            for ns, ne in (r.get("neutral", []) or []):        # 採点対象外の範囲は O とも教えない（GSD の NORP 等）
                for i in range(max(ns, lo) - lo, min(ne, hi) - lo):
                    if f["labels"][i] == ch.LABEL2ID["O"]:
                        f["labels"][i] = ch.IGNORE_INDEX
'''
NEUTRAL_MASK = '''        if neutral_docs:
            for ns, ne in neutral_docs[di]:
                masked.difference_update(range(ns, ne))
'''


def _replace(src: str, old: str, new: str, count: int = 1) -> str:
    assert old in src, old
    return src.replace(old, new) if count == 0 else src.replace(old, new, count)


def transform(nb: dict, encoder: str = "30m") -> dict:
    enc = ENCODERS[encoder]
    cells = nb["cells"]
    header = HEADER
    if encoder != "30m":
        header = header.replace('`ENCODER_INIT = "cl-nagoya/ruri-v3-30m"`（リビジョン `24899e5de370b56d179604a007c0d727bf144504` に固定）',
                                f'`ENCODER_INIT = "{enc["id"]}"`（リビジョン `{enc["revision"]}` に固定。**30m と同じデータ・同じ手順で encoder だけ 70m**）')
        header = header.replace("`outputs_ruri-v3-30m-charhead-v11-frompre/` / `outputs_ruri-v3-30m-charhead-v11-plus-old-frompre/`",
                                "`outputs_ruri-v3-70m-charhead-v11-frompre/` / `outputs_ruri-v3-70m-charhead-v11-plus-old-frompre/`")
        header = header.replace("# 日本語PIIマスキング 文字単位ヘッド — データ v11 / 素の事前学習モデルから",
                                "# 日本語PIIマスキング 文字単位ヘッド — データ v11 / 素の事前学習モデル **70m** から")
    cells[0]["source"] = header.splitlines(keepends=True)
    src = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
    # --- 設定セル（cell 5）
    s5 = src[5]
    s5 = _replace(s5, 'ENCODER_INIT = os.environ.get("MASK_NER_ENCODER_INIT", "cl-nagoya/ruri-v3-30m")',
                  'ENCODER_INIT = os.environ.get("MASK_NER_ENCODER_INIT", "cl-nagoya/ruri-v3-30m")\n'
                  f'ENCODER_REVISION = None if os.path.isdir(ENCODER_INIT) else "{ENCODER_REVISION}"   # HF の commit に固定\n'
                  'def _rev_kwargs():\n    return {"revision": ENCODER_REVISION} if ENCODER_REVISION else {}')
    s5 = _replace(s5, 'DATA_VARIANT_DEFAULT = "neg"  # @param ["neg", "noneg"]', f'DATA_VARIANT_DEFAULT = "{enc["default_variant"]}"  # @param ["v11", "plus_old"]')
    if encoder != "30m":
        s5 = _replace(s5, 'ENCODER_INIT = os.environ.get("MASK_NER_ENCODER_INIT", "cl-nagoya/ruri-v3-30m")\n'
                          f'ENCODER_REVISION = None if os.path.isdir(ENCODER_INIT) else "{ENCODER_REVISION}"',
                      f'ENCODER_INIT = os.environ.get("MASK_NER_ENCODER_INIT", "{enc["id"]}")\n'
                      f'ENCODER_REVISION = None if os.path.isdir(ENCODER_INIT) else "{enc["revision"]}"')
    s5 = _replace(s5, 'if DATA_VARIANT not in ("neg", "noneg"):', 'if DATA_VARIANT not in ("v11", "plus_old"):')
    s5 = _replace(s5, 'raise ValueError(f"MASK_NER_DATA_VARIANT は neg か noneg（受け取った値: {DATA_VARIANT!r}）")',
                  'raise ValueError(f"MASK_NER_DATA_VARIANT は v11 か plus_old（受け取った値: {DATA_VARIANT!r}）")')
    s5 = _replace(s5, '"v10": [_d("data_v10/train_v10.jsonl")],', '"v11": [_d("data_v11/train_v11.jsonl")],')
    s5 = _replace(s5, '"v10-noneg": [_d("data_v10/train_v10_noneg.jsonl")],', '"v11-plus-old": [_d("data_v11/train_v11_plus_old.jsonl")],')
    s5 = _replace(s5, 'DATA_TAG    = "v10" if DATA_VARIANT == "neg" else "v10-noneg"', 'DATA_TAG    = "v11" if DATA_VARIANT == "v11" else "v11-plus-old"')
    s5 = _replace(s5, '_d("data_v10/valid_v10.jsonl")', '_d("data_v11/valid_v11.jsonl")')
    s5 = _replace(s5, 'print("DATA_TAG     :", DATA_TAG, "(LLM 負例", "あり" if DATA_VARIANT == "neg" else "なし", ")")',
                  'print("DATA_TAG     :", DATA_TAG, "(v10 合成層", "なし" if DATA_VARIANT == "v11" else "あり", ")")')
    s5 = s5.replace("# tools/build_data_v10.py で生成（v9 + 帳票の比率調整）。テンプレート上限 5", "# tools/build_data_v11.py で生成（Qwen テンプレート合成 + GSD + Wikipedia + 負例）")
    cells[5]["source"] = s5.splitlines(keepends=True)
    # --- 読み込みセル（cell 9）: neutral を保持
    s9 = src[9]
    s9 = _replace(s9, '        ents = [e2 for e2 in ents if not (tuple(e2) in seen or seen.add(tuple(e2)))]\n',
                  '        ents = [e2 for e2 in ents if not (tuple(e2) in seen or seen.add(tuple(e2)))]\n' + NEUTRAL_NORMALIZE)
    s9 = _replace(s9, 'out.append({"id": str(rec.get("id", f"{source}-{i}")), "text": text, "entities": ents})',
                  'out.append({"id": str(rec.get("id", f"{source}-{i}")), "text": text, "entities": ents, "neutral": neutral})')
    cells[9]["source"] = s9.splitlines(keepends=True)
    # --- エンコードセル（cell 11）: neutral を IGNORE、tokenizer のリビジョン
    s11 = src[11]
    s11 = _replace(s11, '            for a, b in straddle:                             # 切られた部分は O とも教えない\n                for i in range(a, b):\n                    f["labels"][i] = ch.IGNORE_INDEX\n',
                   '            for a, b in straddle:                             # 切られた部分は O とも教えない\n                for i in range(a, b):\n                    f["labels"][i] = ch.IGNORE_INDEX\n' + NEUTRAL_LABELS)
    s11 = _replace(s11, 'tok = AutoTokenizer.from_pretrained(path, use_fast=True)', 'tok = AutoTokenizer.from_pretrained(path, use_fast=True, **_rev_kwargs())')
    cells[11]["source"] = s11.splitlines(keepends=True)
    # --- モデル・指標セル（cell 13）: encoder のリビジョン、mask_metrics が neutral を過剰マスクから除く
    s13 = src[13]
    s13 = _replace(s13, 'ENCODER_INIT, char_dim=CHAR_DIM, kernel_size=KERNEL_SIZE,', 'ENCODER_INIT, **_rev_kwargs(), char_dim=CHAR_DIM, kernel_size=KERNEL_SIZE,')
    s13 = _replace(s13, 'ref, load_info = AutoModel.from_pretrained(encoder_init, output_loading_info=True)',
                   'ref, load_info = AutoModel.from_pretrained(encoder_init, output_loading_info=True, **_rev_kwargs())')
    s13 = _replace(s13, 'def mask_metrics(gold_docs, pred_docs, mask_types=None, min_score=None, min_chars=None,\n                 gold_types=None):',
                   'def mask_metrics(gold_docs, pred_docs, mask_types=None, min_score=None, min_chars=None,\n                 gold_types=None, neutral_docs=None):')
    s13 = _replace(s13, '    for gs, ps in zip(gold_docs, pred_docs):\n        masked = set()\n        for s, e, _lab in masked_spans(ps, mask_types, min_score, min_chars):\n            masked.update(range(s, e))\n',
                   '    for di, (gs, ps) in enumerate(zip(gold_docs, pred_docs)):\n        masked = set()\n        for s, e, _lab in masked_spans(ps, mask_types, min_score, min_chars):\n            masked.update(range(s, e))\n' + NEUTRAL_MASK)
    cells[13]["source"] = s13.splitlines(keepends=True)
    # --- 学習・評価セル（cell 15）: score_predictions が records の neutral を渡す
    s15 = src[15]
    s15 = _replace(s15, '    m.update(mask_metrics(gd, preds))', '    _neutral = [r.get("neutral", []) for r in records]\n    m.update(mask_metrics(gd, preds, neutral_docs=_neutral))')
    s15 = _replace(s15, 'mask_metrics(gd, preds, gold_types=("PERSON", "ADDRESS")).items()})',
                   'mask_metrics(gd, preds, gold_types=("PERSON", "ADDRESS"), neutral_docs=_neutral).items()})')
    cells[15]["source"] = s15.splitlines(keepends=True)
    # --- manifest セル（cell 21）
    s21 = _replace(src[21], '"encoder_init": ENCODER_INIT,', '"encoder_init": ENCODER_INIT, "encoder_revision": ENCODER_REVISION,', count=0)
    cells[21]["source"] = s21.splitlines(keepends=True)
    # --- 末尾の説明セル（cell 22）と残存チェック
    tail = "".join(cells[22]["source"]).replace("data_v10/valid_v10.jsonl", "data_v11/valid_v11.jsonl")
    cells[22]["source"] = tail.splitlines(keepends=True)
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            left = [l for l in s.splitlines() if "data_v10" in l or "train_v10" in l or "valid_v10" in l]
            assert not left, (i, left)
    return nb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=None)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="30m", help="30m（既定）か 70m。70m は既定の変種が plus_old")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "v10.ipynb")
        v10nb.main(["--default-variant", "neg", "--out", tmp_path])
        with open(tmp_path, encoding="utf-8") as f:
            nb = json.load(f)
    nb = transform(nb, args.encoder)
    dst = ENCODERS[args.encoder]["dst"]
    outs = args.out or [os.path.join(notebooks_dir(), dst), os.path.join(drive_dir(), dst)]
    for p in outs:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()

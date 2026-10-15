"""学習ノートブック（train_ja_mask_model_v3.ipynb）のローカル CPU スモークテスト。

Colab で GPU 時間を使う前に、データ構成（DATA_CONFIGS）の読み込み・トークナイズ・1 エポックの学習・保存が通ることを確認する。
コードセルを順に exec し、`!`/`%` 行と google.colab を含むセルは飛ばす。環境変数はノートブック側の規約に従う:
  MASK_NER_DRIVE_BASE / MASK_NER_OUT_ROOT / MASK_NER_SMOKE_LIMIT / MASK_NER_SMOKE_MODELS

usage:
  PYTHONPATH=. .venv/bin/python tools/smoke_train_nb.py [train_ja_mask_model_v3.ipynb] [--limit 200] [--model <local model dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", nargs="?", default="train_ja_mask_model_v3.ipynb")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default=None, help="BASE_MODELS の差し替え（ローカルの学習済みモデルディレクトリ等）")
    ap.add_argument("--out", default=None, help="保存先ルート（既定: スクラッチ）")
    args = ap.parse_args()
    nb_path = args.notebook if os.path.isabs(args.notebook) else os.path.join(notebooks_dir(), args.notebook)
    gd = drive_dir()
    os.environ["MASK_NER_DRIVE_BASE"] = gd
    os.environ["MASK_NER_OUT_ROOT"] = args.out or os.path.join(os.environ.get("TMPDIR", "/tmp"), "mask_ner_smoke")
    os.environ["MASK_NER_SMOKE_LIMIT"] = str(args.limit)
    if args.model:
        os.environ["MASK_NER_SMOKE_MODELS"] = args.model
    os.makedirs(os.environ["MASK_NER_OUT_ROOT"], exist_ok=True)
    nb = json.load(open(nb_path))
    g = {"__name__": "__main__"}
    try:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            lines = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))]
            src = "\n".join(lines)
            if "google.colab" in src or not src.strip():
                print(f"[skip] cell {i}", flush=True); continue
            print(f"[run ] cell {i}", flush=True)
            exec(compile(src, f"<cell {i}>", "exec"), g)
        print("[done] 全セル完了 → 保存先:", os.environ["MASK_NER_OUT_ROOT"], flush=True)
        os._exit(0)
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()

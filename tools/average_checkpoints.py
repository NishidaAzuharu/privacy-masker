"""文字ヘッドモデルのチェックポイント平均（重み空間の単純平均）。GPU 不要。

以前の学習走ではエポック間の揺れ（固定OOD の見逃し 14〜17、過剰マスク 183〜382）が学習データの差と同じ大きさで、
valid の選択指標は OOD の最良点を指さなかった（開発記録、非公開）。
同じ走の後半エポックを平均すると揺れが小さくなるかを、学習し直さずに確かめる。

`save_char_model` の保存形式（`model.safetensors` = encoder、`char_head.safetensors` = ヘッド、`char_head_config.json`）を
そのまま読み、float テンソルを平均して同じ形式で書く。tokenizer と config.json は先頭のディレクトリから写す。

usage:
  PYTHONPATH=. .venv/bin/python tools/average_checkpoints.py --out <dir> <epoch_dir> <epoch_dir> [...] [--weights 1,1,2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.getcwd())

HEAD = "char_head.safetensors"
ENCODER = "model.safetensors"
HEAD_CFG = "char_head_config.json"
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
              "sentencepiece.bpe.model", "tokenizer.model", "added_tokens.json")
KEEP_CFG_KEYS = ("char_dim", "kernel_size", "use_mix", "num_labels", "label_list", "n_intra", "n_script",
                 "added_parameters", "dropout", "class_weights")


def average_tensors(paths: Sequence[str], weights: Sequence[float]) -> dict[str, torch.Tensor]:
    states = [load_file(p) for p in paths]
    keys = list(states[0].keys())
    for p, sd in zip(paths[1:], states[1:]):
        if set(sd) != set(keys):
            raise SystemExit(f"テンソルの名前が一致しません: {paths[0]} vs {p}")
    total = float(sum(weights))
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        ts = [sd[k] for sd in states]
        if any(t.shape != ts[0].shape for t in ts):
            raise SystemExit(f"テンソル {k} の形が一致しません")
        if ts[0].is_floating_point():
            acc = torch.zeros_like(ts[0], dtype=torch.float64)
            for t, w in zip(ts, weights):
                acc += t.to(torch.float64) * (w / total)
            out[k] = acc.to(ts[0].dtype).contiguous()
        else:
            if any(not torch.equal(t, ts[0]) for t in ts[1:]):
                raise SystemExit(f"非 float テンソル {k} が一致しません（平均できない）")
            out[k] = ts[0].contiguous()
    return out


def average_checkpoints(dirs: Sequence[str], out_dir: str, weights: Sequence[float] | None = None) -> dict:
    if len(dirs) < 2:
        raise SystemExit("2 つ以上のチェックポイントが必要")
    weights = list(weights) if weights else [1.0] * len(dirs)
    if len(weights) != len(dirs):
        raise SystemExit("--weights の数がディレクトリ数と一致しません")
    for d in dirs:
        for name in (HEAD, ENCODER, HEAD_CFG):
            if not os.path.exists(os.path.join(d, name)):
                raise SystemExit(f"{name} がありません: {d}")
    cfgs = [json.load(open(os.path.join(d, HEAD_CFG), encoding="utf-8")) for d in dirs]
    for d, c in zip(dirs[1:], cfgs[1:]):
        for k in ("char_dim", "kernel_size", "use_mix", "num_labels", "label_list", "n_intra", "n_script"):
            if c.get(k) != cfgs[0].get(k):
                raise SystemExit(f"ヘッド構成 {k} が一致しません: {dirs[0]} vs {d}")
    os.makedirs(out_dir, exist_ok=True)
    save_file(average_tensors([os.path.join(d, ENCODER) for d in dirs], weights), os.path.join(out_dir, ENCODER),
              metadata={"format": "pt"})
    save_file(average_tensors([os.path.join(d, HEAD) for d in dirs], weights), os.path.join(out_dir, HEAD))
    for name in COPY_FILES:
        src = os.path.join(dirs[0], name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    cfg = {k: cfgs[0][k] for k in KEEP_CFG_KEYS if k in cfgs[0]}
    cfg["epoch"] = "avg(" + ",".join(str(c.get("epoch", "?")) for c in cfgs) + ")"
    cfg["averaged_from"] = [os.path.abspath(d) for d in dirs]
    cfg["average_weights"] = weights
    cfg["data_tag"] = cfgs[0].get("data_tag")
    with open(os.path.join(out_dir, HEAD_CFG), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def verify(out_dir: str) -> None:
    from privacy_masker.detectors import CharHeadNerDetector
    det = CharHeadNerDetector(model=out_dir, device="cpu")
    text = "担当の山田太郎は東京都港区1-2-3の株式会社アオバに連絡した。記録者：夜勤 朱鷺野"
    ents = det.detect(text)
    print("verify:", [(e.entity_type, text[e.start:e.end]) for e in ents])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="カンマ区切りの重み（既定は等重み）")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    cfg = average_checkpoints(args.dirs, args.out, weights)
    print("wrote", args.out, "|", cfg["epoch"], "| weights", cfg["average_weights"])
    if not args.no_verify:
        verify(args.out)


if __name__ == "__main__":
    main()

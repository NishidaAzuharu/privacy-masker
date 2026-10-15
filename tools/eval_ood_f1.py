"""OOD セットでのエンティティ単位 F1 とエラー分析ファイル。

照合手順は `train_ja_mask_model (1).ipynb` のエラー分析セルと同じ:

  1. **厳密一致**（start / end / label がすべて一致）を 1 対 1 で確定して TP
  2. 残りを IoU 最大で 1 対 1 マッチ
     - 同じ label で span が違う → **境界ズレ**（FP と FN の両方に数える）
     - label が違う → **型混同**（FP と FN の両方に数える）
  3. 未マッチの予測 → FP、未マッチの gold → FN

ラベル無視版（span だけ厳密一致）も同じ走査から出す。

**部分注釈セットの扱い（ノートブックには無い処理）**: OpenPII は gold 以外の
159,433 文字すべてが neutral に入っている。gold が網羅的でないセットで
neutral 内の予測を FP に数えると precision が無意味になるので、
**予測 span の文字がすべて neutral に入っているものは FP から除く**（件数は報告する）。

段階を分けて測る（モデルの見逃しと、後段が削った漏れを分離する）:

  - ``model``   : Detector の出力そのまま
  - ``refined`` : Detector + MorphologicalRefiner（本番設定）

usage（リポジトリ直下で）::

    PYTHONPATH=. .venv/bin/python tools/eval_ood_f1.py <model_dir または Hub ID> "表示名=<dir>/best" \\
        --extra 住所書式=<path>.jsonl --example-limit 0 --out-dir <dir>

固定 OOD・外部 GSD・外部 OpenPII・盲検 v1 は自動で読む。それ以外は ``--extra NAME=PATH``。
モデルはディレクトリパス、Hugging Face Hub のリポジトリ ID（owner/name）、データフォルダ内の登録名
（`diff_danger.PATHS` / `eval_in_distribution.PATHS`）、または ``表示名=パス``。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import Counter
from typing import Iterable, Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
from eval_all import build_detector, load_external, model_path  # noqa: E402
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402

LABELS: tuple[str, ...] = ("ADDRESS", "FACILITY", "LOCATION", "ORG", "PERSON")
EXAMPLE_LIMIT = 2000        # 0 で無制限（write_files の引数で上書きする）
CONTEXT = 40


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return p, r, _safe_div(2 * p * r, p + r)


class Analysis:
    """1 つの (セット, モデル, 段階) の集計とエラー例。"""

    def __init__(self) -> None:
        self.tp: Counter = Counter()
        self.fp: Counter = Counter()
        self.fn: Counter = Counter()
        self.boundary: Counter = Counter()
        self.type_conf: Counter = Counter()
        self.boundary_examples: list[dict] = []
        self.type_conf_examples: list[dict] = []
        self.false_negatives: list[dict] = []
        self.false_positives: list[dict] = []
        # 被覆ゼロ（どの予測とも 1 文字も重ならない gold）= 製品としての完全見逃し
        self.uncovered: Counter = Counter()
        self.uncovered_examples: list[dict] = []
        # ラベル無視（span だけ厳密一致）
        self.bin_tp = self.bin_fp = self.bin_fn = 0
        self.n_docs = 0
        self.neutral_skipped = 0

    # ---- 指標 ----
    def labels(self) -> list[str]:
        return sorted(set(self.tp) | set(self.fp) | set(self.fn))

    def micro(self) -> tuple[float, float, float]:
        return _prf(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()))

    def macro_f1(self) -> float:
        labs = self.labels()
        if not labs:
            return 0.0
        return sum(_prf(self.tp[l], self.fp[l], self.fn[l])[2] for l in labs) / len(labs)

    def binary(self) -> tuple[float, float, float]:
        return _prf(self.bin_tp, self.bin_fp, self.bin_fn)


def analyse(docs: Sequence[dict], preds: Sequence[Sequence], *,
            use_neutral: bool = True) -> Analysis:
    a = Analysis()
    for di, (d, pr) in enumerate(zip(docs, preds)):
        text = d["text"]
        neutral: set[int] = set()
        if use_neutral:
            for s, e in d.get("neutral", []) or []:
                neutral.update(range(s, e))
        gold = sorted(({"start": s, "end": e, "label": l, "text": text[s:e]}
                       for s, e, l in d["entities"] if l in LABELS),
                      key=lambda x: (x["start"], x["end"]))
        pred = []
        for p in pr:
            if p.entity_type not in LABELS:
                continue
            # 部分注釈セットでは neutral 内の予測を FP に数えない（gold が網羅的でない）
            if neutral and set(range(p.start, p.end)) <= neutral:
                a.neutral_skipped += 1
                continue
            pred.append({"start": p.start, "end": p.end, "label": p.entity_type,
                         "text": text[p.start:p.end]})
        pred.sort(key=lambda x: (x["start"], x["end"]))
        a.n_docs += 1

        # 1) 厳密一致
        used_g: set[int] = set()
        used_p: set[int] = set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in used_g:
                    continue
                if (p["start"], p["end"], p["label"]) == (g["start"], g["end"], g["label"]):
                    a.tp[p["label"]] += 1
                    used_g.add(gi)
                    used_p.add(pi)
                    break

        # 2) 残りを IoU 最大で 1 対 1
        matches = []
        for pi, p in enumerate(pred):
            if pi in used_p:
                continue
            best = (pi, -1, 0.0)
            for gi, g in enumerate(gold):
                if gi in used_g:
                    continue
                iou = _iou((p["start"], p["end"]), (g["start"], g["end"]))
                if iou > best[2]:
                    best = (pi, gi, iou)
            if best[2] > 0.0:
                matches.append(best)
        matches.sort(key=lambda x: -x[2])
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        for pi, gi, iou in matches:
            if gi in matched_g or pi in matched_p:
                continue
            p, g = pred[pi], gold[gi]
            ex = {"doc": di, "text": text, "pred": p, "gold": g, "iou": iou}
            if p["label"] == g["label"]:
                a.boundary[p["label"]] += 1
                a.boundary_examples.append(ex)
            else:
                a.type_conf[(p["label"], g["label"])] += 1
                a.type_conf_examples.append(ex)
            a.fp[p["label"]] += 1
            a.fn[g["label"]] += 1
            matched_g.add(gi)
            matched_p.add(pi)
        used_g |= matched_g
        used_p |= matched_p

        # 3) 未マッチ
        for pi, p in enumerate(pred):
            if pi not in used_p:
                a.fp[p["label"]] += 1
                a.false_positives.append({"doc": di, "text": text, "pred": p})
        for gi, g in enumerate(gold):
            if gi not in used_g:
                a.fn[g["label"]] += 1
                a.false_negatives.append({"doc": di, "text": text, "gold": g})

        # 被覆ゼロの gold。false_negatives は 1 対 1 照合で「相手を取られた」gold も含む
        # （1 つの予測が隣接する 2 つの gold にまたがると片方が未マッチになる）ので、
        # 文字が実際に隠れたかは型を問わず被覆で数える。安全性はこちらで見る
        cov: set[int] = set()
        for p in pred:
            cov.update(range(p["start"], p["end"]))
        for g in gold:
            if not (cov & set(range(g["start"], g["end"]))):
                a.uncovered[g["label"]] += 1
                a.uncovered_examples.append({"doc": di, "text": text, "gold": g})

        # ラベル無視（span だけ厳密一致）
        gs = Counter((g["start"], g["end"]) for g in gold)
        ps = Counter((p["start"], p["end"]) for p in pred)
        hit = sum((gs & ps).values())
        a.bin_tp += hit
        a.bin_fp += sum(ps.values()) - hit
        a.bin_fn += sum(gs.values()) - hit
    return a


# ---------------------------------------------------------------- 出力

def _context(text: str, s: int, e: int) -> str:
    lo, hi = max(0, s - CONTEXT), min(len(text), e + CONTEXT)
    return text[lo:hi].replace("\n", "\\n").replace("\r", "")


def write_files(a: Analysis, out_dir: str, limit: int = EXAMPLE_LIMIT) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = limit if limit > 0 else None

    with open(os.path.join(out_dir, "per_label_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "tp", "fp", "fn", "precision", "recall", "f1",
                    "boundary_errors", "gold_total"])
        for lab in a.labels():
            p, r, f1 = _prf(a.tp[lab], a.fp[lab], a.fn[lab])
            w.writerow([lab, a.tp[lab], a.fp[lab], a.fn[lab], f"{p:.6f}", f"{r:.6f}",
                        f"{f1:.6f}", a.boundary[lab], a.tp[lab] + a.fn[lab]])

    with open(os.path.join(out_dir, "top_confusions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred_label", "gold_label", "count"])
        for (p, g), c in a.type_conf.most_common():
            w.writerow([p, g, c])

    def _pairs(path: str, examples: Iterable[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_index", "context", "pred_start", "pred_end", "pred_label",
                        "pred_text", "gold_start", "gold_end", "gold_label", "gold_text",
                        "iou", "text"])
            for ex in list(examples)[:cap]:
                p, g = ex["pred"], ex["gold"]
                w.writerow([ex["doc"], _context(ex["text"], g["start"], g["end"]),
                            p["start"], p["end"], p["label"], p["text"],
                            g["start"], g["end"], g["label"], g["text"],
                            f"{ex['iou']:.4f}", ex["text"]])

    _pairs(os.path.join(out_dir, "boundary_examples.csv"), a.boundary_examples)
    _pairs(os.path.join(out_dir, "type_confusion_examples.csv"), a.type_conf_examples)

    def _gold_rows(path: str, examples: Iterable[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_index", "context", "gold_start", "gold_end", "gold_label",
                        "gold_text", "gold_len", "text"])
            for ex in list(examples)[:cap]:
                g = ex["gold"]
                w.writerow([ex["doc"], _context(ex["text"], g["start"], g["end"]),
                            g["start"], g["end"], g["label"], g["text"],
                            g["end"] - g["start"], ex["text"]])

    # 1 対 1 照合で未マッチになった gold（境界ズレ・型混同で相手を取られたものを含む）
    _gold_rows(os.path.join(out_dir, "false_negatives.csv"), a.false_negatives)
    # 被覆ゼロの gold（PII では最重要のエラー種。安全性はこちらで見る）
    _gold_rows(os.path.join(out_dir, "uncovered_gold.csv"), a.uncovered_examples)

    with open(os.path.join(out_dir, "false_positives.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_index", "context", "pred_start", "pred_end", "pred_label",
                    "pred_text", "pred_len", "text"])
        for ex in a.false_positives[:cap]:
            p = ex["pred"]
            w.writerow([ex["doc"], _context(ex["text"], p["start"], p["end"]),
                        p["start"], p["end"], p["label"], p["text"],
                        p["end"] - p["start"], ex["text"]])


def print_analysis(title: str, a: Analysis) -> None:
    p, r, f1 = a.micro()
    bp, br, bf1 = a.binary()
    print(f"\n====== {title} ======")
    print(f"#docs={a.n_docs}  gold={sum(a.tp.values()) + sum(a.fn.values())}  "
          f"neutral 内で FP から除いた予測={a.neutral_skipped}")
    print(f"Micro P/R/F1 (strict): P={p:.4f}  R={r:.4f}  F1={f1:.4f}   "
          f"Macro F1={a.macro_f1():.4f}")
    print(f"ラベル無視 (span 厳密一致): P={bp:.4f}  R={br:.4f}  F1={bf1:.4f}  "
          f"TP={a.bin_tp} FP={a.bin_fp} FN={a.bin_fn}")
    print(f"被覆ゼロの gold（完全見逃し）: {sum(a.uncovered.values())}  "
          + "  ".join(f"{l}={a.uncovered[l]}" for l in LABELS if a.uncovered[l])
          + f"   ／ 1対1照合で未マッチの gold: {len(a.false_negatives)}")
    print("---- ラベル別（F1 昇順）----")
    rows = []
    for lab in a.labels():
        lp, lr, lf = _prf(a.tp[lab], a.fp[lab], a.fn[lab])
        rows.append((lab, a.tp[lab], a.fp[lab], a.fn[lab], lp, lr, lf, a.boundary[lab]))
    for lab, tp, fp, fn, lp, lr, lf, bnd in sorted(rows, key=lambda x: (x[6], -(x[1] + x[3]))):
        print(f"  {lab:<10} TP={tp:>5} FP={fp:>5} FN={fn:>5}  "
              f"P={lp:.3f} R={lr:.3f} F1={lf:.3f}  境界ズレ={bnd}")
    if a.type_conf:
        print("---- 型混同（pred → gold）上位 ----")
        for (pl, gl), c in a.type_conf.most_common(8):
            print(f"  {pl:>10} → {gl:<10} : {c}")


def load_sets(extra: Sequence[str]) -> dict[str, tuple[list[dict], bool]]:
    """(docs, use_neutral) の辞書。固定 OOD・外部 2 種に --extra を足す。"""
    sys.path.insert(0, notebooks_dir())
    import eval_ood_dashboard as dash  # noqa: E402

    out: dict[str, tuple[list[dict], bool]] = {"固定OOD": (dash.load_docs(), True)}
    for name, docs in load_external().items():
        out[f"外部 {name}"] = (docs, True)
    gd = drive_dir()
    blind = os.path.join(gd, "blind_eval_v1", "blind_eval_v1.jsonl")
    if os.path.exists(blind):
        out["盲検v1"] = ([json.loads(l) for l in open(blind, encoding="utf-8") if l.strip()],
                        True)
    for item in extra:
        name, path = item.split("=", 1)
        out[name] = ([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()],
                     True)
    return out


def short_tag(tag: str) -> str:
    """表示・出力先に使う短い名前。パスなら outputs_ を外したフォルダ名にする。"""
    if not os.path.isdir(tag):
        return tag
    name = os.path.basename(tag.rstrip("/"))
    if name == "best":
        name = os.path.basename(os.path.dirname(tag.rstrip("/")))
    return name[len("outputs_"):] if name.startswith("outputs_") else name


NOTES = """
## 読み方

- **F1 は span と型がどちらも厳密一致したものだけを TP に数える。** 境界が 1 文字違えば
  TP にならず、FP と FN の両方に数える。マスキング製品としての安全性（PII が実際に
  隠れたか）とは別の指標。
- **安全性は「被覆ゼロ」（`uncovered_gold.csv`）で見る。** どの予測とも 1 文字も重ならなかった
  gold の数で、型は問わない。`false_negatives.csv` は 1 対 1 照合で未マッチになった gold で、
  1 つの予測が隣接する 2 つの gold にまたがる（`京都市伏見区…の坂本町 6-18-8` を 1 スパンで
  予測し、gold は LOCATION と ADDRESS の 2 つ）と片方が未マッチになる。**その gold の文字は
  隠れているので漏れではない。** false_negatives を完全見逃しと読むと、広いスパンを出す
  モデルほど不当に悪く見える。
- **外部 OpenPII の F1 が低いのは半分が注釈規約の差。** gold は複数語の人名を 1 スパンで
  注釈する（`Hixhrete 林 杉山`）が敬称は含めない（`kun` `ちゃん` は gold 外）。
  住所も町名と番地の切り方が本ライブラリの規約と違う（`8-8-17番地三隅町河内` を 1 スパン）。
  そのため境界ズレが大量に出る。完全見逃しの件数で見ると桁が違う。
- **OpenPII は gold 以外の全文字が neutral** なので、neutral 内に完全に収まる予測は
  FP から除いている（`neutral_skipped` 列）。それでも gold と部分的に重なる予測は
  FP に残るため、precision はこのセットでは比較用にしか使えない。
- **FACILITY の gold は OpenPII に 0 件**（ORG も 5 件）。この 2 型の数値はこのセットでは意味を持たない。
- **段階の差**（`model` と `refined`）は Refiner の寄与。token モデルでは大きく、
  文字ヘッドモデルではほぼ無いか逆にわずかに下がる（境界を学習済みのため）。
- `summary.csv` の `f1_<LABEL>` は、その型の gold も予測も 0 件のとき 0.0 で埋まる
  （GSD の ADDRESS、OpenPII の FACILITY）。型別は `per_label_metrics.csv` を見ること。
  そちらには出現した型だけが載る。
"""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+",
                    help="モデルタグ / ディレクトリパス / NAME=PATH（表示名を明示する）")
    ap.add_argument("--out-dir", default=None, help="エラー分析ファイルの出力先")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                    help="追加の評価セット（JSONL: text/entities[/neutral]）")
    ap.add_argument("--stages", default="model,refined",
                    help="model / refined をカンマ区切りで（既定は両方）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--example-limit", type=int, default=EXAMPLE_LIMIT,
                    help="エラー例 CSV の行数上限（0 で無制限）")
    ap.add_argument("--only-extra", action="store_true",
                    help="--extra で渡したセットだけ採点する（固定OOD・外部・盲検v1 を飛ばす）")
    args = ap.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ("model", "refined"):
            raise SystemExit(f"unknown stage: {s} (model / refined)")
    ref = MorphologicalRefiner(
        backend=os.environ.get("PRIVACY_MASKER_REFINER_BACKEND", "janome"))
    sets = load_sets(args.extra)
    if args.only_extra:
        extra_names = {item.split("=", 1)[0] for item in args.extra}
        sets = {k: v for k, v in sets.items() if k in extra_names}
        if not sets:
            raise SystemExit("--only-extra には --extra NAME=PATH が必要です")
    print(f"評価セット: " + "  ".join(f"{k}({len(v[0])})" for k, v in sets.items()))

    summary: list[dict] = []
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, short_tag(name)
        tag = name
        det, kind = build_detector(model_path(path), device=args.device)
        print(f"\n[detector] {tag}: {kind} ({type(det).__name__})", flush=True)
        for set_name, (docs, use_neutral) in sets.items():
            texts = [d["text"] for d in docs]
            raw = det.detect_batch(texts)
            for stage in stages:
                preds = ([list(x) for x in raw] if stage == "model"
                         else [ref.refine(t, list(x)) for t, x in zip(texts, raw)])
                a = analyse(docs, preds, use_neutral=use_neutral)
                print_analysis(f"{tag} / {set_name} / {stage}", a)
                p, r, f1 = a.micro()
                bp, br, bf1 = a.binary()
                row = {"model": tag, "set": set_name, "stage": stage, "docs": a.n_docs,
                       "gold": sum(a.tp.values()) + sum(a.fn.values()),
                       "tp": sum(a.tp.values()), "fp": sum(a.fp.values()),
                       "fn": sum(a.fn.values()),
                       "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6),
                       "macro_f1": round(a.macro_f1(), 6),
                       "label_agnostic_f1": round(bf1, 6),
                       "boundary_errors": sum(a.boundary.values()),
                       "type_confusions": sum(a.type_conf.values()),
                       "uncovered": sum(a.uncovered.values()),
                       "unmatched_gold": len(a.false_negatives),
                       "neutral_skipped": a.neutral_skipped}
                for lab in LABELS:
                    lp, lr, lf = _prf(a.tp[lab], a.fp[lab], a.fn[lab])
                    row[f"f1_{lab}"] = round(lf, 6)
                    row[f"fn_{lab}"] = a.fn[lab]
                    row[f"uncovered_{lab}"] = a.uncovered[lab]
                summary.append(row)
                if args.out_dir:
                    safe = re.sub(r"_+", "_", "".join(
                        c if c.isalnum() or c in "-_." else "_" for c in tag)).strip("_")
                    d = os.path.join(args.out_dir, safe,
                                     set_name.replace(" ", "_").replace("/", "_"), stage)
                    write_files(a, d, args.example_limit)
        del det

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        keys = list(summary[0])
        with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
        with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(f"# OOD エンティティ単位 F1（span+型の厳密一致）"
                    f" {dt.date.today().isoformat()}\n\n")
            f.write("| モデル | セット | 段階 | 文書 | gold | TP | FP | FN | P | R | "
                    "**F1** | Macro F1 | ラベル無視F1 | 境界ズレ | 型混同 | "
                    "**被覆ゼロ** (P/A) | 1対1未マッチ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in summary:
                f.write(f"| {r['model']} | {r['set']} | {r['stage']} | {r['docs']} | "
                        f"{r['gold']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                        f"{r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** | "
                        f"{r['macro_f1']:.4f} | {r['label_agnostic_f1']:.4f} | "
                        f"{r['boundary_errors']} | {r['type_confusions']} | "
                        f"**{r['uncovered']}** ({r['uncovered_PERSON']}/{r['uncovered_ADDRESS']}) | "
                        f"{r['unmatched_gold']} |\n")
            f.write(NOTES)
        print(f"\n[保存] {args.out_dir}/summary.csv, summary.md, "
              f"<モデル>/<セット>/<段階>/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

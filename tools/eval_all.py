"""統合評価: Refiner・モデルを変えたら必ずこれを回す（固定 OOD だけでは Refiner の系統的な欠けを見逃した。§17）。

  1. 総当たり切り詰め検査（tools/stress_truncation.py、モデル非依存）
  2. 固定 OOD 11 ジャンル（eval_ood_dashboard.py と同じ集計）
  3. 分布内（学習検証分割: wiki / 手テンプレ合成 / LLM テンプレ合成）
  4. 外部（GSD-NE test / OpenPII-JA holdout、変換済みの gold）
すべて本番設定（NER + 正規表現 + MorphologicalRefiner）。結果は Markdown 表で標準出力と --out に保存。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/eval_all.py <model_dir または Hub ID> [<model_dir または Hub ID> ...] [--out report.md] [--skip-stress]

モデルはローカルのモデルフォルダか Hugging Face Hub のリポジトリ ID（owner/name）。"表示名=パス" で表に出す名前を指定できる。
データフォルダ（PRIVACY_MASKER_DRIVE_DIR）に学習済みモデルを持っている場合は diff_danger.PATHS の登録名も使える。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402
import diff_danger as dd  # noqa: E402
import eval_in_distribution as indist  # noqa: E402
from privacy_masker.detectors import (  # noqa: E402
    CharHeadNerDetector, HandleDetector, RegexDetector, TransformersNerDetector,
)
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402

GD = drive_dir(require=False)   # import 時は止めない（tests が import する）。存在確認は main() の冒頭で行う
LABELS = ("PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION")
# gold スパン内の空白・区切り記号は PII を担わないので coverage の計算から除く（「銀行 支店」の空白、中黒、"Ltd." の終端ピリオド）
IGNORE_CHARS = frozenset(" \u3000\t\r\n・,.、。")


def gold_span(text, s, e):
    return {i for i in range(s, e) if text[i] not in IGNORE_CHARS} or set(range(s, e))


def model_path(tag: str) -> str:
    """登録名（diff_danger.PATHS / eval_in_distribution.PATHS）→ フォルダパス → Hub ID の順に解決する。"""
    return dd.model_path(tag, registry=indist.PATHS)


def build_detector(path: str, device: str = "cpu"):
    """モデルディレクトリの中身で token 版と文字ヘッド版を出し分ける。

    char_head_config.json があれば文字ヘッド付き。手で指定させると取り違えるので自動にする。
    """
    if os.path.exists(os.path.join(path, "char_head_config.json")):
        return CharHeadNerDetector(model=path, device=device), "char"
    return TransformersNerDetector(model=path, device=device), "token"


def score(docs, preds, ref, rx, use_neutral=True, hd=None):
    """文字coverage / 危険(cov<50%) / 完全見逃し / 1字以上漏れ（ラベル別）/ 過剰マスク字"""
    gold_c = cov_c = fp_c = 0
    tot, danger, miss, anyleak = Counter(), Counter(), Counter(), Counter()
    for d, raw in zip(docs, preds):
        t = d["text"]
        ents = ref.refine(t, raw + rx.detect(t) + (hd.detect(t) if hd else []))
        pc, pc_ner = set(), set()
        for e in ents:
            pc.update(range(e.start, e.end))
            if e.entity_type in LABELS:          # 過剰マスクは NER 5 ラベルだけで数える（電話・メール等の正規表現検出は gold に無い正当なマスク）
                pc_ner.update(range(e.start, e.end))
        neutral = set()
        if use_neutral:
            for s, e_ in d.get("neutral", []):
                neutral.update(range(s, e_))
        gold_all, gold_full = set(), set()
        for s, e_, lab in d["entities"]:
            span = gold_span(t, s, e_); gold_all |= span; gold_full.update(range(s, e_)); tot[lab] += 1
            c = len(span & pc) / len(span)
            if c < 0.5: danger[lab] += 1
            if c == 0: miss[lab] += 1
            if c < 1: anyleak[lab] += 1
        gold_c += len(gold_all); cov_c += len(gold_all & pc); fp_c += len(pc_ner - gold_full - neutral)
    return {"docs": len(docs), "gold": sum(tot.values()), "coverage": cov_c / gold_c if gold_c else 1.0,
            "danger": sum(danger.values()), "miss": sum(miss.values()), "anyleak": sum(anyleak.values()), "fp_chars": fp_c,
            "by_label": {l: (danger[l], miss[l], anyleak[l], tot[l]) for l in LABELS if tot[l]}}


def load_external():
    import ast
    ext = os.environ.get("PRIVACY_MASKER_EXTERNAL_EVAL", os.path.join(GD, "external_eval"))   # <name>_predictions.jsonl の置き場
    out = {}
    for name in ("gsd_ne_test", "openpii_ja_holdout"):
        path = os.path.join(ext, f"{name}_predictions.jsonl")
        if not os.path.exists(path):
            continue
        docs = []
        for line in open(path):
            d = json.loads(line)
            g = d["gold"]; n = d.get("neutral", [])
            docs.append({"text": d["text"], "entities": g if isinstance(g, list) else ast.literal_eval(g),
                         "neutral": n if isinstance(n, list) else ast.literal_eval(n)})
        out[name] = docs
    return out


def main() -> None:
    drive_dir()   # データフォルダが無ければ分かりやすく止める（import ではなく実行時に）
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", metavar="MODEL",
                    help="モデルフォルダのパスか Hugging Face Hub のリポジトリ ID（owner/name）。\"表示名=パス\" で表の名前を指定できる")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                    help="追加の評価セット（JSONL: text/entities[/neutral]）。盲検セットはこれで渡す")
    args = ap.parse_args()
    lines = [f"# 統合評価 {dt.date.today().isoformat()}", ""]

    if not args.skip_stress:
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(os.path.dirname(__file__), "stress_truncation.py")],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
        summary = [l for l in r.stdout.splitlines() if l.startswith(("検査", "PII欠け"))]
        lines += ["## 総当たり切り詰め検査（Refiner、モデル非依存）", "", *summary, ""]
        print("\n".join(summary))

    # バックエンドは既定 auto（fugashi があれば fugashi）。過去の数値と比較するときは
    # PRIVACY_MASKER_REFINER_BACKEND=janome で固定する（auto の切り替わりが差分に混ざるのを防ぐ）
    ref = MorphologicalRefiner(backend=os.environ.get("PRIVACY_MASKER_REFINER_BACKEND", "auto"))
    rx = RegexDetector.default_ja(); hd = HandleDetector()
    sys.path.insert(0, notebooks_dir())
    import eval_ood_dashboard as dash  # noqa: E402
    ood_docs = dash.load_docs()
    splits = {name: indist.valid_split(f"{GD}/{f}") for name, f in indist.SOURCES.items()}
    external = load_external()
    for item in args.extra:
        name, path = item.split("=", 1)
        external[name] = [json.loads(l) for l in open(path) if l.strip()]

    header = "| モデル | セット | 文書 | gold | 文字cov | 危険 | 完全見逃し | 1字以上漏れ | 過剰マスク字 | PERSON 危険/漏れ/総数 | ADDRESS 危険/漏れ/総数 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines += ["## モデル評価（本番設定 = NER + 正規表現 + MorphologicalRefiner）", "", header, sep]
    for spec in args.models:
        # "表示名=パス" で名前を与えられる。パスだけなら outputs_ を外したフォルダ名にする
        # （フルパスを表に出すと Markdown が読めなくなる）
        tag, _, path = spec.partition("=")
        if not path:
            path = tag
            if os.path.isdir(tag):
                base = os.path.basename(tag.rstrip("/"))
                if base == "best":
                    base = os.path.basename(os.path.dirname(tag.rstrip("/")))
                tag = base[len("outputs_"):] if base.startswith("outputs_") else base
        det, kind = build_detector(model_path(path), device="cpu")
        print(f"[detector] {tag}: {kind} ({type(det).__name__})", flush=True)
        sets = [("固定OOD 11ジャンル", ood_docs, True)]
        sets += [(f"分布内 {n}", d, False) for n, d in splits.items()]
        sets += [(f"外部 {n}", d, True) for n, d in external.items()]
        for name, docs, use_neutral in sets:
            preds = det.detect_batch([d["text"] for d in docs])
            r = score(docs, preds, ref, rx, use_neutral, hd)
            p = r["by_label"].get("PERSON", (0, 0, 0, 0)); a = r["by_label"].get("ADDRESS", (0, 0, 0, 0))
            row = (f"| {tag} | {name} | {r['docs']} | {r['gold']} | {r['coverage']*100:.1f}% | {r['danger']} | {r['miss']} | "
                   f"{r['anyleak']} | {r['fp_chars']} | {p[0]}/{p[2]}/{p[3]} | {a[0]}/{a[2]}/{a[3]} |")
            lines.append(row); print(row, flush=True)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("saved:", args.out)


if __name__ == "__main__":
    main()

"""盲検セットの詳細レポート: 出典グループ別・ジャンル別の coverage と、漏れた gold スパン／過剰マスクの一覧（分類用）。

失敗例を Refiner 規則や学習データに直接使わないこと（docs/blind_eval_protocol.md §4）。分類して対策の種類を決めるための読み物。

usage:
  PYTHONPATH=. .venv/bin/python tools/blind_set_report.py <gold.jsonl> <model_dir または Hub ID> [...] [--out details.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_all as ea  # noqa: E402
from privacy_masker.detectors import HandleDetector, RegexDetector, TransformersNerDetector  # noqa: E402
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402

NER = set(ea.LABELS)


def group_of(doc: dict) -> str:
    g = doc.get("genre", "")
    if g.startswith("web_"):
        return g
    if g.startswith("negative"):
        return "authored_negative"
    return "authored"


def ctx(text: str, s: int, e: int, w: int = 14) -> str:
    return (text[max(0, s - w):s] + "｟" + text[s:e] + "｠" + text[e:e + w]).replace("\n", "⏎").replace("\r", "")


def analyze(docs, preds, ref, rx, hd):
    per_group = defaultdict(lambda: Counter())
    leaks, fps = [], []
    for d, raw in zip(docs, preds):
        t = d["text"]
        ents = ref.refine(t, raw + rx.detect(t) + hd.detect(t))
        pc = set(); pc_ner = set()
        for e in ents:
            pc.update(range(e.start, e.end))
            if e.entity_type in NER:
                pc_ner.update(range(e.start, e.end))
        neutral = set()
        for s, e_ in d.get("neutral", []):
            neutral.update(range(s, e_))
        gold_all, gold_full = set(), set()
        g = per_group[group_of(d)]; g["docs"] += 1
        for s, e_, lab in d["entities"]:
            span = ea.gold_span(t, s, e_); gold_all |= span; gold_full.update(range(s, e_))
            c = len(span & pc) / len(span)
            g["gold"] += 1; g["gold_chars"] += len(span); g["cov_chars"] += len(span & pc)
            if c < 1:
                g["anyleak"] += 1
                if c < 0.5: g["danger"] += 1
                if c == 0: g["miss"] += 1
                missing = "".join(t[i] for i in sorted(span - pc))
                leaks.append((d["id"], d.get("genre", ""), lab, f"{c*100:.0f}%", t[s:e_], missing, ctx(t, s, e_)))
        fp = pc_ner - gold_full - neutral
        g["fp_chars"] += len(fp)
        # 過剰マスクの表層（連続区間）
        for e in ents:
            if e.entity_type in NER:
                span = set(range(e.start, e.end))
                if span and len(span - gold_full - neutral) / len(span) > 0.5:
                    fps.append((d["id"], d.get("genre", ""), e.entity_type, t[e.start:e.end], ctx(t, e.start, e.end)))
    return per_group, leaks, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold"); ap.add_argument("models", nargs="+"); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    docs = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    ref = MorphologicalRefiner(); rx = RegexDetector.default_ja(); hd = HandleDetector()
    out = [f"# 盲検セット詳細 {os.path.basename(args.gold)}（{len(docs)} 文書）", ""]
    for tag in args.models:
        det = TransformersNerDetector(model=ea.model_path(tag), device="cpu")
        preds = det.detect_batch([d["text"] for d in docs])
        per_group, leaks, fps = analyze(docs, preds, ref, rx, hd)
        out += [f"## {tag}", "", "| グループ | 文書 | gold | 文字cov | 危険 | 完全見逃し | 1字以上漏れ | 過剰マスク字 |", "|---|---|---|---|---|---|---|---|"]
        for grp in sorted(per_group):
            g = per_group[grp]
            cov = g["cov_chars"] / g["gold_chars"] if g["gold_chars"] else 1.0
            out.append(f"| {grp} | {g['docs']} | {g['gold']} | {cov*100:.1f}% | {g['danger']} | {g['miss']} | {g['anyleak']} | {g['fp_chars']} |")
        by_lab = Counter(l[2] for l in leaks); by_lab_d = Counter(l[2] for l in leaks if float(l[3].rstrip('%')) < 50)
        out += ["", f"漏れ {len(leaks)} 件（ラベル別 {dict(by_lab)}、うち危険 {dict(by_lab_d)}）", "",
                "| 文書 | ジャンル | ラベル | cov | gold 表層 | 漏れた文字 | 文脈 |", "|---|---|---|---|---|---|---|"]
        for row in sorted(leaks, key=lambda r: (float(r[3].rstrip('%')), r[0])):
            out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
        out += ["", f"過剰マスク（gold・NEUTRAL 外が半分超のスパン） {len(fps)} 件", "", "| 文書 | ジャンル | ラベル | 表層 | 文脈 |", "|---|---|---|---|---|"]
        for row in fps:
            out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
        out.append("")
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text); print("saved:", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()

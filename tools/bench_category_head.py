"""分類ヘッドの効果測定（規則のみ vs ヘッド有り）。評価セットの表層は出力しない（集計のみ）。

usage: PYTHONPATH=. .venv/bin/python tools/bench_category_head.py [--head-dir DIR] [--json OUT]
  --head-dir 省略時は本番モデルフォルダから自動読み込み（category_head.json/.safetensors、linear / mlp 共通）。
  指定時は DIR の category_head.{json,safetensors} を、本番 encoder の SHA 検査付きで読む
  （`HeadCategorizer.load(model_dir, head_dir=...)` が行う。不採用ヘッドの記録用）。

データフォルダ・本番モデル・評価データへのアクセスは `main()` の中でだけ行う（`--help` はデータフォルダが無くても通る）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.getcwd())

from privacy_masker import MorphologicalRefiner, Policy, PrivacyEngine
from privacy_masker.category_head import HeadCategorizer
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir  # noqa: E402
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector
from privacy_masker.operators import Surrogate
from privacy_masker.operators import semantic as sm
from privacy_masker.operators.semantic import CATEGORIES, OTHER, RuleCategorizer

KEY = b"bench-surrogate-key-2026-09-27"
ALL_TYPES = ("PERSON", "ORG", "FACILITY", "LOCATION", "ADDRESS")

# 架空文の置換例（実文書の表層は出さない。従来どおり）
SAMPLES = [
    "来月からアオバに転職し、新しいスマートフォンの開発を担当する。",
    "ミナトヤの店頭で野菜と果物を買ってから帰宅した。",
    "ヒカリノに入院していた祖母が、先週無事に退院した。",
    "サクラノの口座から授業料を振り込んだ。",
    "息子はコハクダイの三年生で、来年は受験を控えている。",
    "週末はシオカゼに泊まって、露天風呂を楽しんだ。",
    "ツキミノに着いたら、改札を出て北口で待っていてください。",
    "ホシゾラから届いた荷物を倉庫で仕分けした。",
    "若葉原の田んぼ沿いを歩いて、隣の集落まで行った。",
    "カザミネの選手たちが優勝パレードに参加した。",
]


class RecHead:
    """HeadCategorizer の呼び出しを記録する薄い代理（min_margin / categories はそのまま見せる）。"""

    def __init__(self, h):
        self.h = h
        self.min_margin = h.min_margin
        self.categories = h.categories
        self.last = None

    def __call__(self, kind, vec):
        r = self.h(kind, vec)
        self.last = r
        return r


def dets(ner):
    return [ner, RegexDetector.default_ja(), HandleDetector()]


def make_engine(ner, head, refiner):
    return PrivacyEngine(dets(ner), policy=Policy(default=Surrogate(kind="auto", key=KEY, category_head=head)), refiner=refiner)


def run(engine, docs, bs=64):
    out = []
    for i in range(0, len(docs), bs):
        out += engine.transform_batch(docs[i:i + bs])
    return out


def fb_counts(results):
    fb = collections.Counter()
    n = collections.Counter()
    for r in results:
        for e in r.entities:
            if e.entity_type in ALL_TYPES:
                n[e.entity_type] += 1
                if e.replacement.startswith("<") and e.replacement.endswith(">"):
                    fb[e.entity_type] += 1
    return fb, n


def pct(a, b):
    return f"{a}/{b} ({a / b * 100:.1f}%)" if b else "-"


def summarize(log_rows, head) -> dict:
    """解決カテゴリの集計（純粋関数、表層は含まない）。

    `log_rows` は `(mode, kind, rule_label, has_vector, head_result, final_label)` のタプルの列
    （`mode` は "off"/"on"、`head_result` は `head(kind, vector)` の戻り値 `(category, margin)` か
    None）。kind ごとに OTHER 率・規則が OTHER にしたスパンのうち head が決めた割合・margin で
    棄却した数・head の決定分布・最終分布を集計する。
    """
    stats = {}
    for kind in ("org", "facility", "location"):
        rows = {m: [r for r in log_rows if r[0] == m and r[1] == kind] for m in ("off", "on")}
        n = len(rows["off"])
        other_off = sum(1 for r in rows["off"] if r[5] == OTHER[kind])
        other_on = sum(1 for r in rows["on"] if r[5] == OTHER[kind])
        rule_other = [r for r in rows["on"] if r[2] == OTHER[kind]]
        no_vec = sum(1 for r in rule_other if not r[3])
        called = [r for r in rule_other if r[4] is not None]
        mm = head.min_margin.get(kind) if kind in head.categories else None
        decided = [r for r in called if mm is not None and r[4][1] >= mm and r[4][0] in CATEGORIES[kind]]
        rejected = [r for r in called if not (mm is not None and r[4][1] >= mm)]
        decided_non_other = [r for r in decided if r[4][0] != OTHER[kind]]
        dist = collections.Counter(r[4][0] for r in decided)
        dist_all = collections.Counter(r[5] for r in rows["on"])
        dist_off = collections.Counter(r[5] for r in rows["off"])
        stats[kind] = {
            "n_resolve": n, "n_resolve_on": len(rows["on"]), "other_off": other_off, "other_on": other_on,
            "rule_other": len(rule_other), "no_vector": no_vec, "head_called": len(called),
            "head_in_file": kind in head.categories, "min_margin": mm,
            "decided": len(decided), "decided_non_other": len(decided_non_other), "rejected_margin": len(rejected),
            "head_dist": dict(dist.most_common()), "final_dist_on": dict(dist_all.most_common()),
            "final_dist_off": dict(dist_off.most_common()),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-dir", default=None,
                    help="候補ヘッドのフォルダ（category_head.json/.safetensors）。省略時はモデルフォルダの自動読み込み")
    ap.add_argument("--model", default=None, help="評価する encoder（モデルフォルダか Hub ID）。省略時は Hub の本番モデル")
    ap.add_argument("--json", default=None, help="集計結果を書く JSON のパス")
    args = ap.parse_args(argv)

    # データフォルダ・本番モデル・評価データの読み込みはここから（--help は上の parse_args で終わるので、ここへは来ない）
    from tools._paths import drive_dir, notebooks_dir

    sys.path.insert(0, notebooks_dir())
    import eval_ood_dashboard as dash

    MODEL = resolve_model_dir(args.model or DEFAULT_MODEL_ID)
    EXT = os.environ.get("PRIVACY_MASKER_EXTERNAL_EVAL", os.path.join(drive_dir(), "external_eval"))

    texts = [d["text"] for d in dash.load_docs()]
    n_ood = len(texts)
    for name in ("gsd_ne_test", "openpii_ja_holdout"):
        p = os.path.join(EXT, f"{name}_predictions.jsonl")
        texts += [json.loads(line)["text"] for line in open(p) if line.strip()]
    N = len(texts)
    print(f"docs: {N} (固定OOD {n_ood}, GSD/OpenPII {N - n_ood})")

    ner = CharHeadNerDetector(model=MODEL, device="cpu", category_head=False)
    head = HeadCategorizer.load(MODEL, head_dir=args.head_dir)
    ner.category_head = head          # vector_features がヘッドの features に従う
    head_src = args.head_dir or "auto (model dir)"
    print(f"head: {head_src} kinds={sorted(head.categories)} min_margin={head.min_margin}")

    refiner = MorphologicalRefiner()

    # ---------- 1) 時間: apply の累積時間（Surrogate.apply をクラスで包む）+ パイプライン全体 ----------
    apply_t_acc = [0.0]
    orig_apply = Surrogate.apply

    def timed_apply(self, value, entity, ctx):
        t0 = time.perf_counter()
        try:
            return orig_apply(self, value, entity, ctx)
        finally:
            apply_t_acc[0] += time.perf_counter() - t0

    Surrogate.apply = timed_apply

    eng_off, eng_on = make_engine(ner, None, refiner), make_engine(ner, head, refiner)
    run(eng_off, texts[:64]); run(eng_on, texts[:64])          # warm-up
    timing = {"off": [], "on": []}
    apply_t = {"off": [], "on": []}
    res = {}
    for rep in range(2):
        for mode, eng in (("off", eng_off), ("on", eng_on)) if rep == 0 else (("on", eng_on), ("off", eng_off)):
            apply_t_acc[0] = 0.0
            t0 = time.perf_counter(); res[mode] = run(eng, texts); timing[mode].append(time.perf_counter() - t0)
            apply_t[mode].append(apply_t_acc[0])
    Surrogate.apply = orig_apply
    pipe_ms = {m: min(v) * 1000 / N for m, v in timing.items()}
    apply_ms = {m: min(v) * 1000 / N for m, v in apply_t.items()}
    print("\n== 時間（CPU, 2 回の最小, ms/doc）==")
    for m in ("off", "on"):
        print(f"  head {m:3s}: pipeline {pipe_ms[m]:.2f}  apply(Surrogate) {apply_ms[m]:.3f}")
    print(f"  差（on-off）: pipeline {pipe_ms['on'] - pipe_ms['off']:+.2f}  apply {apply_ms['on'] - apply_ms['off']:+.3f}")

    # ---------- 2) 解決カテゴリの記録（CategoryResolver.resolve を包む）----------
    log_rows: list[tuple] = []
    orig_resolve = sm.CategoryResolver.resolve
    mode_box = ["off"]

    def rec_resolve(self, kind, norm, vector):
        rule = RuleCategorizer.categorize(kind, norm)
        if isinstance(self.head, RecHead):
            self.head.last = None
        out = orig_resolve(self, kind, norm, vector)
        ho = self.head.last if isinstance(self.head, RecHead) else None
        log_rows.append((mode_box[0], kind, rule, vector is not None, ho, out))
        return out

    sm.CategoryResolver.resolve = rec_resolve
    rec_head = RecHead(head)
    eng_off_r, eng_on_r = make_engine(ner, None, refiner), make_engine(ner, rec_head, refiner)
    res_r = {}
    for mode, eng in (("off", eng_off_r), ("on", eng_on_r)):
        mode_box[0] = mode; res_r[mode] = run(eng, texts)
    sm.CategoryResolver.resolve = orig_resolve

    # 置換結果が計時ランと同一（決定的）であることの確認
    same = all(a.text == b.text for a, b in zip(res["on"], res_r["on"])) and \
        all(a.text == b.text for a, b in zip(res["off"], res_r["off"]))
    print(f"\n計時ランと記録ランの出力一致: {same}")

    stats = summarize(log_rows, head)
    print("\n== 解決カテゴリ（resolve 呼び出し = 出現単位）==")
    for kind, s in stats.items():
        print(f"[{kind}] n={s['n_resolve']} (on {s['n_resolve_on']})  head_in_file={s['head_in_file']} min_margin={s['min_margin']}")
        print(f"  OTHER: off {pct(s['other_off'], s['n_resolve'])} → on {pct(s['other_on'], s['n_resolve_on'])}")
        print(f"  規則で OTHER {s['rule_other']}（うち vector 無し {s['no_vector']}、head 呼び出し {s['head_called']}）")
        print(f"  head 決定 {pct(s['decided'], s['rule_other'])}（非OTHER {s['decided_non_other']}）、margin で棄却 {pct(s['rejected_margin'], s['rule_other'])}")
        print(f"  head 決定の分布: {s['head_dist']}")
        print(f"  最終分布 off: {s['final_dist_off']}")
        print(f"  最終分布 on : {s['final_dist_on']}")

    # ---------- 3) トークン退避（エンティティ単位）----------
    fb_off, n_off = fb_counts(res["off"]); fb_on, n_on = fb_counts(res["on"])
    print("\n== トークン退避（エンティティ単位）==")
    for t in ALL_TYPES:
        print(f"  {t:9s} n={n_off[t]:5d}  off {fb_off[t]:4d}  on {fb_on[t]:4d}  Δ {fb_on[t] - fb_off[t]:+d}")
    changed = sum(1 for a, b in zip(res["off"], res["on"]) if a.text != b.text)
    print(f"  出力が変わった文書: {changed}/{N}")

    # ---------- 4) 架空文の置換例 ----------
    examples = []
    print("\n== 置換例（架空文, hmac, head 無し / 有り）==")
    log_rows.clear(); sm.CategoryResolver.resolve = rec_resolve
    for s in SAMPLES:
        mode_box[0] = "off"; r_off = eng_off_r.transform(s)
        mode_box[0] = "on"; log_rows.clear(); r_on = eng_on_r.transform(s)
        heads_seen = [(r[1], r[2], r[4], r[5]) for r in log_rows if r[0] == "on"]
        pairs = []
        for eo, en in zip(r_off.entities, r_on.entities):
            pairs.append({"type": eo.entity_type, "orig": s[eo.original_start:eo.original_end], "off": eo.replacement, "on": en.replacement})
        examples.append({"text": s, "off": r_off.text, "on": r_on.text, "pairs": pairs,
                         "resolve_on": [{"kind": k, "rule": ru, "head": list(ho) if ho else None, "final": fi} for k, ru, ho, fi in heads_seen]})
        print("  " + s)
        print("    off: " + r_off.text)
        print("    on : " + r_on.text)
        for p in pairs:
            print(f"      {p['type']}: {p['orig']} → {p['off']} / {p['on']}")
        for k, ru, ho, fi in heads_seen:
            print(f"      resolve[{k}] rule={ru} head={(ho[0], round(ho[1], 3)) if ho else None} final={fi}")
    sm.CategoryResolver.resolve = orig_resolve

    if args.json:
        json.dump({"docs": N, "model": MODEL, "head": head_src, "min_margin": head.min_margin, "pipe_ms": pipe_ms, "apply_ms": apply_ms,
                   "stats": stats, "fallback_off": dict(fb_off), "fallback_on": dict(fb_on), "entities": dict(n_off),
                   "changed_docs": changed, "deterministic_match": same, "examples": examples},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""補助検証（表層なし）: 実文書 1,870 件で規則がカテゴリを決めた ORG/FACILITY/LOCATION スパンの種別語を
学習データと同じ mask_type_word で削り、ヘッドの予測を規則ラベルと比べる（実文書での (ii) の近似）。
採用判定には使わない（判定は valid の (i)(ii) で事前に決めた条件）。候補ヘッド（全 kind、linear / mlp / features を問わない）で測る。

usage: PYTHONPATH=. .venv/bin/python tools/bench_category_head_realcheck.py --head-dir DIR [--json OUT] [--margins ...]

データフォルダ・本番モデル・評価データへのアクセスは `main()` の中でだけ行う（`--help` はデータフォルダが無くても通る）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))

from privacy_masker import MorphologicalRefiner, Policy, PrivacyEngine
from privacy_masker.category_head import HeadCategorizer
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir  # noqa: E402
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector
from privacy_masker.normalize import default_normalize
from privacy_masker.operators import Tokenize
from privacy_masker.operators.semantic import OTHER, RuleCategorizer

T2K = {"ORG": "org", "FACILITY": "facility", "LOCATION": "location"}
DEFAULT_MARGINS = (0.15, 0.30, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90)


def precision_table(items, probs, cats, margins=DEFAULT_MARGINS) -> dict[float, dict]:
    """items = [(kind, 規則ラベル, 種別語を削ったか)]、probs = 対応する確率。margin ≥ m の予測だけを採用した precision。"""
    out = {}
    for m in margins:
        kept = []
        for (kind, label, _masked), p in zip(items, probs):
            order = sorted(range(len(p)), key=p.__getitem__, reverse=True)
            top2 = p[order[1]] if len(order) > 1 else 0.0
            if p[order[0]] - top2 >= m - 1e-9:
                kept.append(cats[order[0]] == label)
        out[m] = {"n": len(kept), "correct": sum(kept), "precision": (sum(kept) / len(kept)) if kept else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-dir", required=True, help="候補ヘッドのフォルダ（category_head.json/.safetensors）")
    ap.add_argument("--model", default=None, help="評価する encoder（モデルフォルダか Hub ID）。省略時は Hub の本番モデル")
    ap.add_argument("--json", default=None, help="表を書く JSON のパス")
    ap.add_argument("--margins", default=",".join(str(m) for m in DEFAULT_MARGINS),
                    help="precision を測る margin のカンマ区切りリスト（既定: %(default)s）")
    args = ap.parse_args(argv)
    margins = tuple(float(x) for x in args.margins.split(","))

    # データフォルダ・本番モデル・評価データの読み込みはここから（--help は上の parse_args で終わるので、ここへは来ない）
    from tools._paths import drive_dir, notebooks_dir

    sys.path.insert(0, notebooks_dir())
    import eval_ood_dashboard as dash
    from category_head_data import mask_type_word
    from privacy_masker.char_head import encode_span_vectors, vector_to_floats

    D = drive_dir()
    MODEL = resolve_model_dir(args.model or DEFAULT_MODEL_ID)
    EXT = os.environ.get("PRIVACY_MASKER_EXTERNAL_EVAL", os.path.join(D, "external_eval"))
    texts = [d["text"] for d in dash.load_docs()]
    for name in ("gsd_ne_test", "openpii_ja_holdout"):
        p = os.path.join(EXT, f"{name}_predictions.jsonl")
        texts += [json.loads(line)["text"] for line in open(p) if line.strip()]

    head = HeadCategorizer.load(MODEL, head_dir=args.head_dir)
    print(f"head: {args.head_dir} kinds={sorted(head.categories)} head_type={head.head_type} "
          f"features={head.features} min_margin={head.min_margin}")

    ner = CharHeadNerDetector(model=MODEL, device="cpu", category_head=False)
    eng = PrivacyEngine([ner, RegexDetector.default_ja(), HandleDetector()],
                        policy=Policy(default=Tokenize(strategy="sequential")), refiner=MorphologicalRefiner())
    res = []
    for i in range(0, len(texts), 64):
        res += eng.transform_batch(texts[i:i + 64])

    # items: (kind, 規則ラベル, 種別語を削ったか, テキスト, start, end) -- テキスト/位置はベクトル計算にしか使わず、表層は出力しない
    items: list[tuple[str, str, bool, str, int, int]] = []
    for t, r in zip(texts, res):
        for e in r.entities:
            kind = T2K.get(e.entity_type)
            if kind is None or kind not in head.categories:
                continue
            s, en = e.original_start, e.original_end
            lab = RuleCategorizer.categorize(kind, default_normalize(t[s:en]))
            if lab == OTHER[kind]:
                continue
            items.append((kind, lab, False, t, s, en))
            masked = mask_type_word(t, s, en, kind, lab)
            if masked is not None:
                items.append((kind, lab, True, masked[0], masked[1], masked[2]))

    vecs = encode_span_vectors([it[3] for it in items], [[(it[4], it[5])] for it in items], ner.model, ner.tokenizer,
                                batch_size=32, features=head.features)

    def head_probs(kind: str, vec: bytes | None):
        if vec is None:
            return None
        return head._probs(head._heads[kind], vector_to_floats(vec))

    scored = [(it, head_probs(it[0], v[0])) for it, v in zip(items, vecs)]

    out = {"model": MODEL}
    for kind in ("org", "facility", "location"):
        if kind not in head.categories:
            continue
        cats = head.categories[kind]
        for masked in (False, True):
            sel = [(it, p) for it, p in scored if it[0] == kind and it[2] == masked and p is not None]
            if not sel:
                continue
            sel_items = [(it[0], it[1], it[2]) for it, _ in sel]
            sel_probs = [p for _, p in sel]
            correct = sum(1 for it, p in sel if cats[max(range(len(p)), key=p.__getitem__)] == it[1])
            table = precision_table(sel_items, sel_probs, cats, margins=margins)
            mm = head.min_margin.get(kind)
            at_min_margin = precision_table(sel_items, sel_probs, cats, margins=(mm,))[mm] if mm is not None else None
            row = {
                "n": len(sel),
                "acc": correct / len(sel),
                "labels": dict(collections.Counter(it[1] for it, _ in sel).most_common()),
                "margins": table,
                "min_margin": mm,
                "at_min_margin": at_min_margin,
            }
            out[f"{kind}/{'masked' if masked else 'unmasked'}"] = row
            print(kind, "masked" if masked else "unmasked", json.dumps(row, ensure_ascii=False))

    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

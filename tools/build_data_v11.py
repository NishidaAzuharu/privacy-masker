"""学習データ v11 の構築。Wikipedia 実テキスト（v10 と同じ所属）+ GSD 学習・開発分割 + Qwen テンプレートの差し込み文書 + LLM 負例 + 長文連結。
`train_v11_plus_old.jsonl` は v10 の合成層（テンプレート・フレーム生成・連結）を足した切り分け用。

usage:
  PYTHONPATH=. .venv/bin/python tools/build_data_v11.py [--out-dir <データフォルダ>/data_v11] [--n-negatives 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import Counter
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from build_data_v6 import stats as record_stats  # noqa: E402
from build_data_v8 import load_negatives  # noqa: E402
from build_data_v9 import load_hard_negative_texts, select_negatives, split_rows  # noqa: E402
from negatives_v8 import file_sha256, read_jsonl  # noqa: E402
from split_by_template import template_signature  # noqa: E402
from synth_v11_fill import DOCS_FILE, FILLED_DIR_NAME  # noqa: E402
from synth_v11_plan import PLAN_DIR_NAME  # noqa: E402
from synth_v11_values import build_exclusions, is_excluded  # noqa: E402

SEED = 42
VALID_RATIO = 0.10
N_NEGATIVES = 2000
CONCAT_RATIO = 0.10
LONG_MIN_CHARS = 400
OLD_SYNTH_EXCLUDE = frozenset({"out_ner", "qwen38_negatives_v8"})
OUT_DIR_NAME = "data_v11"


def split_by_template_id(docs: Sequence[dict], rng: random.Random, ratio: float) -> tuple[list[dict], list[dict]]:
    ids = sorted({d["template_id"] for d in docs})
    rng.shuffle(ids)
    k = max(1, int(len(ids) * ratio)) if ids else 0
    valid_ids = set(ids[:k])
    return [d for d in docs if d["template_id"] not in valid_ids], [d for d in docs if d["template_id"] in valid_ids]


def split_random(records: Sequence[dict], rng: random.Random, ratio: float) -> tuple[list[dict], list[dict]]:
    xs = list(records)
    rng.shuffle(xs)
    k = int(len(xs) * ratio)
    return xs[k:], xs[:k]


def make_concat_docs(records: Sequence[dict], rng: random.Random, target: int) -> list[dict]:
    """別テンプレート／別文の文書を 2〜4 本つないで 400 字超の文書を作る。neutral も位置をずらして持ち越す。"""
    pool = [r for r in records if len(r["text"]) <= LONG_MIN_CHARS and r["entities"]]
    rng.shuffle(pool)
    out: list[dict] = []
    i = 0
    while len(out) < target and i + 2 <= len(pool):
        k = rng.choice((2, 3, 4))
        parts = pool[i:i + k]
        i += k
        if len({template_signature(p, loose=True) for p in parts}) < len(parts):
            continue
        text, ents, neutral, spans = "", [], [], []
        for p in parts:
            if text:
                text += "\n\n"
            off = len(text)
            text += p["text"]
            ents += [[s + off, e + off, l] for s, e, l in p["entities"]]
            neutral += [[s + off, e + off] for s, e in p.get("neutral", [])]
            spans.append([off, len(text), p["source"]])           # 部分ごとの出所（漏洩検査が合成部分だけを見るため）
        if len(text) <= LONG_MIN_CHARS:
            continue
        d = {"text": text, "entities": sorted(ents, key=lambda x: (x[0], x[1])), "source": "concat", "variant": f"concat:{k}", "parts": spans}
        if neutral:
            d["neutral"] = sorted(neutral)
        out.append(d)
    return out


def leak_checks(records: Sequence[dict], excl_person: set[str], eval_texts: set[str]) -> dict:
    overlap = [r for r in records if r["text"].strip() in eval_texts]
    person_hits = []
    for r in records:
        # 評価セットの PERSON 表層の混入を見るのは合成層だけ。実テキスト（Wikipedia・GSD）が評価セットと姓を共有するのは漏洩ではない
        if r["source"] == "qwen38_synth_v11":
            spans = r["entities"]
        elif r["source"] == "concat":
            synth = [(a, b) for a, b, src in r.get("parts", []) if src == "qwen38_synth_v11"]
            spans = [x for x in r["entities"] if any(a <= x[0] and x[1] <= b for a, b in synth)]
        else:
            continue
        for s, e, l in spans:
            if l == "PERSON" and is_excluded(r["text"][s:e], excl_person):
                person_hits.append(r["text"][s:e])
    return {"eval_text_overlap": len(overlap), "eval_text_overlap_ids": [r.get("id") or r.get("template_id") for r in overlap][:20],
            "eval_person_surface": len(person_hits), "eval_person_surface_examples": sorted(set(person_hits))[:20]}


def _drop_overlap(records: Sequence[dict], eval_texts: set[str]) -> list[dict]:
    return [r for r in records if r["text"].strip() not in eval_texts]


def build(*, v10_train: Sequence[dict], v10_valid: Sequence[dict], gsd: Sequence[dict], synth_docs: Sequence[dict],
          negatives: Sequence[dict], hard_texts: set[str], excl_person: set[str], eval_texts: set[str],
          seed: int = SEED, n_negatives: int = N_NEGATIVES, concat_ratio: float = CONCAT_RATIO, valid_ratio: float = VALID_RATIO) -> dict:
    rng = random.Random(seed)
    wiki_tr = [r for r in v10_train if r["source"] == "out_ner"]
    wiki_va = [r for r in v10_valid if r["source"] == "out_ner"]
    old_synth = [r for r in v10_train if r["source"] not in OLD_SYNTH_EXCLUDE]
    gsd_tr, gsd_va = split_random(_drop_overlap(gsd, eval_texts), rng, valid_ratio)
    syn_tr, syn_va = split_by_template_id(_drop_overlap(synth_docs, eval_texts), rng, valid_ratio)
    neg = select_negatives(list(negatives), min(n_negatives, len(negatives)), rng, hard_texts)
    neg_tr, neg_va = split_rows(neg, rng, valid_ratio)
    concat_tr = make_concat_docs(syn_tr + gsd_tr, rng, int((len(syn_tr) + len(gsd_tr)) * concat_ratio))
    concat_va = make_concat_docs(syn_va + gsd_va, rng, int((len(syn_va) + len(gsd_va)) * concat_ratio))
    train = wiki_tr + gsd_tr + syn_tr + concat_tr + neg_tr
    valid = wiki_va + gsd_va + syn_va + concat_va + neg_va
    rng.shuffle(train); rng.shuffle(valid)
    plus_old = list(train) + list(old_synth)
    rng.shuffle(plus_old)
    leak = leak_checks(train + valid, excl_person, eval_texts)
    leak["eval_text_overlap_dropped"] = (len(gsd) - len(gsd_tr) - len(gsd_va)) + (len(synth_docs) - len(syn_tr) - len(syn_va))
    info = {"seed": seed, "valid_ratio": valid_ratio, "concat_ratio": concat_ratio,
            "layers": {"out_ner": {"train": len(wiki_tr), "valid": len(wiki_va)}, "gsd_train": {"train": len(gsd_tr), "valid": len(gsd_va)},
                       "qwen38_synth_v11": {"train": len(syn_tr), "valid": len(syn_va), "templates_train": len({d["template_id"] for d in syn_tr}),
                                            "templates_valid": len({d["template_id"] for d in syn_va})},
                       "concat": {"train": len(concat_tr), "valid": len(concat_va)}},
            "negatives": {"train": len(neg_tr), "valid": len(neg_va), "hard": sum(1 for r in neg if r["text"] in hard_texts),
                          "by_genre": dict(Counter((r.get("variant") or "").split("/")[0] for r in neg))},
            "old_synth_added_in_plus_old": len(old_synth), "old_synth_sources": dict(Counter(r["source"] for r in old_synth)),
            "leak": leak, "neutral_docs_train": sum(1 for r in train if r.get("neutral"))}
    return {"train": train, "train_plus_old": plus_old, "valid": valid, "info": info}


def write_records(path: str, records: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = {"text": r["text"], "entities": r["entities"], "source": r["source"]}
            for k in ("variant", "template_id", "id"):
                if r.get(k) is not None:
                    row[k] = r[k]
            if r.get("neutral"):
                row["neutral"] = r["neutral"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_table(train: Sequence[dict], valid: Sequence[dict]) -> str:
    tr, va = Counter(r["source"] for r in train), Counter(r["source"] for r in valid)
    ntr = Counter(r["source"] for r in train if not r["entities"]); nva = Counter(r["source"] for r in valid if not r["entities"])
    lines = ["| 出所 | train | valid | うち負例 train | うち負例 valid |", "|---|---:|---:|---:|---:|"]
    for s in sorted(set(tr) | set(va)):
        lines.append(f"| {s} | {tr[s]} | {va[s]} | {ntr[s]} | {nva[s]} |")
    return "\n".join(lines)


def _label_table(train: Sequence[dict], valid: Sequence[dict]) -> str:
    tr = Counter(l for r in train for _, _, l in r["entities"]); va = Counter(l for r in valid for _, _, l in r["entities"])
    lines = ["| ラベル | train | valid |", "|---|---:|---:|"]
    for lab in sorted(set(tr) | set(va)):
        lines.append(f"| {lab} | {tr[lab]} | {va[lab]} |")
    return "\n".join(lines)


def data_card(info: dict, train: Sequence[dict], valid: Sequence[dict], plus_old: Sequence[dict]) -> str:
    st_tr, st_va = record_stats(train), record_stats(valid)
    real = sum(1 for r in train if r["source"] in ("out_ner", "gsd_train"))
    lines = ["# 学習データ v11", "", f"作成 {dt.datetime.now():%Y-%m-%d} / seed {info['seed']} / `tools/build_data_v11.py`", "",
             "| 指標 | train | valid |", "|---|---:|---:|", f"| 文書数 | {len(train)} | {len(valid)} |",
             f"| 負例 | {sum(1 for r in train if not r['entities'])} | {sum(1 for r in valid if not r['entities'])} |",
             f"| 実テキスト（out_ner + gsd_train） | {real} ({real / max(len(train), 1):.1%}) | — |",
             f"| neutral を持つ文書 | {info['neutral_docs_train']} | — |",
             f"| 400 字超の文書 | {st_tr['counts'].get('long', 0)} | {st_va['counts'].get('long', 0)} |",
             f"| エンティティ数 | {sum(st_tr['entities'].values())} | {sum(st_va['entities'].values())} |",
             f"| テンプレート署名の種類 | {st_tr['templates']} | {st_va['templates']} |",
             "", "## 出所", "", _source_table(train, valid), "", "## ラベル", "", _label_table(train, valid), "",
             "## 層", "", "```json", json.dumps(info["layers"], ensure_ascii=False, indent=2), "```", "",
             f"負例: train {info['negatives']['train']} / valid {info['negatives']['valid']}（ハード負例 {info['negatives']['hard']}、ジャンル {info['negatives']['by_genre']}）", "",
             f"`train_v11_plus_old.jsonl` = train + v10 の合成層 {info['old_synth_added_in_plus_old']} 文書（{info['old_synth_sources']}）。valid は共通。", "",
             "## 漏洩検査", "", f"- 評価本文との完全一致（落とした数）: {info['leak']['eval_text_overlap_dropped']} / 残存 {info['leak']['eval_text_overlap']}",
             f"- 評価セットの PERSON 表層の混入（合成層）: {info['leak']['eval_person_surface']}", "",
             "## 規約", "", "- ADDRESS は番地あり、LOCATION は番地なし。敬称・肩書きはスパン外。メール・URL・電話・郵便番号・日付は NER のラベルなし",
             "- GSD の写像しない型（NORP・DATE・EVENT・PRODUCT・…）は `neutral` に入れ、学習では損失を計算しない",
             "- 国名・公人は LOCATION / PERSON として付ける（現行どおり）"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None); ap.add_argument("--n-negatives", type=int, default=N_NEGATIVES); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synth", default=None); ap.add_argument("--gsd", default=None); ap.add_argument("--negatives", default=None)
    args = ap.parse_args(argv)
    gd = drive_dir()
    out_dir = args.out_dir or os.path.join(gd, OUT_DIR_NAME)
    paths = {"v10_train": os.path.join(gd, "data_v10", "train_v10.jsonl"), "v10_valid": os.path.join(gd, "data_v10", "valid_v10.jsonl"),
             "gsd": args.gsd or os.path.join(gd, "gsd_train", "gsd_train_ner.jsonl"),
             "synth": args.synth or os.path.join(gd, PLAN_DIR_NAME, FILLED_DIR_NAME, DOCS_FILE),
             "negatives": args.negatives or os.path.join(gd, "negatives_v8", "filtered_v1", "negatives_v8.jsonl")}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"入力がありません: {k} = {p}")
    from gsd_train_to_ner import load_eval_texts
    excl_person, _ = build_exclusions()
    eval_texts = load_eval_texts((os.path.join(gd, "blind_eval_v2", "blind_eval_v2.jsonl"),))
    res = build(v10_train=read_jsonl(paths["v10_train"]), v10_valid=read_jsonl(paths["v10_valid"]), gsd=read_jsonl(paths["gsd"]),
                synth_docs=read_jsonl(paths["synth"]), negatives=load_negatives(paths["negatives"]),
                hard_texts=load_hard_negative_texts(gd), excl_person=excl_person, eval_texts=eval_texts,
                seed=args.seed, n_negatives=args.n_negatives)
    if res["info"]["leak"]["eval_text_overlap"] or res["info"]["leak"]["eval_person_surface"]:
        raise SystemExit(f"漏洩検査で不合格: {res['info']['leak']}")
    os.makedirs(out_dir, exist_ok=True)
    files = {"train": "train_v11.jsonl", "train_plus_old": "train_v11_plus_old.jsonl", "valid": "valid_v11.jsonl"}
    for key, name in files.items():
        write_records(os.path.join(out_dir, name), res[key])
    with open(os.path.join(out_dir, "DATA_CARD_v11.md"), "w", encoding="utf-8") as f:
        f.write(data_card(res["info"], res["train"], res["valid"], res["train_plus_old"]))
    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"), "seed": args.seed, "inputs": {k: {"path": p, "sha256": file_sha256(p)} for k, p in paths.items()},
                "files": {k: {"name": v, "sha256": file_sha256(os.path.join(out_dir, v)), "rows": len(res[k])} for k, v in files.items()},
                "info": res["info"], "labels": dict(Counter(l for r in res["train"] for _, _, l in r["entities"]))}
    with open(os.path.join(out_dir, "manifest_v11.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"files": manifest["files"], "layers": res["info"]["layers"], "negatives": res["info"]["negatives"], "leak": res["info"]["leak"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

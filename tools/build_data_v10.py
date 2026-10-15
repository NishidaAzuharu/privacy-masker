"""学習データ v10 — v9 のまま、帳票・ログの「ラベル: 珍しい姓」を戻すための比率調整だけ。

v9（結果は開発記録、非公開）で、帳票にメール・URL の非エンティティ行を足したところ英数字の誤検出は消えたが、
固定OOD の帳票の珍しい姓（`宅地建物取引士: 雪見沢` `記録者：夜勤 朱鷺野` `受領者:企画総務部 犬 飼`）を平均も後半エポックも落とした。
「ラベル: 値」の値が PERSON である事前分布が薄まったと見て、v10 では次だけ変える（Qwen によるテンプレート生成は使わない: 2026-09-17 決定）。

| 変更 | v9 | v10 |
|---|---|---|
| 帳票のフレーム行で名が入る割合（`name_slot_ratio`） | 70% | **85%** |
| 帳票・ログの文書数 | 1,100 | **1,600** |
| 姓の候補のうち頻度の低い姓（`EXTRA_SURNAMES`）の重み | 1 | **3**（候補リストで 3 回並べる） |
| 字間空白の姓（`山 田` のように字の間に空白）| なし | 姓の表層の **5%** |
| メール・URL 行、カタカナ人名 600、町名 500、上限 5、LLM 負例 2,000 | — | v9 のまま |

使い方::

    PYTHONPATH=. .venv/bin/python tools/build_data_v10.py --out-dir <データフォルダ>/data_v10 [--seed 42] [--forms 1600]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import unicodedata
from typing import Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from build_data_v6 import _digest, data_card, stats, write_jsonl  # noqa: E402
from build_data_v7 import SURNAMES_KATA_JA, VALID_BANK_RATIO, to_halfwidth_kana  # noqa: E402
from build_data_v8 import EXTRA_SURNAMES, eval_person_exclusions, load_negatives, without_llm_negatives  # noqa: E402
import build_data_v9 as v9  # noqa: E402
from split_by_template import BASE_LABELS, DATA_FILES, load_records  # noqa: E402
from privacy_masker.data.ja_pools import SURNAMES_KANJI  # noqa: E402

SEED = 42
DEFAULT_FORMS = 1600
NAME_SLOT_RATIO = 0.85
EXTRA_SURNAME_WEIGHT = 3
SPACED_SURNAME_RATIO = 0.05


def surname_bank_v10(exclusions: set[str], rng: random.Random, extra_weight: int = EXTRA_SURNAME_WEIGHT) -> tuple[list[str], list[str]]:
    """頻度の低い姓を extra_weight 回並べ、抽選確率を上げる。train / valid の分割は姓の種類で行う（同じ姓は両側に出ない）。"""
    common = [s for s in SURNAMES_KANJI if s not in exclusions]
    extra = [s for s in EXTRA_SURNAMES if s not in exclusions]
    long_ex = [x for x in exclusions if len(x) >= 3]
    keep = lambda s: not (len(s) >= 3 and any(x.startswith(s) for x in long_ex))  # noqa: E731
    common = [s for s in common if keep(s)]
    extra = [s for s in extra if keep(s)]
    kinds = list(dict.fromkeys(common + extra))
    rng.shuffle(kinds)
    k = max(1, int(len(kinds) * VALID_BANK_RATIO))
    va_kinds, tr_kinds = set(kinds[:k]), set(kinds[k:])
    extra_set = set(extra)

    def weighted(kind_set: set[str]) -> list[str]:
        out: list[str] = []
        for s in kinds:
            if s in kind_set:
                out.extend([s] * (extra_weight if s in extra_set else 1))
        return out

    return weighted(tr_kinds), weighted(va_kinds)


def name_surface_v10(rng: random.Random, surnames: Sequence[str], given: Sequence[str],
                     kata: Sequence[str] = SURNAMES_KATA_JA, exclusions: set[str] | None = None) -> str:
    """v9 の表層（裸の姓 / 姓 名 / カタカナ姓 / 半角カナ姓）に、字間空白の姓（犬 飼）を 5% 足す。評価セットの表層は引き直す。"""
    exclusions = exclusions or set()
    for _ in range(20):
        r = rng.random()
        sur = rng.choice(surnames)
        if r < SPACED_SURNAME_RATIO and len(sur) >= 2:
            cand = rng.choice((" ", "　")).join(sur)              # 犬 飼、外　薗
        elif r < 0.72:
            cand = sur
        elif r < 0.84:
            cand = f"{sur}{rng.choice((' ', '　', ''))}{rng.choice(given)}"
        elif r < 0.92:
            cand = rng.choice(kata)
        else:
            cand = to_halfwidth_kana(rng.choice(kata))
        if cand not in exclusions and unicodedata.normalize("NFKC", cand) not in exclusions \
                and cand.replace(" ", "").replace("　", "") not in exclusions:
            return cand
    return sur


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=v9.DEFAULT_CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chat", type=int, default=v9.DEFAULT_CHAT)
    ap.add_argument("--dialogue", type=int, default=v9.DEFAULT_DIALOGUE)
    ap.add_argument("--halfwidth", type=int, default=v9.DEFAULT_HALFWIDTH)
    ap.add_argument("--forms", type=int, default=DEFAULT_FORMS)
    ap.add_argument("--katakana", type=int, default=v9.DEFAULT_KATAKANA)
    ap.add_argument("--towns", type=int, default=v9.DEFAULT_TOWNS)
    ap.add_argument("--negatives", default=None)
    ap.add_argument("--n-negatives", type=int, default=v9.DEFAULT_N_NEGATIVES)
    ap.add_argument("--name-slot-ratio", type=float, default=NAME_SLOT_RATIO)
    args = ap.parse_args(argv)

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書")
    neg_path = args.negatives or os.path.join(gd, "negatives_v8", "filtered_v1", "negatives_v8.jsonl")
    all_neg = load_negatives(neg_path)
    hard = v9.load_hard_negative_texts(gd)
    negatives = v9.select_negatives(all_neg, args.n_negatives, random.Random(args.seed), hard)
    print(f"LLM 負例: {len(all_neg)} → {len(negatives)} 件（ハード {sum(1 for r in negatives if r['text'] in hard)}）")
    eval_person = eval_person_exclusions(v9.collect_eval_surfaces(("PERSON",)))
    eval_place = v9.collect_eval_surfaces(("LOCATION", "ADDRESS"))

    train, valid, info = v9.build(
        records, cap=args.cap, seed=args.seed, n_chat=args.chat, n_dialogue=args.dialogue, n_halfwidth=args.halfwidth,
        n_forms=args.forms, n_katakana=args.katakana, n_towns=args.towns, negatives=negatives, hard_texts=hard,
        eval_person=eval_person, eval_place=eval_place,
        form_kwargs={"name_slot_ratio": args.name_slot_ratio, "name_fn": name_surface_v10},
        surname_bank_fn=surname_bank_v10)
    if info["leak"]:
        raise SystemExit(f"train / valid にテンプレート重複が {info['leak']} 件あります。生成を中止")
    info["v10"] = {"name_slot_ratio": args.name_slot_ratio, "extra_surname_weight": EXTRA_SURNAME_WEIGHT,
                   "spaced_surname_ratio": SPACED_SURNAME_RATIO}
    forms = [d for d in train + valid if d["source"] == v9.FORM_SOURCE]
    spaced = sum(1 for d in forms for s, e, l in d["entities"] if l == "PERSON" and any(c in d["text"][s:e] for c in " 　")
                 and len(d["text"][s:e].replace(" ", "").replace("　", "")) <= 3 and not any(g in d["text"][s:e] for g in v9.GIVEN_KANJI[:0]))
    info["v10"]["spaced_surname_spans"] = spaced

    os.makedirs(args.out_dir, exist_ok=True)
    train_noneg = without_llm_negatives(train)
    write_jsonl(os.path.join(args.out_dir, "train_v10.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "train_v10_noneg.jsonl"), train_noneg)
    write_jsonl(os.path.join(args.out_dir, "valid_v10.jsonl"), valid)
    st_train, st_valid, st_all = stats(train), stats(valid), stats(train + valid)
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [f for _, f in DATA_FILES] + [os.path.relpath(neg_path, gd)],
        "seed": args.seed, "cap": args.cap, "labels": list(BASE_LABELS),
        "train": {"docs": len(train), "text_digest": _digest(train), **st_train},
        "train_noneg": {"docs": len(train_noneg), "text_digest": _digest(train_noneg)},
        "valid": {"docs": len(valid), "text_digest": _digest(valid), **st_valid},
        "processing": {k: v for k, v in info.items() if k not in ("before", "person_made")},
        "person_variants_made": dict(info["person_made"]),
        "files": {"train": "train_v10.jsonl", "train_noneg": "train_v10_noneg.jsonl", "valid": "valid_v10.jsonl"},
    }
    with open(os.path.join(args.out_dir, "manifest_v10.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    card = data_card("学習データ v10", info["before"], st_all, st_train, st_valid, info, version="v10")
    card = card.replace(
        "- LLM 生成の新規負例（CPU では作れない。`qwen_synth_pii_data` 系ノートブックで別途）",
        "- 評価セット由来の人名・地名表層（帳票・カタカナ・町名の候補から機械的に除外）")
    card += (v9.v9_appendix(info, train, valid).replace("## v9 の追加分", "## v9 と同じ追加分")
             .replace("`train_v9_noneg.jsonl`", "`train_v10_noneg.jsonl`"))
    card += ("\n## v10 の変更\n\n"
             f"- 帳票のフレーム行で名が入る割合 {args.name_slot_ratio:.0%}（v9 70%）、帳票文書 {args.forms}（v9 1,100）\n"
             f"- 頻度の低い姓の重み {EXTRA_SURNAME_WEIGHT}、字間空白の姓 {SPACED_SURNAME_RATIO:.0%}（実際の span 数 {spaced}）\n")
    with open(os.path.join(args.out_dir, "DATA_CARD_v10.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

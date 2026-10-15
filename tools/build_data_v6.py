"""学習データ v6 を組む — 汎化を優先した再設計。

当時の本番モデルの学習データ（wiki 実テキスト + 合成 3 層、24,035 文書）で見えた過学習の構造:

- **テンプレートの反復**: 手テンプレ合成は 6,235 文書がわずか 407 テンプレート（数字・空白を潰した
  署名）。1 テンプレート平均 15 文書、最大 150。上位 50 テンプレートで train の 26%。
  8 エポックで平均 29 回、上位は 1,000 回以上同じ文型を見る。
- **valid の弁別能力ゼロ**: valid の 57〜69% が train と同じテンプレートで、3 つの別モデルが
  valid の全スパンで一致した（OOD では 30〜84% の文書で違う）。
- **書式の偏り**: ADDRESS 4,698 件の 100% が都道府県始まり。番地先行の住所は危険率 35〜53%。
  PERSON は空白含む 8.7%（OpenPII 71.5%）、3 要素以上 9 件、かな+英字 0 件。
- **負例の縮退**: 元 15.5% → train 5.1%。手テンプレ合成の負例は本文 18 種類。
- **長文が無い**: 400 字超は 1.4%（盲検 v1 は 36%）。

v6 の方針（すべて CPU で完結、GPU 学習は Colab）:

1. wiki の 地名 ラベルのうち番地付き住所を ADDRESS に付け替える（ラベルノイズ 5 件）
2. **テンプレート当たりの上限**（既定 4 文書）— 同じ文型を何十回も見せない
3. **住所の書式を足す**（`augment_address_formats` の 7 変種）— 都道府県始まり一辺倒を崩す
4. **人名の書式を足す** — 漢字姓 空白 漢字名、3 要素、かな+英字、英字フルネーム
5. **テンプレート単位で train / valid を分ける**（`split_by_template.grouped_split`）
6. **長文を作る** — 別テンプレートの文書を 2〜4 本つないで 400 字超を増やす（分割後に各側で）
7. 負例は全部残す（実データ out_ner の 716 件が中心）

書式の変種は train と valid の**両側**で作るので、v6 の valid はそのまま
「テンプレート未知 + 書式ストレス」の評価セットになる。

使い方::

    PYTHONPATH=. .venv/bin/python tools/build_data_v6.py --out-dir <dir> [--cap 4] [--seed 42]

出力: train_v6.jsonl / valid_v6.jsonl / manifest_v6.json / DATA_CARD_v6.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from augment_address_formats import augment as augment_addresses  # noqa: E402
from augment_address_formats import rewrite  # noqa: E402
from split_by_template import (  # noqa: E402
    BASE_LABELS,
    DATA_FILES,
    grouped_split,
    load_records,
    template_signature,
)
from privacy_masker.data.ja_pools import (  # noqa: E402
    GIVEN_KANJI, GIVEN_KATA, SURNAMES_KANJI, SURNAMES_KATA,
)

SEED = 42
DEFAULT_CAP = 4
LONG_TARGET_RATIO = 0.12      # 400 字超の文書をこの割合まで増やす
LONG_MIN_CHARS = 400
PERSON_VARIANT_RATIO = 0.35   # PERSON を含む文書のうち書き換える割合

# 番地付きの住所（街区レベル）。丁目だけの町名は LOCATION のまま残す
_STREET_LEVEL_RE = re.compile(r"番地|[0-9０-９]+番[0-9０-９]*号|号室|[0-9０-９]+[-－‐―ー−][0-9０-９]+")

# 平仮名の名（一般的なもの。個人を指さない）
GIVEN_HIRA: tuple[str, ...] = (
    "さくら", "ひなた", "あおい", "ゆい", "りん", "はな", "みお", "ゆな", "ひろき", "たくみ",
    "そうた", "はると", "ゆうと", "りく", "けんた", "だいき", "なおき", "まさひろ", "たかし",
    "ひろし", "けいこ", "ようこ", "みちこ", "かずこ", "ゆみこ", "なおこ", "まゆみ", "あきこ",
    "さとみ", "えみ", "かな", "まり", "えりか", "なな", "ゆか", "あや", "みさき", "ともこ",
    "たけし", "しんじ", "こうじ", "まこと", "つよし", "あきら", "ゆうき", "しょうた",
)
_KANJI_RE = re.compile(r"^[一-鿿々〆ヶ]+$")
_LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")


# --------------------------------------------------------------- 1. wiki の付け替え

def relabel_wiki_addresses(records: list[dict]) -> int:
    """out_ner の LOCATION で番地付きのものを ADDRESS に。件数を返す。"""
    n = 0
    for r in records:
        if r["source"] != "out_ner":
            continue
        for ent in r["entities"]:
            s, e, lab = ent
            if lab == "LOCATION" and _STREET_LEVEL_RE.search(r["text"][s:e]):
                ent[2] = "ADDRESS"
                n += 1
    return n


# --------------------------------------------------------------- 2. テンプレート上限

def cap_templates(records: Sequence[dict], cap: int, rng: random.Random) -> tuple[list[dict], Counter]:
    """loose 署名ごとに最大 cap 文書を残す。本文の完全一致重複も落とす。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for r in records:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        groups[template_signature(r, loose=True)].append(r)
    kept: list[dict] = []
    dropped: Counter = Counter()
    for sig, docs in groups.items():
        if len(docs) > cap:
            rng.shuffle(docs)
            for d in docs[cap:]:
                dropped[d["source"]] += 1
            docs = docs[:cap]
        kept.extend(docs)
    return kept, dropped


# --------------------------------------------------------------- 4. 人名の書式

def _harvest_name_parts(records: Sequence[dict]) -> tuple[list[str], list[str]]:
    """train の空白区切り英字 2 要素の人名から、英字の姓と名を集める。"""
    given: set[str] = set()
    sur: set[str] = set()
    for r in records:
        t = r["text"]
        for s, e, lab in r["entities"]:
            if lab != "PERSON":
                continue
            ps = [p for p in re.split(r"[ 　]+", t[s:e]) if p]
            if len(ps) == 2 and all(_LATIN_RE.match(p) for p in ps):
                given.add(ps[0])
                sur.add(ps[1])
    return sorted(given), sorted(sur)


def person_variants(rng: random.Random, latin_given: Sequence[str],
                    latin_sur: Sequence[str]) -> dict[str, callable]:
    """人名の書式ごとの生成関数。学習データに薄い形だけを作る。"""
    def kanji_space() -> str:              # 山田 太郎（漢字2+空白+漢字2 は train に 259 件）
        return f"{rng.choice(SURNAMES_KANJI)} {rng.choice(GIVEN_KANJI)}"

    def three_hira_kanji() -> str:         # かな 麻里 松本（OpenPII 型。train に 9 件）
        return f"{rng.choice(GIVEN_HIRA)} {rng.choice(GIVEN_KANJI)} {rng.choice(SURNAMES_KANJI)}"

    def hira_latin() -> str:               # なほみ Ivers（かな+英字。train に 0 件）
        sur = rng.choice(latin_sur) if latin_sur else rng.choice(SURNAMES_KATA)
        return f"{rng.choice(GIVEN_HIRA)} {sur}"

    def latin_kanji() -> str:              # Vildan 上田
        given = rng.choice(latin_given) if latin_given else rng.choice(GIVEN_KATA)
        return f"{given} {rng.choice(SURNAMES_KANJI)}"

    def kata_full() -> str:                # ジョン・スミス
        return f"{rng.choice(GIVEN_KATA)}・{rng.choice(SURNAMES_KATA)}"

    def hira_only() -> str:                # さくら（ひらがな単体は train 2,169 件あるが 2 文字以下が中心）
        return rng.choice(GIVEN_HIRA)

    def latin_full() -> str:               # Maria Andryc
        if latin_given and latin_sur:
            return f"{rng.choice(latin_given)} {rng.choice(latin_sur)}"
        return f"{rng.choice(GIVEN_KATA)} {rng.choice(SURNAMES_KATA)}"

    return {
        "漢字姓 空白 漢字名": kanji_space,
        "かな名 漢字名 漢字姓(3要素)": three_hira_kanji,
        "かな名 英字姓": hira_latin,
        "英字名 漢字姓": latin_kanji,
        "カタカナ・カタカナ": kata_full,
        "ひらがな単体": hira_only,
        "英字 空白 英字": latin_full,
    }


def augment_persons(records: Sequence[dict], *, ratio: float, rng: random.Random,
                    latin_given: Sequence[str], latin_sur: Sequence[str]) -> tuple[list[dict], Counter]:
    """PERSON を含む文書の一部で、1 つの PERSON を薄い書式に書き換えた文書を作る。"""
    gens = person_variants(rng, latin_given, latin_sur)
    names = list(gens)
    out: list[dict] = []
    made: Counter = Counter()
    for r in records:
        idxs = [i for i, (_, _, l) in enumerate(r["entities"]) if l == "PERSON"]
        if not idxs or rng.random() > ratio:
            continue
        i = rng.choice(idxs)
        vname = rng.choice(names)
        new = gens[vname]()
        s, e, _ = r["entities"][i]
        if new == r["text"][s:e]:
            continue
        got = rewrite(r, i, new)
        if got is None:
            continue
        got["variant"] = f"person:{vname}"
        out.append(got)
        made[vname] += 1
    return out, made


# --------------------------------------------------------------- 6. 長文

def make_long_docs(records: Sequence[dict], *, target_ratio: float,
                   rng: random.Random) -> list[dict]:
    """別テンプレートの文書を 2〜4 本つないで 400 字超の文書を作る。"""
    n_total = len(records)
    n_long_now = sum(1 for r in records if len(r["text"]) > LONG_MIN_CHARS)
    need = int(n_total * target_ratio) - n_long_now
    if need <= 0:
        return []
    pool = [r for r in records if len(r["text"]) <= LONG_MIN_CHARS and r["entities"]]
    rng.shuffle(pool)
    out: list[dict] = []
    i = 0
    while len(out) < need and i + 2 <= len(pool):
        k = rng.choice((2, 3, 4))
        parts = pool[i:i + k]
        i += k
        sigs = {template_signature(p, loose=True) for p in parts}
        if len(sigs) < len(parts):
            continue                      # 同じテンプレートを繋いでも長文の意味がない
        text = ""
        ents = []
        for p in parts:
            if text:
                text += "\n\n"
            off = len(text)
            text += p["text"]
            ents += [[s + off, e + off, l] for s, e, l in p["entities"]]
        if len(text) <= LONG_MIN_CHARS:
            continue
        out.append({"text": text, "entities": sorted(ents, key=lambda x: (x[0], x[1])),
                    "source": "concat", "variant": f"concat:{k}"})
    return out


# --------------------------------------------------------------- 統計

_PREF_RE = re.compile(r"^(北海道|京都府|大阪府|東京都|.{2,3}県)")


def stats(records: Sequence[dict]) -> dict:
    c: Counter = Counter()
    ent: Counter = Counter()
    addr: Counter = Counter()
    person: Counter = Counter()
    sigs = set()
    neg_texts: set[str] = set()
    for r in records:
        t = r["text"]
        c["docs"] += 1
        c["by_source|" + r["source"]] += 1
        if not r["entities"]:
            c["negatives"] += 1
            neg_texts.add(t)
        if len(t) > LONG_MIN_CHARS:
            c["long"] += 1
        c["chars"] += len(t)
        sigs.add(template_signature(r, loose=True))
        for s, e, l in r["entities"]:
            ent[l] += 1
            a = t[s:e]
            if l == "ADDRESS":
                addr["n"] += 1
                if _PREF_RE.match(a):
                    addr["都道府県始まり"] += 1
                if a[:1].isdigit() or (a[:1] and a[0] in "０１２３４５６７８９"):
                    addr["数字始まり"] += 1
            if l == "PERSON":
                person["n"] += 1
                ps = [p for p in re.split(r"[ 　]+", a) if p]
                if len(ps) >= 2:
                    person["空白含む"] += 1
                # 監査と同じ定義: 3 要素以上で、少なくとも 1 要素が 2 文字以上
                # （「山 田 太 郎」のような字間空白を多要素の人名と混同しない）
                if len(ps) >= 3 and any(len(p) >= 2 for p in ps):
                    person["3要素以上"] += 1
                if re.search(r"[ぁ-ゖ]", a) and re.search(r"[A-Za-z]", a):
                    person["かな+英字"] += 1
                if len(a) <= 2:
                    person["2文字以下"] += 1
    c["negatives_distinct"] = len(neg_texts)
    return {"counts": dict(c), "entities": dict(ent), "address": dict(addr),
            "person": dict(person), "templates": len(sigs)}


def _pct(a: int, b: int) -> str:
    return f"{a / max(b, 1) * 100:.1f}%"


def data_card(name: str, before: dict, after: dict, train: dict, valid: dict,
              info: dict, version: str = "v6") -> str:
    def row(label, key_fn):
        return f"| {label} | {key_fn(before)} | {key_fn(after)} | {key_fn(train)} | {key_fn(valid)} |"
    v = version
    L = [f"# {name}", "", f"作成 {dt.date.today().isoformat()} / seed {info['seed']} / "
         f"テンプレート上限 {info['cap']} 文書",
         "", f"| 指標 | 元データ(当時の本番) | {v} 全体 | {v} train | {v} valid |", "|---|---|---|---|---|",
         row("文書数", lambda d: d["counts"]["docs"]),
         row("異なるテンプレート(loose)", lambda d: d["templates"]),
         row("文書/テンプレート", lambda d: f"{d['counts']['docs'] / max(d['templates'], 1):.2f}"),
         row("負例", lambda d: f"{d['counts'].get('negatives', 0)} ({_pct(d['counts'].get('negatives', 0), d['counts']['docs'])})"),
         row("  負例の異なる本文", lambda d: d["counts"].get("negatives_distinct", 0)),
         row("400 字超", lambda d: f"{d['counts'].get('long', 0)} ({_pct(d['counts'].get('long', 0), d['counts']['docs'])})"),
         row("エンティティ数", lambda d: sum(d["entities"].values())),
         row("ADDRESS", lambda d: d["entities"].get("ADDRESS", 0)),
         row("  都道府県始まり", lambda d: _pct(d["address"].get("都道府県始まり", 0), d["address"].get("n", 0))),
         row("  数字始まり", lambda d: _pct(d["address"].get("数字始まり", 0), d["address"].get("n", 0))),
         row("PERSON", lambda d: d["entities"].get("PERSON", 0)),
         row("  空白含む", lambda d: _pct(d["person"].get("空白含む", 0), d["person"].get("n", 0))),
         row("  3要素以上", lambda d: d["person"].get("3要素以上", 0)),
         row("  かな+英字", lambda d: d["person"].get("かな+英字", 0)),
         row("  2文字以下", lambda d: _pct(d["person"].get("2文字以下", 0), d["person"].get("n", 0))),
         "",
         "## 処理の内訳", "",
         f"- wiki の 地名→ADDRESS 付け替え: {info['relabeled']} 件",
         f"- テンプレート上限で落とした文書: {sum(info['dropped'].values())} 件 "
         f"（{', '.join(f'{k} {v}' for k, v in sorted(info['dropped'].items()))}）",
         f"- 住所の書式変種: {info['addr_variants']} 文書",
         f"- 人名の書式変種: {info['person_variants']} 文書 "
         f"（{', '.join(f'{k} {v}' for k, v in info['person_made'].most_common())}）",
         f"- 長文（2〜4 本連結）: train {info['long_train']} / valid {info['long_valid']} 文書",
         f"- train / valid のテンプレート重複: {info['leak']} 件（0 が期待値）",
         "",
         "## valid の使い方", "",
         "valid はテンプレートが train と重複せず、住所・人名の書式変種と長文を含む。",
         "エポック選択は valid の span F1 ではなく **PERSON / ADDRESS の未被覆文字数**（学習ノートの",
         "`id_mask_leak_chars`）を主指標にし、F1 は補助にする。学習後の比較は文字被覆ベースの",
         "被覆ゼロ（`tools/eval_ood_f1.py` の `uncovered_gold.csv`、`tools/eval_all.py` の完全見逃し）で見る。",
         "最終判定は 4 評価セット（固定OOD / 盲検v1 / GSD / OpenPII）で行う。",
         "",
         "## 意図的に入れていないもの", "",
         "- OpenPII / GSD / 盲検 v1 / 固定 OOD の本文・表層（評価セットの漏洩を避ける）",
         "- LLM 生成の新規負例（CPU では作れない。`qwen_synth_pii_data` 系ノートブックで別途）",
         ]
    return "\n".join(L) + "\n"


def _digest(records: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for t in sorted(r["text"] for r in records):
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = {"text": r["text"], "entities": r["entities"], "source": r["source"]}
            if "variant" in r:
                row["variant"] = r["variant"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(records: list[dict], *, cap: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    before = stats(records)
    relabeled = relabel_wiki_addresses(records)

    capped, dropped = cap_templates(records, cap, rng)

    latin_given, latin_sur = _harvest_name_parts(capped)
    addr_aug, _, _ = augment_addresses(capped, per_doc=1, seed=seed)
    person_aug, person_made = augment_persons(capped, ratio=PERSON_VARIANT_RATIO, rng=rng,
                                              latin_given=latin_given, latin_sur=latin_sur)
    for r in addr_aug:
        r["variant"] = "address:" + r["variant"]
    combined = capped + addr_aug + person_aug

    train, valid = grouped_split(combined, loose=True, seed=seed)
    tsig = {template_signature(r, loose=True) for r in train}
    leak = sum(1 for r in valid if template_signature(r, loose=True) in tsig)

    long_train = make_long_docs(train, target_ratio=LONG_TARGET_RATIO, rng=rng)
    long_valid = make_long_docs(valid, target_ratio=LONG_TARGET_RATIO, rng=rng)
    train += long_train
    valid += long_valid
    rng.shuffle(train)
    rng.shuffle(valid)

    info = {"seed": seed, "cap": cap, "relabeled": relabeled, "dropped": dict(dropped),
            "addr_variants": len(addr_aug), "person_variants": len(person_aug),
            "person_made": person_made, "long_train": len(long_train),
            "long_valid": len(long_valid), "leak": leak, "before": before}
    return train, valid, info


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help="loose 署名 1 つにつき残す最大文書数（既定 4）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.cap < 1:
        raise SystemExit("--cap は 1 以上")

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書")
    train, valid, info = build(records, cap=args.cap, seed=args.seed)
    if info["leak"]:
        raise SystemExit(f"train / valid にテンプレート重複が {info['leak']} 件あります。生成を中止")

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train_v6.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "valid_v6.jsonl"), valid)
    st_train, st_valid = stats(train), stats(valid)
    st_all = stats(train + valid)
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [f for _, f in DATA_FILES], "seed": args.seed, "cap": args.cap,
        "labels": list(BASE_LABELS),
        "train": {"docs": len(train), "text_digest": _digest(train), **st_train},
        "valid": {"docs": len(valid), "text_digest": _digest(valid), **st_valid},
        "processing": {k: v for k, v in info.items() if k not in ("before", "person_made")},
        "person_variants_made": dict(info["person_made"]),
    }
    with open(os.path.join(args.out_dir, "manifest_v6.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    card = data_card("学習データ v6", info["before"], st_all, st_train, st_valid, info)
    with open(os.path.join(args.out_dir, "DATA_CARD_v6.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

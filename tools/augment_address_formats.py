"""未対応の住所書式を学習データに足す（優先度 2）。

現行の学習データの ADDRESS は **4,211 件すべてが都道府県から始まる**（valid の 487 件も 100%）。
一方 OpenPII の ADDRESS 455 件は都道府県始まりが 0 件で、数字始まり 83 件、
空白の直後に数字があるものが 161 件ある。

    北小路町 2-1-9
    4-15-39 東上善寺町
    8-8-17番地三隅町河内

このため現行 valid では「都道府県のない住所を拾えるか」を評価できない。実測でも
OpenPII の危険（正解 span の文字被覆率 50% 未満）は ADDRESS に集中している
（30m 8ep で 60 件中 53 件、30m 20ep で 146 件中 131 件）。

このツールは既存の住所を分解して、欠けている書式へ書き換えた文書を作る。
原文の他のエンティティの span は差分だけずらして保つ。

使い方::

    # 書式の分布を見るだけ
    python tools/augment_address_formats.py --report

    # 学習用の追加データと、書式評価用の held-out セットを書き出す
    python tools/augment_address_formats.py --out-dir <dir>

テンプレート単位で train / eval を分けるので（``split_by_template`` と同じ署名）、
同じテンプレートから作った変種が両側に出ることはない。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from split_by_template import load_records, template_signature  # noqa: E402

SEED = 42
EVAL_RATIO = 0.15

# 都道府県 / 郡・市区町村 / 残り（町名 + 番地 + 建物）
_PREF_CITY_RE = re.compile(
    r"^(?P<pref>東京都|北海道|京都府|大阪府|.{2,3}?県)"
    r"(?P<city>(?:.{1,6}?郡)?.{1,8}?[市区町村](?:.{1,4}?区)?)"
    r"(?P<rest>.+)$"
)
# 残りを 町名 / 番地 / 建物 に割る。番地は数字で始まり、数字・ハイフン・丁目・番・号で続く
_BANCHI_RE = re.compile(
    r"(?P<banchi>[0-9０-９]+(?:[-－‐‑‒–—―−ー][0-9０-９]+)*"
    r"(?:丁目[0-9０-９]*(?:[-－][0-9０-９]+)*)?"
    r"(?:番地?[0-9０-９]*)?(?:号(?![室階]))?)"
)
def split_address(addr: str) -> dict[str, str] | None:
    """住所を pref / city / town / banchi / building に分ける。分解できなければ None。"""
    m = _PREF_CITY_RE.match(addr.strip())
    if not m:
        return None
    rest = m.group("rest")
    bm = _BANCHI_RE.search(rest)
    if not bm or not bm.group("banchi"):
        return None
    town = rest[:bm.start()]
    banchi = bm.group("banchi")
    building = rest[bm.end():]
    # 「北海道旭川市緑が丘南５条5-4-8」のような ○条 / ○線 型は、町名の中の数字を
    # 番地と誤認して town='緑が丘南' banchi='５' に切れてしまう。並べ替えると
    # 壊れたラベルになるので分解しない（実測 4,698 件中 33 件 = 0.70%）
    if building[:1] and building[0] in "条線丁番号":
        return None
    # 町名が無い住所（「山口県萩市4-8-11」）もある。並べ替えはできないが
    # 都道府県を落とす変種は作れるので、town="" のまま返す
    return {"pref": m.group("pref"), "city": m.group("city"), "town": town,
            "banchi": banchi, "building": building}


# ---- 書式のバリアント（それぞれ ADDRESS として注釈する文字列を返す）----
# ライブラリの規約は「番地があれば ADDRESS」なので、都道府県を落としても
# 番地が残る限り ADDRESS のままにする。


def v_drop_pref(p: dict[str, str]) -> str:
    return f"{p['city']}{p['town']}{p['banchi']}{p['building']}"


def v_drop_pref_city(p: dict[str, str]) -> str:
    return f"{p['town']}{p['banchi']}{p['building']}"


def v_town_space_banchi(p: dict[str, str]) -> str:
    return f"{p['town']} {p['banchi']}{p['building']}"


def v_banchi_first(p: dict[str, str]) -> str:
    return f"{p['banchi']} {p['town']}{p['building']}"


def v_banchi_first_banchi_suffix(p: dict[str, str]) -> str:
    b = p["banchi"] if p["banchi"].endswith(("番", "番地", "号")) else p["banchi"] + "番地"
    return f"{b} {p['town']}{p['building']}"


def v_banchi_first_no_space(p: dict[str, str]) -> str:
    b = p["banchi"] if p["banchi"].endswith(("番", "番地", "号")) else p["banchi"] + "番地"
    return f"{b}{p['town']}{p['building']}"


def v_town_no_pref_particle(p: dict[str, str]) -> str:
    return f"{p['town']}の{p['banchi']}{p['building']}"


VARIANTS: dict[str, Callable[[dict[str, str]], str]] = {
    "都道府県なし": v_drop_pref,
    "町名+番地のみ": v_drop_pref_city,
    "町名 空白 番地": v_town_space_banchi,
    "番地 空白 町名": v_banchi_first,
    "番地+番地語 町名": v_banchi_first_banchi_suffix,
    "番地+番地語 町名(空白なし)": v_banchi_first_no_space,
    "町名の番地": v_town_no_pref_particle,
}


def rewrite(rec: dict, addr_idx: int, new_addr: str) -> dict | None:
    """rec の addr_idx 番目のエンティティを new_addr に差し替えた新しいレコードを作る。

    後続のエンティティ span は長さの差分だけずらす。跨いでいるものがあれば None。
    """
    text = rec["text"]
    s, e, lab = rec["entities"][addr_idx]
    for i, (s2, e2, _) in enumerate(rec["entities"]):
        if i == addr_idx:
            continue
        if s2 < e and s < e2:
            return None          # 住所と重なるエンティティがあると span を作り直せない
    delta = len(new_addr) - (e - s)
    ents = []
    for i, (s2, e2, l2) in enumerate(rec["entities"]):
        if i == addr_idx:
            ents.append([s, s + len(new_addr), lab])
        elif s2 >= e:
            ents.append([s2 + delta, e2 + delta, l2])
        else:
            ents.append([s2, e2, l2])
    out = {"text": text[:s] + new_addr + text[e:], "entities": sorted(ents, key=lambda x: (x[0], x[1])),
           "source": rec["source"]}
    # 自己検査: すべての span が原文の該当文字列を指しているか
    if out["text"][s:s + len(new_addr)] != new_addr:
        return None
    return out


def address_format_stats(records: Sequence[dict]) -> Counter:
    c: Counter = Counter()
    pref = re.compile(r"^(北海道|京都府|大阪府|東京都|.{2,3}県)")
    for r in records:
        t = r["text"]
        for s, e, lab in r["entities"]:
            if lab != "ADDRESS":
                continue
            a = t[s:e]
            c["total"] += 1
            if pref.match(a):
                c["都道府県始まり"] += 1
            if a[:1].isdigit() or (a[:1] and a[0] in "０１２３４５６７８９"):
                c["数字始まり"] += 1
            if re.search(r"\s[0-9０-９]", a):
                c["空白+数字"] += 1
            if not pref.match(a) and re.search(r"[市区町村]", a):
                c["市区町村始まり"] += 1
    return c


def augment(records: Sequence[dict], *, per_doc: int = 2,
            seed: int = SEED) -> tuple[list[dict], Counter, Counter]:
    """住所を含む文書から、欠けている書式の変種を作る。"""
    rnd = random.Random(seed)
    out: list[dict] = []
    made: Counter = Counter()
    skipped: Counter = Counter()
    names = list(VARIANTS)
    for rec in records:
        idxs = [i for i, (_, _, l) in enumerate(rec["entities"]) if l == "ADDRESS"]
        if not idxs:
            continue
        for i in idxs:
            s, e, _ = rec["entities"][i]
            parts = split_address(rec["text"][s:e])
            if parts is None:
                skipped["分解できない"] += 1
                continue
            for vname in rnd.sample(names, min(per_doc, len(names))):
                new_addr = VARIANTS[vname](parts)
                if new_addr == rec["text"][s:e]:
                    skipped["変化なし"] += 1
                    continue
                if new_addr != new_addr.strip() or "  " in new_addr:
                    skipped["町名が無く並べ替えできない"] += 1
                    continue
                got = rewrite(rec, i, new_addr)
                if got is None:
                    skipped["span を作り直せない"] += 1
                    continue
                got["variant"] = vname
                out.append(got)
                made[vname] += 1
    return out, made, skipped


def print_format_stats(c: Counter) -> None:
    total = max(c["total"], 1)
    print(f"  {'ADDRESS 総数':<16}{c['total']:>7}")
    for k in ("都道府県始まり", "市区町村始まり", "数字始まり", "空白+数字"):
        print(f"  {k:<16}{c[k]:>7}  ({c[k] / total * 100:>5.1f}%)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--per-doc", type=int, default=2,
                    help="1 つの住所から作る変種の数（既定 2）")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    before = address_format_stats(records)
    print("■ 元データの ADDRESS 書式")
    print_format_stats(before)

    aug, made, skipped = augment(records, per_doc=args.per_doc)
    print(f"\n■ 生成した変種: {len(aug)} 文書")
    for k, v in made.most_common():
        print(f"  {k:<24}{v:>7}")
    if skipped:
        print("  作れなかったもの: " + "  ".join(f"{k}={v}" for k, v in skipped.most_common()))

    print("\n■ 変種の ADDRESS 書式")
    print_format_stats(address_format_stats(aug))

    # テンプレート単位で train / eval に分ける（同じ元文書の変種が両側に出ないように、
    # 元文書のテンプレート署名でグループ化する）
    groups: dict[str, list[dict]] = {}
    for r in aug:
        groups.setdefault(template_signature(r), []).append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    target = len(aug) * EVAL_RATIO
    ev: list[dict] = []
    tr: list[dict] = []
    n = 0
    for k in keys:
        if n + len(groups[k]) <= target:
            ev += groups[k]
            n += len(groups[k])
        else:
            tr += groups[k]
    print(f"\n■ 分割: 学習追加 {len(tr)} / 書式評価 {len(ev)}")
    print(f"  評価側のテンプレートが学習側にもあるもの: "
          f"{sum(1 for r in ev if template_signature(r) in {template_signature(x) for x in tr})} 件")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for fname, data in (("address_formats_train.jsonl", tr),
                            ("address_formats_eval.jsonl", ev)):
            with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps({"text": r["text"], "entities": r["entities"],
                                        "source": r["source"], "variant": r["variant"]},
                                       ensure_ascii=False) + "\n")
        print(f"\n書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

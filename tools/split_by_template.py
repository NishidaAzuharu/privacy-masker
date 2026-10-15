"""テンプレート単位で train / valid を分ける（優先度 1）。

現行の分割は「本文の完全一致」だけを重複とみなすため、合成データの
「同じ文章の名前や住所だけを差し替えた例」が train と valid の両方に残る。
実測では valid 2,623 件のうち 1,499 件（57.1%）が、エンティティを型名に置換した
文面まで一致していた（内訳: 合成 v4 82.8% / v5c 69.3% / v5e 57.6% / out_ner 0.4%）。

このため現行 valid は「既知の文型で別の固有表現を処理できるか」しか測れない。
未知の文型・書式への汎化を測るには、同じテンプレートから作られた例が
train と valid に分かれない分割が必要になる。

テンプレート署名 = 本文のエンティティ部分を ``<TYPE>`` に置換した文字列。
生成時の template_id が残っていないデータから、事後的にテンプレートを復元する近似。

  紹介状送付先: ○○病院 御中（○○県○○市……）
  → 紹介状送付先: <FACILITY> 御中（<ADDRESS>）

同じ署名を持つ文書は必ず同じ側に入れる（グループ分割）。

使い方::

    # 現行 valid のうちテンプレートが train と重複しないものを書き出す
    python tools/split_by_template.py --report

    # テンプレート単位で分割し直して jsonl を書き出す
    python tools/split_by_template.py --out-dir <dir>

``--loose`` を付けると数字・空白を潰した署名でグループ化する（近似重複も同じ側に寄せる）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402

# 学習 Notebook（train_ja_mask_charhead_*.ipynb）と同じ構成・同じラベル写像
DATA_FILES: tuple[tuple[str, str], ...] = (
    ("out_ner", "out_ner.jsonl"),
    ("synth_v4", "synthetic_pii_v4.jsonl"),
    ("synth_v5c", "synthetic_pii_v5c.jsonl"),
    ("synth_v5e", "synthetic_pii_v5e.jsonl"),
)
LABEL_MAP = {
    "人名": "PERSON", "法人名": "ORG", "政治的組織名": "ORG", "その他の組織名": "ORG",
    "地名": "LOCATION", "施設名": "FACILITY",
}
REMOVE_LABELS = frozenset({"製品名", "イベント名"})
BASE_LABELS: tuple[str, ...] = ("ADDRESS", "FACILITY", "LOCATION", "ORG", "PERSON")
SEED = 42
VALID_RATIO = 0.1

# ゆるい署名: 数字を 0 に、空白を 1 個に潰す（「1丁目」と「3丁目」を同じ型とみなす）
_DIGITS = re.compile(r"[0-9０-９]+")
_SPACES = re.compile(r"[ 　\t]+")


def _coerce_entity(item: object, text_len: int) -> tuple[int, int, str] | None:
    """Notebook の normalize_records と同じ寛容さで 1 エンティティを読む。"""
    while isinstance(item, list) and len(item) == 1 and isinstance(item[0], (list, dict, tuple)):
        item = item[0]
    if isinstance(item, dict) and all(k in item for k in ("start", "end", "label")):
        s, e, lab = item["start"], item["end"], item["label"]
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        s, e, lab = item[0], item[1], item[2]
    else:
        return None
    try:
        s, e = int(s), int(e)
    except (TypeError, ValueError):
        return None
    s, e = max(0, s), min(text_len, e)
    if s >= e:
        return None
    lab = str(lab).strip()
    if lab in LABEL_MAP:
        return s, e, LABEL_MAP[lab]
    if lab in BASE_LABELS:
        return s, e, lab
    return None      # REMOVE_LABELS と未知ラベルは落とす


def load_records(gd: str, files: Sequence[tuple[str, str]] = DATA_FILES) -> list[dict]:
    """4 ファイルを読み、Notebook と同じ整形をして source を付けて返す。"""
    out: list[dict] = []
    for source, fname in files:
        path = os.path.join(gd, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                ents = []
                for item in (rec.get("entities") or []):
                    got = _coerce_entity(item, len(text))
                    if got is not None:
                        ents.append(list(got))
                ents.sort(key=lambda x: (x[0], x[1]))
                seen: set[tuple[int, int, str]] = set()
                dedup = []
                for s, e, lab in ents:
                    if (s, e, lab) in seen:
                        continue
                    seen.add((s, e, lab))
                    dedup.append([s, e, lab])
                out.append({"text": text, "entities": dedup, "source": source})
    return out


def template_signature(rec: dict, loose: bool = False) -> str:
    """エンティティ部分を ``<TYPE>`` に置換した文面。重なるエンティティは先着を優先する。"""
    text = rec["text"]
    parts: list[str] = []
    pos = 0
    for s, e, lab in rec["entities"]:
        if s < pos:
            continue          # 前のエンティティと重なる分は飛ばす
        parts.append(text[pos:s])
        parts.append(f"<{lab}>")
        pos = e
    parts.append(text[pos:])
    sig = "".join(parts)
    if loose:
        sig = _SPACES.sub(" ", _DIGITS.sub("0", sig))
    return sig


def _digest(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in sorted(values):
        h.update(v.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def current_split(records: Sequence[dict], seed: int = SEED,
                  valid_ratio: float = VALID_RATIO) -> tuple[list[dict], list[dict], int]:
    """現行 Notebook の分割を再現する（本文の完全一致だけ重複とみなし、ソースごとに 90/10）。"""
    train: list[dict] = []
    valid: list[dict] = []
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for source in (s for s, _ in DATA_FILES):
        seen: set[str] = set()
        dedup = []
        for r in by_source[source]:
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            dedup.append(r)
        idx = list(range(len(dedup)))
        random.Random(seed).shuffle(idx)
        n_valid = max(1, int(len(dedup) * valid_ratio))
        valid += [dedup[i] for i in idx[:n_valid]]
        train += [dedup[i] for i in idx[n_valid:]]
    train_texts = {r["text"] for r in train}
    removed = sum(1 for r in valid if r["text"] in train_texts)
    valid = [r for r in valid if r["text"] not in train_texts]
    return train, valid, removed


def grouped_split(records: Sequence[dict], *, loose: bool = False, seed: int = SEED,
                  valid_ratio: float = VALID_RATIO) -> tuple[list[dict], list[dict]]:
    """テンプレート署名でグループ化してから 90/10 で分ける。

    グループ化は**ソースを横断して**行う。合成 v4 / v5c / v5e は同じテンプレートを
    共有しているため、ソースごとにグループ化すると同一テンプレートが train（v4 由来）と
    valid（v5c 由来）に分かれてしまう（実測 1,381 件）。本文の完全一致重複も
    ソース横断で落とす。

    グループ単位で振るので valid の文書数はちょうど 10% にはならない。
    件数を合わせるより「同じテンプレートが両側に出ない」ことを優先する。
    小さいグループから valid に詰めることで、目標比率からの乖離を小さくする。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for r in records:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        groups[template_signature(r, loose)].append(r)
    # ソース比率を保つ。グループ内のソース内訳を見て、どのソースも目標を
    # 超えないグループだけ valid に入れる（out_ner はテンプレートが 1 件ずつなので、
    # 大きさ順に詰めると valid が実データばかりになってしまう）
    target: Counter[str] = Counter()
    for r in records:
        target[r["source"]] += 1
    for s in target:
        target[s] = int(target[s] * valid_ratio)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train: list[dict] = []
    valid: list[dict] = []
    got: Counter[str] = Counter()
    for k in keys:
        need = Counter(r["source"] for r in groups[k])
        if all(got[s] + n <= target[s] for s, n in need.items()):
            valid += groups[k]
            got += need
        else:
            train += groups[k]
    return train, valid


def _fmt(n: int, d: int) -> str:
    return f"{n:>6} ({n / max(d, 1) * 100:>5.1f}%)"


def report(records: Sequence[dict], *, loose: bool = False) -> dict:
    """現行分割とテンプレート分割を並べて出す。"""
    train, valid, removed = current_split(records)
    sigs = {template_signature(r, loose) for r in train}
    print("■ 現行の分割（本文の完全一致だけを重複とみなす）")
    print(f"  train {len(train)} / valid {len(valid)}"
          f"（完全一致で valid から除いた分 {removed} 件）")
    print()
    print("■ valid のうち、エンティティを型名に置換した文面が train にもあるもの")
    print(f"  {'valid の由来':<12}{'valid':>8}{'train と重複':>14}{'割合':>8}")
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in valid:
        per[r["source"]][0] += 1
        if template_signature(r, loose) in sigs:
            per[r["source"]][1] += 1
    tot = [0, 0]
    for source, _ in DATA_FILES:
        n, m = per[source]
        tot[0] += n
        tot[1] += m
        print(f"  {source:<12}{n:>8}{m:>14}{m / max(n, 1) * 100:>7.1f}%")
    print(f"  {'全体':<12}{tot[0]:>8}{tot[1]:>14}{tot[1] / max(tot[0], 1) * 100:>7.1f}%")
    clean = [r for r in valid if template_signature(r, loose) not in sigs]
    print(f"\n  → テンプレートが train と重複しない valid: {len(clean)} 件")
    print(f"     内訳: {dict(Counter(r['source'] for r in clean))}")
    gt, gv = grouped_split(records, loose=loose)
    print()
    print("■ テンプレート単位で分割し直した場合")
    print(f"  train {len(gt)} / valid {len(gv)}")
    gts = {template_signature(r, loose) for r in gt}
    leak = sum(1 for r in gv if template_signature(r, loose) in gts)
    print(f"  valid のうち train とテンプレートが重複するもの: {leak} 件（0 が期待値）")
    print(f"  valid の内訳: {dict(Counter(r['source'] for r in gv))}")
    ent_t = Counter(l for r in gt for *_, l in r["entities"])
    ent_v = Counter(l for r in gv for *_, l in r["entities"])
    print(f"\n  {'型':<10}{'train':>16}{'valid':>16}")
    for lab in BASE_LABELS:
        print(f"  {lab:<10}{_fmt(ent_t[lab], sum(ent_t.values())):>16}"
              f"{_fmt(ent_v[lab], sum(ent_v.values())):>16}")
    return {
        "current": {"train": len(train), "valid": len(valid), "exact_removed": removed,
                    "template_overlap": tot[1], "template_overlap_ratio": tot[1] / max(tot[0], 1)},
        "template_disjoint_subset": len(clean),
        "grouped": {"train": len(gt), "valid": len(gv), "leak": leak,
                    "train_text_digest": _digest(r["text"] for r in gt),
                    "valid_text_digest": _digest(r["text"] for r in gv)},
    }


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"text": r["text"], "entities": r["entities"],
                                "source": r["source"]}, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None, help="既定は _paths.drive_dir()")
    ap.add_argument("--out-dir", default=None,
                    help="指定するとテンプレート分割の train/valid と、"
                         "現行 valid のテンプレート重複なし部分を書き出す")
    ap.add_argument("--loose", action="store_true",
                    help="数字・空白を潰した署名でグループ化する")
    ap.add_argument("--report", action="store_true", help="集計だけ表示する（既定）")
    args = ap.parse_args(argv)

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書 / "
          f"{sum(len(r['entities']) for r in records)} エンティティ\n")
    summary = report(records, loose=args.loose)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        gt, gv = grouped_split(records, loose=args.loose)
        write_jsonl(os.path.join(args.out_dir, "train_template_split.jsonl"), gt)
        write_jsonl(os.path.join(args.out_dir, "valid_template_split.jsonl"), gv)
        train, valid, _ = current_split(records)
        sigs = {template_signature(r, args.loose) for r in train}
        clean = [r for r in valid if template_signature(r, args.loose) not in sigs]
        write_jsonl(os.path.join(args.out_dir, "valid_current_template_disjoint.jsonl"), clean)
        with open(os.path.join(args.out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
            json.dump({**summary, "loose": args.loose, "seed": SEED,
                       "valid_ratio": VALID_RATIO,
                       "files": [f for _, f in DATA_FILES]}, f, ensure_ascii=False, indent=2)
        print(f"\n書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

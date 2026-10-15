"""UD Japanese GSD（megagonlabs、固有表現付き）の学習・開発分割を本ライブラリの学習形式に変換する。

- 文の復元はトークンの表層を SpaceAfter=No に従って結合。NE タグは **BILOU**（B/I/L/U）を読む（2026-09-23 に B/I だけ読んでいた不具合を修正）
- ラベル写像は評価（外部 GSD テスト）と同じ: FAC→FACILITY, GPE/LOC→LOCATION, ORG→ORG, PERSON→PERSON
- 写像しない型（NORP・DATE・EVENT・PRODUCT・…）は `neutral` に入れ、学習では損失を計算しない
- 評価セット・盲検セットとの本文完全一致（文単位）は機械照合で落とす

usage:
  PYTHONPATH=. .venv/bin/python tools/gsd_train_to_ner.py [--cache-dir <データフォルダ>/gsd_train/raw] [--out <データフォルダ>/gsd_train/gsd_train_ner.jsonl]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from negatives_v8 import file_sha256, write_jsonl  # noqa: E402

GSD_REVISION = "daca379d207673af653cfc317cbc1d54207b4e3b"
GSD_URL = "https://raw.githubusercontent.com/megagonlabs/UD_Japanese-GSD/{rev}/spacy/ja_gsd-ud-{split}.ne.conllu"
SPLITS: tuple[str, ...] = ("train", "dev")
LABEL_MAP: dict[str, str] = {"FAC": "FACILITY", "GPE": "LOCATION", "LOC": "LOCATION", "ORG": "ORG", "PERSON": "PERSON"}
SOURCE_NAME = "gsd_train"
OUT_DIR_NAME = "gsd_train"
OUT_FILE = "gsd_train_ner.jsonl"


@dataclass
class Sentence:
    sent_id: str
    text: str
    entities: list[list] = field(default_factory=list)
    neutral: list[list] = field(default_factory=list)


def _close(span, out_ents, out_neutral):
    if span is None:
        return
    s, e, t = span
    if e <= s:
        return
    if t in LABEL_MAP:
        out_ents.append([s, e, LABEL_MAP[t]])
    else:
        out_neutral.append([s, e])


def parse_conllu(text: str) -> tuple[list[Sentence], dict]:
    sents: list[Sentence] = []
    stats: Counter = Counter({"sentences": 0, "text_mismatch": 0, "neutral_spans": 0})
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        sent_id, gold_text = "", None
        tokens: list[tuple[str, bool, str | None]] = []       # (form, space_after, ne_tag)
        for line in block.splitlines():
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                gold_text = line.split("=", 1)[1].strip()
            elif line.startswith("#") or not line.strip():
                continue
            else:
                cols = line.split("\t")
                if len(cols) < 10 or "-" in cols[0] or "." in cols[0]:
                    continue                                    # 範囲行・空ノードは飛ばす
                misc = cols[9]
                space_after = "SpaceAfter=No" not in misc
                ne = None
                for kv in misc.split("|"):
                    if kv.startswith("NE="):
                        ne = kv[3:]
                tokens.append((cols[1], space_after, ne))
        if not tokens:
            continue
        text_out = ""
        ents: list[list] = []
        neutral: list[list] = []
        cur: tuple[int, int, str] | None = None
        for form, space_after, ne in tokens:
            start = len(text_out)
            text_out += form
            end = len(text_out)
            prefix, _, typ = (ne or "").partition("-")
            if ne:
                stats[f"tag_{prefix}"] += 1
            # GSD の NE は BILOU（B 先頭 / I 途中 / L 末尾 / U 単独）。B・I だけ読むと U の実体（佐藤氏 の 佐藤、日本）が
            # 丸ごと落ち、L を落とすと末尾が欠ける（藤本博史 → 藤本）。O・未知の接頭辞は実体の外
            if prefix == "B":
                _close(cur, ents, neutral); cur = (start, end, typ)
            elif prefix == "U":
                _close(cur, ents, neutral); _close((start, end, typ), ents, neutral); cur = None
            elif prefix in ("I", "L") and cur is not None and cur[2] == typ:
                cur = (cur[0], end, typ)
                if prefix == "L":
                    _close(cur, ents, neutral); cur = None
            elif prefix in ("I", "L"):                       # 先頭が無い続き（データの崩れ）。単独で切り出す
                _close(cur, ents, neutral); cur = (start, end, typ)
                if prefix == "L":
                    _close(cur, ents, neutral); cur = None
            else:
                _close(cur, ents, neutral); cur = None
            if space_after:
                text_out += " "
        _close(cur, ents, neutral)
        text_out = text_out.rstrip()
        if gold_text is not None and text_out != gold_text:
            stats["text_mismatch"] += 1
            continue
        stats["sentences"] += 1
        for _, _, lab in ents:
            stats[f"label_{lab}"] += 1
        stats["neutral_spans"] += len(neutral)
        sents.append(Sentence(sent_id, text_out, ents, neutral))
    return sents, dict(stats)


def to_records(sentences: Iterable[Sentence], split: str) -> list[dict]:
    return [{"id": s.sent_id, "text": s.text, "entities": s.entities, "neutral": s.neutral,
             "source": SOURCE_NAME, "variant": f"gsd/{split}"} for s in sentences]


def load_eval_texts(extra_paths: Sequence[str] = ()) -> set[str]:
    """評価セットの本文（文書全体と、句点で切った文）。盲検セットは本文を見ずに機械照合だけに使う。"""
    import eval_ood_f1 as f1
    texts: set[str] = set()
    docs: list[str] = []
    for name, (ds, _) in f1.load_sets([]).items():
        docs += [d["text"] for d in ds]
    for p in extra_paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                docs += [json.loads(l)["text"] for l in f if l.strip()]
    for d in docs:
        texts.add(d.strip())
        for sent in re.split(r"(?<=。)", d):
            if sent.strip():
                texts.add(sent.strip())
    return texts


def drop_overlaps(records: Sequence[dict], eval_texts: set[str]) -> tuple[list[dict], int]:
    kept = [r for r in records if r["text"].strip() not in eval_texts]
    return kept, len(records) - len(kept)


def fetch(split: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"ja_gsd-ud-{split}.ne.conllu")
    if not os.path.exists(path):
        with urllib.request.urlopen(GSD_URL.format(rev=GSD_REVISION, split=split), timeout=120) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    return path


def convert(cache_dir: str, out_path: str, splits: Sequence[str] = SPLITS, extra_eval_paths: Sequence[str] = ()) -> dict:
    records: list[dict] = []
    manifest: dict = {"revision": GSD_REVISION, "label_map": LABEL_MAP, "splits": {}, "created_at": dt.datetime.now().isoformat(timespec="seconds")}
    for split in splits:
        path = fetch(split, cache_dir)
        with open(path, encoding="utf-8") as f:
            sents, stats = parse_conllu(f.read())
        manifest["splits"][split] = {"raw_file": path, "raw_sha256": file_sha256(path), **stats}
        records += to_records(sents, split)
    eval_texts = load_eval_texts(extra_eval_paths)
    records, dropped = drop_overlaps(records, eval_texts)
    manifest["dropped_overlap_with_eval"] = dropped
    manifest["records"] = len(records)
    manifest["labels"] = dict(Counter(l for r in records for _, _, l in r["entities"]))
    manifest["neutral_spans"] = sum(len(r["neutral"]) for r in records)
    write_jsonl(out_path, records)
    manifest["out_file"] = out_path
    manifest["out_sha256"] = file_sha256(out_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    gd = drive_dir()
    cache = args.cache_dir or os.path.join(gd, OUT_DIR_NAME, "raw")
    out = args.out or os.path.join(gd, OUT_DIR_NAME, OUT_FILE)
    blind2 = os.path.join(gd, "blind_eval_v2", "blind_eval_v2.jsonl")
    m = convert(cache, out, extra_eval_paths=(blind2,))
    with open(out.replace(".jsonl", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: m[k] for k in ("records", "labels", "neutral_spans", "dropped_overlap_with_eval")}, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

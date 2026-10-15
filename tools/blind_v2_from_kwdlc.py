"""盲検評価セット v2 — 京都大学ウェブ文書リードコーパス（KWDLC）から作る。

KWDLC（https://github.com/ku-nlp/KWDLC、研究目的で公開）は Web 文書の冒頭 3 文に、IREX 準拠の固有表現が
人手で付いている。本プロジェクトの学習・開発には一度も使っていない実テキストなので、最終判定用の盲検セットにする
（`docs/blind_eval_protocol.md` の運用: 評価は判定のためだけに使い、失敗例を Refiner 規則や学習データの作成に直接使わない）。

KNP 形式の読み方:
  - `# S-ID:<doc>-<n>` が 1 文。`# S-ID:<doc>-<n>-01` は括弧を除いた本文、`-02` `-03` は括弧の中身で、ヘッダの `括弧位置`（本文中の
    文字 index）に `括弧始` + 中身 + `括弧終` を戻すと元の文になる。
  - `+ ` 行（基本句）の素性 `<NE:TYPE:表層>` は、その基本句の末尾で終わる固有表現。表層の長さから開始位置を求める。
  - 括弧の位置をまたぐ固有表現（`１９９３年（平成５年）９月６日` の `１９９３年９月６日`）は、戻すと不連続になるので 2 つに割る。

ラベルの写像（IREX → 本ライブラリ）:
  PERSON → PERSON / LOCATION → LOCATION（直後に番地が続くものは番地まで含めて ADDRESS）/ ORGANIZATION → ORG /
  ARTIFACT・OPTIONAL → NEUTRAL（採点対象外）/ DATE・TIME・MONEY・PERCENT → 付けない（本ライブラリの NER の対象外）。
  IREX は学校・病院・駅などを LOCATION か ORGANIZATION に入れるので、本ライブラリの FACILITY とは型が合わない。安全性の指標
  （被覆ゼロ・危険・過剰マスク）は型を問わないので影響しない。型別 F1 はこのセットでは参考値。

注意: KWDLC はブログ等の実テキストで、私人の名が含まれうる。データフォルダの中だけで扱い、会話やレポートに PERSON の表層を出さない。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/blind_v2_from_kwdlc.py --kwdlc <clone dir> --out <データフォルダ>/blind_eval_v2 [--split all|test|dev|train]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from typing import Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402

NE_RE = re.compile(r"<NE:([A-Z_]+):([^>]*)>")
HDR_RE = re.compile(r"# S-ID:(\S+)(.*)")
LABEL_MAP = {"PERSON": "PERSON", "LOCATION": "LOCATION", "ORGANIZATION": "ORG"}
NEUTRAL_TYPES = {"ARTIFACT", "OPTIONAL"}
DROP_TYPES = {"DATE", "TIME", "MONEY", "PERCENT"}
# LOCATION の直後の番地: ６−２１−６ / ３丁目４番５号 / １２番地 / １０１号室 など
_BANCHI_TAIL_RE = re.compile(
    r"^(?:[0-9０-９]+(?:丁目|番地|番|号室|号|階|F|Ｆ)?(?:[-−－‐‑–ー][0-9０-９]+)*(?:丁目|番地|番|号室|号|階)?)+")


def parse_knp(path: str) -> list[dict]:
    """文ごとに {sid, attrs, text, nes:[(type, surface, start, end)]} を返す。"""
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    sents: list[dict] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("# S-ID:"):
            i += 1
            continue
        m = HDR_RE.match(lines[i])
        attrs = dict(re.findall(r"(\S+?):(\S+)", m.group(2)))
        sid, text, nes, pending = m.group(1), "", [], []
        i += 1

        def flush() -> None:
            for typ, surf in pending:
                e = len(text)
                s = e - len(surf)
                if s < 0 or text[s:e] != surf:
                    s2 = text.rfind(surf)
                    s, e = (s2, s2 + len(surf)) if s2 >= 0 else (-1, -1)
                nes.append((typ, surf, s, e))

        while i < len(lines) and lines[i] != "EOS":
            l = lines[i]
            if l.startswith("+ "):
                flush()
                pending = NE_RE.findall(l)
            elif not (l.startswith("* ") or l.startswith("#")):
                text += l.split(" ")[0]
            i += 1
        flush()
        sents.append({"sid": sid, "attrs": attrs, "text": text, "nes": nes})
        i += 1
    return sents


def rebuild_document(sents: Sequence[dict]) -> tuple[str, list[tuple[str, str, int, int]], list[str]]:
    """括弧削除で分かれた文を戻して連結する。戻り値: (本文, NE 一覧, 警告)。"""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for s in sents:
        base = re.sub(r"-\d\d$", "", s["sid"]) if re.search(r"-\d+-\d\d$", s["sid"]) else s["sid"]
        groups.setdefault(base, []).append(s)
    text, nes, warns = "", [], []
    for base, parts in groups.items():
        off = len(text)
        if len(parts) == 1:
            st = parts[0]
            text += st["text"]
            for t, sf, s, e in st["nes"]:
                if s < 0:
                    warns.append(f"{st['sid']}: NE 位置不明 {t}")
                    continue
                nes.append((t, sf, s + off, e + off))
            continue
        main, subs = parts[0], parts[1:]
        inserts = []
        for sub in subs:
            a = sub["attrs"]
            pos = int(a.get("括弧位置", len(main["text"])))
            inserts.append((pos, a.get("括弧始", "（"), sub["text"], a.get("括弧終", "）"), sub))
        inserts.sort(key=lambda x: x[0])
        # 本文の各位置 → 復元後の位置（挿入分だけ後ろへずれる）
        def shift(pos_: int, end: bool = False) -> int:
            add = 0
            for pos, op, body, cl, _ in inserts:
                if pos < pos_ or (pos == pos_ and not end):
                    add += len(op) + len(body) + len(cl)
            return pos_ + add
        new = ""
        cursor = 0
        sub_spans = []
        for pos, op, body, cl, sub in inserts:
            new += main["text"][cursor:pos]
            sub_off = len(new) + len(op)
            new += op + body + cl
            cursor = pos
            for t, sf, s, e in sub["nes"]:
                if s < 0:
                    warns.append(f"{sub['sid']}: NE 位置不明 {t}")
                    continue
                sub_spans.append((t, sf, s + sub_off, e + sub_off))
        new += main["text"][cursor:]
        for t, sf, s, e in main["nes"]:
            if s < 0:
                warns.append(f"{main['sid']}: NE 位置不明 {t}")
                continue
            crossing = [ins for ins in inserts if s < ins[0] < e]
            if not crossing:
                nes.append((t, sf, shift(s) + off, shift(e, end=True) + off))
                continue
            # 括弧をまたぐ固有表現は 2 つ以上に割る（表層は割れた部分）
            cuts = [s] + [ins[0] for ins in crossing] + [e]
            for a, b in zip(cuts, cuts[1:]):
                if b > a:
                    ns, ne = shift(a) + off, shift(b, end=True) + off
                    nes.append((t, None, ns, ne))
            warns.append(f"{main['sid']}: 括弧をまたぐ NE を分割 {t}")
        text += new
        nes += [(t, sf, s + off, e + off) for t, sf, s, e in sub_spans]
    # 表層の検証（分割したものは None）
    checked = []
    for t, sf, s, e in nes:
        if sf is not None and text[s:e] != sf:
            warns.append(f"{base}: 表層不一致 {t}")
            continue
        checked.append((t, text[s:e], s, e))
    return text, checked, warns


def map_entities(text: str, nes: Sequence[tuple[str, str, int, int]]) -> tuple[list[list], list[list], Counter]:
    """IREX → 本ライブラリの型。LOCATION + 番地 → ADDRESS。ARTIFACT / OPTIONAL → NEUTRAL。"""
    ents: list[list] = []
    neutral: list[list] = []
    c: Counter = Counter()
    for t, sf, s, e in nes:
        if t in DROP_TYPES:
            c["dropped_" + t] += 1
            continue
        if t in NEUTRAL_TYPES:
            neutral.append([s, e])
            c["neutral_" + t] += 1
            continue
        lab = LABEL_MAP.get(t)
        if lab is None:
            c["unknown_" + t] += 1
            continue
        if lab == "LOCATION":
            m = _BANCHI_TAIL_RE.match(text[e:])
            if m and re.search(r"[0-9０-９]", m.group(0)):
                e = e + len(m.group(0))
                lab = "ADDRESS"
                c["location_to_address"] += 1
        ents.append([s, e, lab])
        c[lab] += 1
    # 重なりの解消（IREX は入れ子なし。念のため長い方を残す）
    ents.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[list] = []
    for s, e, lab in ents:
        if out and s < out[-1][1]:
            c["overlap_dropped"] += 1
            continue
        out.append([s, e, lab])
    return out, neutral, c


def load_ids(kwdlc: str, split: str) -> list[str]:
    if split == "all":
        return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(kwdlc, "knp", "*", "*.knp")))
    path = os.path.join(kwdlc, "id", "split_for_pas", f"{split}.id")
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def build(kwdlc: str, split: str) -> tuple[list[dict], Counter, list[str]]:
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(kwdlc, "knp", "*", "*.knp"))}
    split_of: dict[str, str] = {}
    for name in ("train", "dev", "test"):
        p = os.path.join(kwdlc, "id", "split_for_pas", f"{name}.id")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    split_of[l.strip()] = name
    docs, counts, warns = [], Counter(), []
    for did in load_ids(kwdlc, split):
        text, nes, w = rebuild_document(parse_knp(files[did]))
        warns += w
        ents, neutral, c = map_entities(text, nes)
        counts.update(c)
        counts["docs"] += 1
        counts["chars"] += len(text)
        docs.append({"id": did, "text": text, "entities": ents, "neutral": neutral,
                     "source": "kwdlc", "split": split_of.get(did, "?")})
    return docs, counts, warns


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kwdlc", required=True, help="git clone した KWDLC のディレクトリ")
    ap.add_argument("--split", default="all", choices=("all", "train", "dev", "test"))
    ap.add_argument("--out", default=None, help="既定: <データフォルダ>/blind_eval_v2")
    ap.add_argument("--name", default="blind_eval_v2")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(drive_dir(), "blind_eval_v2")
    os.makedirs(out, exist_ok=True)
    docs, counts, warns = build(args.kwdlc, args.split)
    for d in docs:
        for s, e, lab in d["entities"]:
            assert 0 <= s < e <= len(d["text"]), (d["id"], s, e)
    path = os.path.join(out, f"{args.name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    try:
        commit = subprocess.check_output(["git", "-C", args.kwdlc, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    manifest = {
        "name": args.name, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"corpus": "KWDLC (Kyoto University Web Document Leads Corpus)", "url": "https://github.com/ku-nlp/KWDLC",
                   "commit": commit, "split": args.split, "terms": "released for the purpose of contributing to the research of NLP"},
        "file": os.path.basename(path), "sha256": sha256_file(path),
        "counts": dict(counts), "warnings": len(warns), "warning_examples": warns[:20],
        "label_map": {**LABEL_MAP, "LOCATION+番地": "ADDRESS", "ARTIFACT/OPTIONAL": "NEUTRAL", "DATE/TIME/MONEY/PERCENT": "(付けない)"},
        "notes": ["学習・開発に未使用の実テキスト。評価は判定のためだけに使い、失敗例を規則や学習データに直接使わない",
                  "私人の名を含みうる。データフォルダの外へ出さず、レポートに PERSON の表層を出さない",
                  "IREX は施設を LOCATION / ORGANIZATION に入れるので FACILITY の型別数値は参考値"],
    }
    with open(os.path.join(out, f"{args.name}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "warning_examples"}, ensure_ascii=False, indent=2))
    print("warnings:", len(warns), warns[:8])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

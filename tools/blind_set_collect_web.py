"""盲検評価セット用に、公開されている実文書を Web から採取して JSONL にする（再現可能: 固定シード）。

採取元（いずれも公開データ。利用条件は出典ごとに異なり、livedoor ニュースコーパスは CC BY-ND のため改変物の再配布ができない。
本ツールの用途は評価・解析に限り、採取した本文や派生テキスト（盲検セット）は再配布しない。PII を含む私文書は使わない）:
  kokkai   国会会議録検索システム API（国立国会図書館）。発言 = 議事録ジャンル。話者は公人
  livedoor livedoor ニュースコーパス（ldcc-20140209、CC BY-ND 2.1 JP、株式会社ロンウイット配布）。9 カテゴリの記事冒頭。
           評価・解析のみに使い、抜粋・加工したテキストは配布しない（ND 条項）
  aozora   青空文庫（globis-university/aozorabunko-clean、CC BY 4.0、本文は主にパブリックドメイン）。新字新仮名の会話を含む一節
  jagovfaq 官公庁 FAQ（matsuxr/JaGovFaqs-22k、CC BY 4.0）。PII をほぼ含まない負例（官公庁名・電話番号は含む）

usage:
  PYTHONPATH=. .venv/bin/python tools/blind_set_collect_web.py --out <dir> [--cache <dir>] [--seed 42]
出力: <dir>/web_corpus.jsonl  {id, text, genre, meta:{source, url, license, ...}}
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "privacy-masker-blind-eval/0.1 (research; contact: repository maintainer)"}
LIVEDOOR_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
AOZORA_URL = "https://huggingface.co/datasets/globis-university/aozorabunko-clean/resolve/main/aozorabunko-dedupe-clean.jsonl.gz"
JAGOV_URL = "https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k/resolve/main/data.jsonl"
KOKKAI_API = "https://kokkai.ndl.go.jp/api/speech"

_SENT_END = re.compile(r"[。！？!?」』]")


def _download(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); total += len(chunk)
    os.replace(dest + ".part", dest)
    print(f"downloaded {url} -> {dest} ({total/1e6:.1f} MB)", file=sys.stderr)
    return dest


def _cut(text: str, max_chars: int) -> str:
    """max_chars 付近の文末で切る（文の途中で切らない）。"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    ends = [m.end() for m in _SENT_END.finditer(head)]
    return head[:ends[-1]] if ends and ends[-1] > max_chars * 0.5 else head


# ---------------------------------------------------------------- 国会会議録

def collect_kokkai(rng: random.Random, n: int) -> list[dict]:
    keywords = ["市民", "地域", "病院", "学校", "災害", "農業", "予算", "企業", "観光", "介護", "住民", "港"]
    windows = [(f"{y}-01-01", f"{y}-12-31") for y in range(2003, 2025)]
    pool: dict[str, dict] = {}
    plan = [(kw, w) for kw in keywords for w in windows]
    rng.shuffle(plan)
    for kw, (frm, until) in plan[:36]:
        q = urllib.parse.urlencode({"any": kw, "from": frm, "until": until, "maximumRecords": 30,
                                    "recordPacking": "json", "startRecord": rng.randint(1, 200)})
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{KOKKAI_API}?{q}", headers=UA), timeout=60) as r:
                data = json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"kokkai: {kw} {frm}: {e}", file=sys.stderr); continue
        for rec in data.get("speechRecord", []):
            sp = rec.get("speech", "")
            if rec.get("speaker") in (None, "会議録情報") or not (300 <= len(sp) <= 1500):
                continue
            pool[rec["speechID"]] = rec
        time.sleep(0.4)
    recs = sorted(pool.values(), key=lambda r: r["speechID"])
    rng.shuffle(recs)
    out = []
    for rec in recs[:n]:
        out.append({"text": _cut(rec["speech"], 700), "genre": "web_kokkai",
                    "meta": {"source": "kokkai.ndl.go.jp", "url": rec.get("speechURL"), "license": "public record (NDL API)",
                             "date": rec.get("date"), "meeting": rec.get("nameOfMeeting"), "house": rec.get("nameOfHouse")}})
    return out


# ---------------------------------------------------------------- livedoor

def collect_livedoor(rng: random.Random, n: int, cache: str) -> list[dict]:
    path = _download(LIVEDOOR_URL, os.path.join(cache, "ldcc-20140209.tar.gz"))
    by_cat: dict[str, list[tuple[str, str]]] = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 3 or parts[2] in ("LICENSE.txt",) or not parts[2].endswith(".txt"):
                continue
            body = tf.extractfile(m).read().decode("utf-8", errors="replace")
            by_cat.setdefault(parts[1], []).append((m.name, body))
    per = max(1, n // len(by_cat))
    out = []
    for cat in sorted(by_cat):
        files = sorted(by_cat[cat]); rng.shuffle(files)
        for name, body in files[:per]:
            lines = body.split("\n")
            url, date, title, rest = lines[0].strip(), lines[1].strip(), lines[2].strip(), "\n".join(lines[3:])
            rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
            out.append({"text": f"{title}\n\n{_cut(rest, 550)}", "genre": "web_livedoor",
                        "meta": {"source": "livedoor news corpus (ldcc-20140209)", "url": url, "license": "CC BY-ND 2.1 JP",
                                 "category": cat, "date": date}})
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------- 青空文庫

def collect_aozora(rng: random.Random, n: int, cache: str) -> list[dict]:
    path = _download(AOZORA_URL, os.path.join(cache, "aozorabunko-dedupe-clean.jsonl.gz"))
    chosen: list[dict] = []
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:  # リザーバサンプリング（新字新仮名・会話あり・十分な長さ）
            row = json.loads(line)
            meta = row.get("meta", {})
            text = row.get("text", "")
            if meta.get("文字遣い種別") != "新字新仮名" or len(text) < 3000 or "「" not in text:
                continue
            seen += 1
            if len(chosen) < n:
                chosen.append(row)
            else:
                j = rng.randrange(seen)
                if j < n:
                    chosen[j] = row
    out = []
    for row in chosen:
        text, meta = row["text"], row.get("meta", {})
        starts = [m.start() for m in re.finditer("「", text)]
        pos = max(0, rng.choice(starts) - rng.randint(60, 200))
        # 文頭に揃える
        nl = text.rfind("\n", 0, pos)
        pos = nl + 1 if nl >= 0 and pos - nl < 200 else pos
        out.append({"text": _cut(text[pos:pos + 900], 450), "genre": "web_aozora",
                    "meta": {"source": "aozorabunko-clean (globis-university)", "url": meta.get("図書カードURL"),
                             "license": "CC BY 4.0 (dataset) / public domain text", "title": meta.get("作品名"),
                             "author": meta.get("姓", "") + meta.get("名", "")}})
    return out


# ---------------------------------------------------------------- 官公庁 FAQ

def collect_jagovfaq(rng: random.Random, n: int, cache: str) -> list[dict]:
    path = _download(JAGOV_URL, os.path.join(cache, "jagovfaqs_data.jsonl"))
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if 80 <= len(r.get("Answer", "")) <= 900]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        out.append({"text": f"Q. {r['Question'].strip()}\nA. {_cut(r['Answer'], 600)}", "genre": "web_jagovfaq",
                    "meta": {"source": "JaGovFaqs-22k (matsuxr)", "url": r.get("url"), "license": "CC BY 4.0",
                             "copyright": r.get("copyright")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="ダウンロードのキャッシュ（既定: <out>/raw）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-kokkai", type=int, default=30)
    ap.add_argument("--n-livedoor", type=int, default=36)
    ap.add_argument("--n-aozora", type=int, default=15)
    ap.add_argument("--n-jagovfaq", type=int, default=15)
    args = ap.parse_args()
    cache = args.cache or os.path.join(args.out, "raw")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    docs: list[dict] = []
    docs += collect_kokkai(random.Random(args.seed + 1), args.n_kokkai)
    print(f"kokkai: {sum(d['genre']=='web_kokkai' for d in docs)}", file=sys.stderr)
    docs += collect_livedoor(random.Random(args.seed + 2), args.n_livedoor, cache)
    docs += collect_aozora(random.Random(args.seed + 3), args.n_aozora, cache)
    docs += collect_jagovfaq(random.Random(args.seed + 4), args.n_jagovfaq, cache)
    counters: dict[str, int] = {}
    with open(os.path.join(args.out, "web_corpus.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            g = d["genre"].replace("web_", "")
            counters[g] = counters.get(g, 0) + 1
            d = {"id": f"web-{g}-{counters[g]:03d}", **d}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"web_corpus.jsonl: {len(docs)} docs {counters}", file=sys.stderr)
    del rng


if __name__ == "__main__":
    main()

"""日本郵便 CSV から都道府県・市区町村の重複なし一覧を作る。

既定は公式 UTF-8 CSV（ヘッダーなし、7・8列目）。列番号は1始まり。
例: --src input.csv --pref-column 1 --city-column 2 --header
    --acquired-date YYYY-MM-DD --source-sha256 SHA256
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re

SOURCE_URL = "https://www.post.japanpost.jp/service/search/zipcode/download/utf/zip/utf_ken_all.zip"
COPYRIGHT = "日本郵便は郵便番号データについて著作権を主張していない。"


def build(src, out, *, pref_column=7, city_column=8, header=False,
          acquired_date, source_sha256, source_url=SOURCE_URL):
    if min(pref_column, city_column) < 1 or pref_column == city_column:
        raise ValueError("columns must be distinct positive 1-based numbers")
    if date.fromisoformat(acquired_date).isoformat() != acquired_date:
        raise ValueError("acquired_date must be YYYY-MM-DD")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA256 digest")
    pairs = set()
    with Path(src).open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        if header:
            next(rows, None)
        for row in rows:
            if len(row) < max(pref_column, city_column):
                raise ValueError("CSV row has too few columns")
            pair = (row[pref_column - 1].strip(), row[city_column - 1].strip())
            if any(not s or any(c in s for c in "\t\r\n") for s in pair):
                raise ValueError("invalid prefecture or municipality")
            if not pair[0].endswith(tuple("都道府県")) or not pair[1].endswith(tuple("市区町村")):
                raise ValueError("invalid prefecture or municipality suffix")
            pairs.add(pair)
    if not pairs:
        raise ValueError("empty municipality CSV")
    payload = "".join(p + "\t" + c + "\n" for p, c in sorted(pairs)).encode("utf-8")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    manifest = {"file": out.name, "entries": len(pairs), "sha256": hashlib.sha256(payload).hexdigest(),
                "format": "UTF-8 TSV: prefecture<TAB>city, sorted, unique, no header",
                "source": "日本郵便 郵便番号データ", "source_url": source_url,
                "acquired_date": acquired_date, "source_sha256": source_sha256,
                "copyright": COPYRIGHT, "generator": "tools/build_municipalities.py"}
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="privacy_masker/data/ja_municipalities.tsv")
    ap.add_argument("--pref-column", type=int, default=7)
    ap.add_argument("--city-column", type=int, default=8)
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--acquired-date", required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--source-url", default=SOURCE_URL)
    args = vars(ap.parse_args(argv))
    try:
        manifest = build(**args)
    except (ValueError, OSError, csv.Error) as exc:
        ap.error(str(exc))
    print(json.dumps({k: manifest[k] for k in ("entries", "sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

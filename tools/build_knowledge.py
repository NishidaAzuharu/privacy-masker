"""公知データ（PublicKnowledge .pmk）をビルドする。

入力: Wikipedia 日本語版 ns0 見出し語（1行1件、.gz可。データフォルダの jawiki_titles_ns0.txt.gz）
出力: 64bit ハッシュのソート済み配列（.pmk）。IdentifiabilityAnnotator(knowledge=PublicKnowledge.load(path))

実行（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/build_knowledge.py [--src PATH] [--out PATH]

見出し語の取得元（再ビルド時）:
  https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-all-titles-in-ns0.gz
"""
from __future__ import annotations

import argparse
import os
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
import sys
import time

sys.path.insert(0, os.getcwd())
from privacy_masker.annotators import PublicKnowledge  # noqa: E402

GD = drive_dir()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{GD}/jawiki_titles_ns0.txt.gz")
    ap.add_argument("--out", default=f"{GD}/jawiki_titles_ns0.pmk")
    args = ap.parse_args()

    t0 = time.perf_counter()
    k = PublicKnowledge.from_titles_file(args.src)
    t1 = time.perf_counter()
    k.save(args.out)
    t2 = time.perf_counter()
    loaded = PublicKnowledge.load(args.out)
    t3 = time.perf_counter()
    size = os.path.getsize(args.out)
    print(f"entries : {len(loaded):,}  (build {t1 - t0:.1f}s, save {t2 - t1:.1f}s, load {t3 - t2:.2f}s)")
    print(f"file    : {args.out}  ({size / 1e6:.1f} MB)")
    probes = ["安倍晋三", "岸田 文雄", "新宿駅", "東京都庁", "トヨタ自動車", "西新宿五丁目駅", "山田太郎", "樋渡アキ", "株式会社カナメ精工"]
    t4 = time.perf_counter()
    hits = {p: (p in loaded) for p in probes}
    t5 = time.perf_counter()
    print(f"lookup  : {(t5 - t4) / len(probes) * 1e6:.1f} us/probe")
    for p, h in hits.items():
        print(f"  {'hit ' if h else 'miss'}  {p}")


if __name__ == "__main__":
    main()

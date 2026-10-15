# tools/build_surrogate_denylist.py
"""同梱の禁止リストを作る（設計 §7 B 第 1 層）。

生成器が出しうる架空の名前（`surrogate_generators.enumerate_candidates()`）を Wikipedia 見出し語の PublicKnowledge
（`jawiki_titles_ns0.pmk`、282 万件）と照合し、**衝突した名前だけ**を PublicKnowledge 形式（.pmk）で
`privacy_masker/data/surrogate_denylist.pmk` に書く。知識ファイルの無い環境でも「公知の実体を指さない」が成立する。
プールやテンプレートを変えたら作り直す（manifest に候補数・衝突数・知識ファイルの SHA を残す）。

manifest には知識ファイルのフルパス（開発機のパスにメールアドレスを含みうる）は書かない。
ファイル名（basename）だけを "knowledge_file" として残す（このファイルは wheel に同梱され配布されるため）。

usage:
  PYTHONPATH=. .venv/bin/python tools/build_surrogate_denylist.py [--knowledge <データフォルダ>/jawiki_titles_ns0.pmk] [--out privacy_masker/data/surrogate_denylist.pmk]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _paths import drive_dir  # noqa: E402
from privacy_masker.annotators import PublicKnowledge  # noqa: E402
from privacy_masker.operators.surrogate_generators import enumerate_candidates  # noqa: E402

DEFAULT_OUT = os.path.join("privacy_masker", "data", "surrogate_denylist.pmk")


def manifest_path_for(out_path: str) -> str:
    """出力 .pmk のパスから manifest のパスを作る（拡張子を丸ごと置き換える。`.pmk.pmk` のような
    二重拡張子でも最後の拡張子だけを `.manifest.json` にする）。"""
    return os.path.splitext(out_path)[0] + ".manifest.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(knowledge_path: str, out_path: str) -> dict:
    knowledge = PublicKnowledge.load(knowledge_path)
    candidates = set(enumerate_candidates())
    hits = sorted(c for c in candidates if c in knowledge)
    deny = PublicKnowledge.from_iterable(hits)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    deny.save(out_path)
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/build_surrogate_denylist.py",
        # フルパスは開発機のパス（メールアドレスを含みうる）なので書かない。ファイル名だけ残す。
        "knowledge_file": os.path.basename(knowledge_path),
        "knowledge_sha256": _sha256(knowledge_path),
        "knowledge_entries": len(knowledge),
        # 同梱物の帰属表示（wheel に入るので出所とライセンスを残す）
        "knowledge_source": {"name": "Wikipedia (ja) all-titles-in-ns0", "url": "https://dumps.wikimedia.org/jawiki/",
                             "license": "CC BY-SA 4.0", "note": "見出し語との衝突分を 64bit ハッシュにしたもの。見出し語の本文は含まない"},
        "candidates": len(candidates),
        "hits": len(hits),
        "entries_written": len(hits),
        "pmk_entries": len(deny),
        "out_sha256": _sha256(out_path),
        "sample_hits": hits[:30],
    }
    with open(manifest_path_for(out_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    knowledge = args.knowledge or os.path.join(drive_dir(), "jawiki_titles_ns0.pmk")
    m = build(knowledge, args.out)
    print(json.dumps({k: m[k] for k in ("candidates", "hits", "pmk_entries", "knowledge_entries", "sample_hits")}, ensure_ascii=False, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

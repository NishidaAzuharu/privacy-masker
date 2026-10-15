"""盲検評価セットの準備: 対象コーパスから文書を抽出し、本番パイプラインで事前注釈した「レビュー用タグ付きテキスト」を書き出す。

レビュー担当はタグ付きテキスト（<PERSON>…</PERSON> 形式）を直接修正する。offset を手で触らないので安全。
修正後は tools/blind_set_finalize.py で評価用 JSONL に変換し、SHA-256 を固定する。手順は docs/blind_eval_protocol.md。

入力（--input）:
  - JSONL: 1 行 1 文書。"text" 必須、"id" / "genre" 任意
  - .txt: 1 行 1 文書（空行は無視）
  - ディレクトリ: 中の *.txt を 1 ファイル 1 文書として読む

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/blind_set_prepare.py --input corpus.jsonl --n 250 --out blind_set/ [--stratify genre] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from privacy_masker import MorphologicalRefiner, PrivacyEngine  # noqa: E402
from privacy_masker.detectors import CharHeadNerDetector, RegexDetector, TransformersNerDetector  # noqa: E402
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir  # noqa: E402

LABELS = ("PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION", "EMAIL", "PHONE_NUMBER", "POSTAL_CODE",
          "CREDIT_CARD", "IP_ADDRESS", "MY_NUMBER", "DATE")


def read_docs(path: str) -> list[dict]:
    docs: list[dict] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".txt"):
                text = open(os.path.join(path, name), encoding="utf-8").read().strip()
                if text:
                    docs.append({"id": os.path.splitext(name)[0], "text": text})
    elif path.endswith(".jsonl"):
        for i, line in enumerate(open(path, encoding="utf-8")):
            if not line.strip():
                continue
            d = json.loads(line)
            docs.append({"id": str(d.get("id", f"doc-{i:05d}")), "text": d["text"],
                         **({"genre": d["genre"]} if "genre" in d else {})})
    else:
        for i, line in enumerate(open(path, encoding="utf-8")):
            if line.strip():
                docs.append({"id": f"doc-{i:05d}", "text": line.rstrip("\n")})
    return docs


def sample(docs: list[dict], n: int, seed: int, stratify: str | None) -> list[dict]:
    rnd = random.Random(seed)
    if not stratify or not all(stratify in d for d in docs):
        pool = list(docs); rnd.shuffle(pool)
        return sorted(pool[:n], key=lambda d: d["id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        groups[str(d[stratify])].append(d)
    per = max(1, n // len(groups))
    picked: list[dict] = []
    for key in sorted(groups):
        g = list(groups[key]); rnd.shuffle(g)
        picked += g[:per]
    rest = [d for d in docs if d not in picked]; rnd.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    return sorted(picked, key=lambda d: d["id"])


def tagged_text(text: str, ents) -> str:
    out, pos = [], 0
    for e in sorted(ents, key=lambda x: x.start):
        if e.start < pos:
            continue  # 重なりは先勝ち（解決済みのはず）
        out.append(text[pos:e.start])
        out.append(f"<{e.entity_type}>{text[e.start:e.end]}</{e.entity_type}>")
        pos = e.end
    out.append(text[pos:])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", default=None, help="層化に使うフィールド名（JSONL の genre など）")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--model", default=None,
                    help=f"NER モデル（モデルフォルダのパスか Hub ID。既定: Hub の {DEFAULT_MODEL_ID}）")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    docs = read_docs(args.input)
    picked = sample(docs, args.n, args.seed, args.stratify)
    model = resolve_model_dir(args.model or DEFAULT_MODEL_ID)
    ner_cls = CharHeadNerDetector if os.path.exists(os.path.join(model, "char_head_config.json")) else TransformersNerDetector
    engine = PrivacyEngine(
        detectors=[ner_cls(model=model, device=args.device), RegexDetector.default_ja()],
        refiner=MorphologicalRefiner(),
    )
    os.makedirs(args.out, exist_ok=True)
    review_path = os.path.join(args.out, "blind_set.review.txt")
    source_path = os.path.join(args.out, "blind_set.source.jsonl")
    label_counter: Counter = Counter()
    with open(review_path, "w", encoding="utf-8") as rv, open(source_path, "w", encoding="utf-8") as src:
        rv.write("# 盲検評価セット レビュー用（docs/blind_eval_protocol.md の方針でタグを修正してください）\n")
        rv.write("# 記法: <PERSON>田中太郎</PERSON>  <NEUTRAL>…</NEUTRAL> は採点対象外領域。タグの外の本文は変更しないこと。\n")
        rv.write(f"# ラベル: {' '.join(LABELS)}\n\n")
        for d in picked:
            ents = engine.detect(d["text"])
            for e in ents:
                label_counter[e.entity_type] += 1
            rv.write(f"### {d['id']}" + (f" [{d['genre']}]" if "genre" in d else "") + "\n")
            rv.write(tagged_text(d["text"], ents) + "\n\n")
            src.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest = {
        "input": os.path.abspath(args.input), "n_input": len(docs), "n_sampled": len(picked), "seed": args.seed,
        "stratify": args.stratify, "model": model, "pre_annotation_labels": dict(label_counter),
        "source_sha256": hashlib.sha256(open(source_path, "rb").read()).hexdigest(),
        "note": "事前注釈はレビューの下書き。gold は人手修正後の blind_set_finalize.py の出力とする",
    }
    with open(os.path.join(args.out, "blind_set.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"抽出 {len(picked)}/{len(docs)} 文書 → {review_path}")
    print(f"事前注釈のラベル数: {dict(label_counter)}")
    print("次: レビュー担当が review.txt のタグを修正 → tools/blind_set_finalize.py で JSONL 化")


if __name__ == "__main__":
    main()

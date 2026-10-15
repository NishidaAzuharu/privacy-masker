"""盲検評価セットの確定: 人手修正済みのタグ付きテキストを評価用 JSONL に変換し、SHA-256 で凍結する。

検証すること:
  - タグを外した本文が blind_set.source.jsonl の原文と完全一致する（本文を書き換えていない）
  - タグの入れ子・未知ラベル・空タグが無い
  - <NEUTRAL> は entities ではなく neutral（採点対象外領域）に入れる
  - lint（警告、--strict でエラー）: PERSON の末尾が敬称 / タグ端の空白・句読点 / タグの外に電話・メール・郵便番号らしき文字列 /
    同じ文書内で一度タグ付けした表層が別の場所ではタグ無し

usage:
  PYTHONPATH=. .venv/bin/python tools/blind_set_finalize.py --dir blind_set/ [--out blind_set/blind_eval_v1.jsonl]
      [--tagged authored/*.txt]   # 執筆者がタグ付きで書いた文書（原文照合なし。タグを外したものを原文とする）
評価:
  PYTHONPATH=. .venv/bin/python tools/eval_all.py <model_dir または Hub ID> --extra blind=blind_set/blind_eval_v1.jsonl --skip-stress
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

TAG_RE = re.compile(r"<(/?)([A-Z_]+)>")
KNOWN = {"PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION", "EMAIL", "PHONE_NUMBER", "POSTAL_CODE",
         "CREDIT_CARD", "IP_ADDRESS", "MY_NUMBER", "DATE", "NEUTRAL"}


def parse_tagged(tagged: str, doc_id: str) -> tuple[str, list, list]:
    text_parts: list[str] = []
    entities: list = []
    neutral: list = []
    pos = 0
    length = 0
    open_tag: tuple[str, int] | None = None
    for m in TAG_RE.finditer(tagged):
        chunk = tagged[pos:m.start()]
        text_parts.append(chunk); length += len(chunk); pos = m.end()
        closing, label = m.group(1) == "/", m.group(2)
        if label not in KNOWN:
            raise ValueError(f"{doc_id}: 未知のラベル <{label}>")
        if not closing:
            if open_tag is not None:
                raise ValueError(f"{doc_id}: タグの入れ子 <{label}> in <{open_tag[0]}>")
            open_tag = (label, length)
        else:
            if open_tag is None or open_tag[0] != label:
                raise ValueError(f"{doc_id}: 対応しない閉じタグ </{label}>")
            start = open_tag[1]
            if length == start:
                raise ValueError(f"{doc_id}: 空のタグ <{label}>")
            (neutral if label == "NEUTRAL" else entities).append(
                [start, length] if label == "NEUTRAL" else [start, length, label])
            open_tag = None
    if open_tag is not None:
        raise ValueError(f"{doc_id}: 閉じられていないタグ <{open_tag[0]}>")
    text_parts.append(tagged[pos:])
    return "".join(text_parts), entities, neutral


_HONORIFIC_TAIL = re.compile(r"(さん|様|さま|氏|君|くん|殿|先生|ちゃん|夫人|議員|大臣|部長|課長|社長)$")
_EDGE_BAD = re.compile(r"^[\s、。，．,.:：]|[\s、。，．,.:：]$")
_PHONE_LIKE = re.compile(r"(?<![0-9０-９\-－])(?:\+81[\s\-]?|０?0)[0-9０-９]{1,4}[\s\-－()（）]{0,2}[0-9０-９]{2,4}[\s\-－)）]{0,2}[0-9０-９]{3,4}(?![0-9０-９\-－])")
_EMAIL_LIKE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_POSTAL_LIKE = re.compile(r"(?<![0-9０-９])[0-9０-９]{3}[\-－][0-9０-９]{4}(?![0-9０-９])")
_SPAN_TYPES_FOR_REPEAT = {"PERSON", "ORG", "FACILITY", "LOCATION", "ADDRESS", "EMAIL", "PHONE_NUMBER"}


def lint(text: str, entities: list, neutral: list, doc_id: str) -> list[str]:
    """gold の作り方のブレを機械的に拾う（方針: docs/blind_eval_protocol.md）。"""
    warns: list[str] = []
    covered = [False] * len(text)
    for s, e, *_ in entities + neutral:
        for i in range(s, e):
            covered[i] = True
    for s, e, lab in entities:
        surf = text[s:e]
        if lab == "PERSON" and _HONORIFIC_TAIL.search(surf):
            warns.append(f"{doc_id}: PERSON の末尾に敬称/肩書き「{surf}」")
        if _EDGE_BAD.search(surf) and not re.search(r"[A-Za-z]\.$", surf):   # "Ltd." "J. D." は許容
            warns.append(f"{doc_id}: タグ端に空白・句読点「{surf}」({lab})")
        if lab == "PERSON" and ("、" in surf or "，" in surf):
            warns.append(f"{doc_id}: PERSON に読点（列挙をまとめてタグ付け？）「{surf}」")
    untagged = "".join(c if not covered[i] else " " for i, c in enumerate(text))
    for name, pat in (("電話番号", _PHONE_LIKE), ("メール", _EMAIL_LIKE), ("郵便番号", _POSTAL_LIKE)):
        for m in pat.finditer(untagged):
            warns.append(f"{doc_id}: タグ外に{name}らしき文字列「{m.group().strip()}」")
    surfaces = {text[s:e] for s, e, lab in entities if lab in _SPAN_TYPES_FOR_REPEAT and e - s >= 2}
    for surf in sorted(surfaces):
        if surf in untagged:
            warns.append(f"{doc_id}: 「{surf}」はタグ付きとタグ無しが混在")
    return warns


def iter_blocks(review: str):
    for block in re.split(r"^### ", review, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        header = header.rstrip("\r")
        doc_id = header.split(" [")[0].strip()
        genre = header[header.find("[") + 1:header.rfind("]")] if "[" in header else None
        yield doc_id, genre, body.rstrip("\r\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tagged", nargs="*", default=[], help="執筆者がタグ付きで書いた追加文書（原文照合なし）")
    ap.add_argument("--strict", action="store_true", help="lint の警告をエラー扱いにする")
    args = ap.parse_args()
    review_path = os.path.join(args.dir, "blind_set.review.txt")
    source_path = os.path.join(args.dir, "blind_set.source.jsonl")
    # newline="" で \r\n をそのまま読む（国会会議録など原文が CRLF の文書と一致させる）
    review = open(review_path, encoding="utf-8", newline="").read() if os.path.exists(review_path) else ""
    source = ({json.loads(l)["id"]: json.loads(l) for l in open(source_path, encoding="utf-8") if l.strip()}
              if os.path.exists(source_path) else {})
    out_path = args.out or os.path.join(args.dir, "blind_eval_v1.jsonl")
    rows, labels, errors, warns, seen_ids = [], Counter(), [], [], set()
    jobs = [(doc_id, genre, tagged, True) for doc_id, genre, tagged in iter_blocks(review)]
    for path in args.tagged:
        jobs += [(doc_id, genre, tagged, False) for doc_id, genre, tagged in iter_blocks(open(path, encoding="utf-8", newline="").read())]
    for doc_id, genre, tagged, check_source in jobs:
        if doc_id in seen_ids:
            errors.append(f"{doc_id}: id が重複"); continue
        seen_ids.add(doc_id)
        try:
            text, entities, neutral = parse_tagged(tagged, doc_id)
        except ValueError as e:
            errors.append(str(e)); continue
        if check_source:
            if doc_id not in source:
                errors.append(f"{doc_id}: source に無い id"); continue
            # 改行コード（CRLF/LF）の差はレビュー時のエディタ保存で生じるので正規化して照合する。
            # 出力する本文は entity のオフセットを計算した text 自体（原文と改行コード以外は同一）
            if text.replace("\r\n", "\n") != source[doc_id]["text"].replace("\r\n", "\n").rstrip("\n"):
                errors.append(f"{doc_id}: タグを外した本文が原文と一致しない（本文を編集していないか確認）"); continue
            full_text, meta = text, source[doc_id].get("meta")
        else:
            full_text, meta = text, {"source": "authored"}
        if not text.strip():
            errors.append(f"{doc_id}: 本文が空"); continue
        warns += lint(text, entities, neutral, doc_id)
        for _, _, lab in entities:
            labels[lab] += 1
        row = {"id": doc_id, "text": full_text, "entities": entities, "neutral": neutral}
        if genre:
            row["genre"] = genre
        if meta:
            row["meta"] = meta
        rows.append(row)
    if warns:
        print(f"lint 警告 {len(warns)} 件:")
        for w in warns:
            print("  -", w)
        if args.strict:
            errors.append(f"lint 警告 {len(warns)} 件（--strict）")
    if errors:
        print("エラー（修正してから再実行）:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    manifest_path = os.path.splitext(out_path)[0] + ".manifest.json"   # 出力 JSONL の隣に置く
    with open(manifest_path, "w", encoding="utf-8") as f:
        genres = Counter(r.get("genre", "?") for r in rows)
        json.dump({"file": os.path.basename(out_path), "sha256": sha, "documents": len(rows),
                   "genres": dict(sorted(genres.items())),
                   "entities": dict(labels), "neutral_spans": sum(len(r["neutral"]) for r in rows),
                   "policy": "docs/blind_eval_protocol.md",
                   "rule": "凍結後は 1 回だけ評価する。調整に使ったら盲検ではなくなる"}, f, ensure_ascii=False, indent=2)
    print(f"確定: {len(rows)} 文書, entities {dict(labels)} → {out_path}")
    print(f"sha256: {sha}  (manifest: {manifest_path})")


if __name__ == "__main__":
    main()

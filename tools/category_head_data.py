"""分類ヘッド（設計 §8）の学習用スパンを data_v11 から作る。

ラベルの出所は 2 つ: (1) RuleCategorizer が種別語で決めたカテゴリ（実テキスト・合成の両方）、
(2) 合成データの生成時の form（templates.jsonl のスロット順から復元。規則と両方あれば規則を優先）。
種別語を隠す拡張: 規則ラベル付きスパンの mask_rate で、種別語を本文から削った写しを同じラベルで足す。
削った後も規則が別の種別語を検出できてしまう場合（例:「ネットワークス商船」→ type_word_span が指す
末尾の「商船」(transport) を削った残り「ネットワークス」がなお it と判定できる）は「隠しきれていない」
とみなし、そのスパンは写しを作らない。

v2（`--labels`）: `--labels` を渡すと、`tools/category_labels.py filter` の `category_labels_v1.jsonl`
を供給源にし、`split`（head_train / head_dev / head_judge）ごとに `spans_train.jsonl` /
`spans_dev.jsonl` / `spans_judge.jsonl` を書く。Qwen ラベルの行は種別語の無いスパン
（`no_type_word=True`）として扱い、写しは作らない。規則ラベルの行は従来どおり `mask_rate` で
種別語を削った写しを足す（対象は ORG / FACILITY の種別語、LOCATION の市区町村・都府県・丁目）。
`--labels` の隣の `category_labels_v1.manifest.json` を読み、門（`agreement_ok`）が true で、
`files.labels.sha256` が `--labels` の SHA-256 と一致するときだけスパンを作る（無い・不合格・不一致は拒否）。

`--judge-split`（`--labels` と併用、案 A・設計 2026-10-01 §0）: `head_judge` を文書単位で A / B の
2 つに分け、A を新しい `spans_dev.jsonl`（判定用）、B を新しい `spans_judge.jsonl`（最終評価用）に
する。旧 `head_dev` の行は丸ごと `spans_train.jsonl`（学習）へ回す。manifest の `mode` は
`"labels+judge-split"` になり、`judge_split` キーに割り振りの内訳を記録する。

usage: PYTHONPATH=. .venv/bin/python tools/category_head_data.py --data <data_v11> --templates <templates.jsonl> --out category_head_data
usage (v2): PYTHONPATH=. .venv/bin/python tools/category_head_data.py --data <data_v11> --labels <category_labels_v1.jsonl> --out category_head_data
usage (v2 + judge-split): PYTHONPATH=. .venv/bin/python tools/category_head_data.py --data <data_v11> --labels <category_labels_v1.jsonl> --out category_head_data --judge-split
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, os, random, re, sys
from collections import Counter
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from synth_v11_fill import parse_slots                       # noqa: E402  スロット順
from privacy_masker.normalize import default_normalize      # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES, OTHER, RuleCategorizer  # noqa: E402

KINDS = {"ORG": "org", "FACILITY": "facility", "LOCATION": "location"}
LABELED_TYPES = {"PERSON", "ORG", "FACILITY", "LOCATION", "ADDRESS"}
FORM_LABELS = {"location": {"prefecture": "prefecture", "city": "city", "town": "town", "country": "country", "foreign_kata": "foreign"},
               "org": {"dept": "department"}}
_LOC_SUFFIX = re.compile(r"[市区町村]$")
_PREF_SUFFIX = re.compile(r"[都府県]$")           # 「道」（北海道）は削らない
_CHOME_SUFFIX = re.compile(r"丁目$")

TRAIN_FILE = "train_v11.jsonl"
VALID_FILE = "valid_v11.jsonl"

# v2（--labels）: category_labels_v1.jsonl の split ごとの出力
HEAD_SPLITS = ("head_train", "head_dev", "head_judge")
SPLIT_FILES = {"head_train": "spans_train.jsonl", "head_dev": "spans_dev.jsonl", "head_judge": "spans_judge.jsonl"}
# `tools/category_labels.py filter` が --labels の隣に書く manifest（category_labels.LABELS_MANIFEST と同じ名前。テストで固定）
LABELS_MANIFEST = "category_labels_v1.manifest.json"


def form_label(kind: str, form: str | None) -> str | None:
    return FORM_LABELS.get(kind, {}).get(form) if form else None


def recover_forms(doc: dict, template: dict) -> list[str | None] | None:
    """エンティティ順の form。役職置換のある文書、型が合わない文書は None（復元不能）。

    data_v11 の実レコードには ``role_substituted`` が残っていない（生成時のみの一時フィールド）。
    その場合はここを素通りし、下のスロット/エンティティ整合チェックに委ねる: 役職置換が起きた
    PERSON スロットは本文中でラベル無し（肩書き語）になるため entities から丸ごと抜け、
    テンプレート由来のスロット数と件数が合わなくなって None になる。
    """
    if doc.get("role_substituted"):
        return None
    spec = {s["slot"]: s for s in template["slots"]}
    labeled = [spec[s.base] for s in parse_slots(template["template"]) if s.base in spec and s.type in LABELED_TYPES]
    ents = doc["entities"]
    if len(labeled) != len(ents) or any(sp["type"] != e[2] for sp, e in zip(labeled, ents)):
        return None
    return [sp.get("form") for sp in labeled]


def mask_type_word(text: str, start: int, end: int, kind: str, label: str) -> tuple[str, int, int] | None:
    """スパン内の種別語を本文から削った写し (text, start, end)。削れる語が無ければ None。

    MorphologicalRefiner が出すスパンの表層は前後の空白をトリム済みという前提で、
    ``type_word_span`` が返す位置（``norm.strip()`` 上）をそのままスパンの位置として使う。
    その前提が崩れている（前後に空白が残っている）スパンは位置がずれるので None にする。
    """
    # type_word_span は生の surface（raw text のスライス）にそのまま適用する位置を返す一方、
    # categorize によるラベル判定・削除後の再検査は default_normalize した文字列に対して行う
    # （正規化した文字列上のカテゴリ判定と、生の文字列上のオフセットは別物なので混同しない）。
    surface = text[start:end]
    if surface != surface.strip():
        return None
    if kind == "location":
        if label == "city" and _LOC_SUFFIX.search(surface):
            a, b = len(surface) - 1, len(surface)
        elif label == "prefecture" and _PREF_SUFFIX.search(surface):        # 「道」（北海道）は削らない
            a, b = len(surface) - 1, len(surface)
        elif label == "town" and _CHOME_SUFFIX.search(surface):
            a, b = len(surface) - 2, len(surface)
        else:
            return None
    else:
        span = RuleCategorizer.type_word_span(kind, surface)
        if span is None or span[1] - span[0] >= len(surface):
            return None
        a, b = span
    new_surface = surface[:a] + surface[b:]
    if len(new_surface) < 2:            # 削った後の表層が短すぎる（従来の len(surface) < 3 相当）
        return None
    # 削った後も規則が別の種別語を拾えてしまうなら「種別語を隠せた」とは言えない
    # （例: org「ネットワークス商船」→ type_word_span の指す末尾「商船」(transport) を削った
    #   残り「ネットワークス」がなお it と判定できる、location「田村市」→「田村」は
    #   _LOC_CITY が末尾「村」にもマッチするので city のまま。どちらも再検査で弾く）
    if RuleCategorizer.categorize(kind, default_normalize(new_surface)) != OTHER[kind]:
        return None
    return text[:start] + new_surface + text[end:], start, start + len(new_surface)


def build_spans(docs: Iterable[dict], templates: dict[str, dict] | None, *, seed: int = 42, mask_rate: float = 0.30) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for di, d in enumerate(docs):
        text = d["text"]
        forms = None
        if templates and d.get("template_id") in templates:
            forms = recover_forms(d, templates[d["template_id"]])
        for ei, (s, e, lab) in enumerate(d["entities"]):
            kind = KINDS.get(lab)
            if kind is None:
                continue
            norm = default_normalize(text[s:e])
            label, src = RuleCategorizer.categorize(kind, norm), "rule"
            if label == OTHER[kind]:
                fl = form_label(kind, forms[ei]) if forms else None
                if fl is None:
                    continue
                label, src = fl, "form"
            if label not in CATEGORIES[kind]:   # assert だと python -O で消えるので明示的に検査する
                raise ValueError(f"invalid label {label!r} for kind {kind!r}")
            doc_id = f"{d.get('source', '?')}:{di}"
            rows.append({"doc_id": doc_id, "text": text, "start": s, "end": e, "kind": kind, "label": label,
                         "label_source": src, "masked": False, "source": d.get("source", "?")})
            if src == "rule" and rng.random() < mask_rate:
                m = mask_type_word(text, s, e, kind, label)
                if m is not None:
                    rows.append({"doc_id": doc_id + "#masked", "text": m[0], "start": m[1], "end": m[2], "kind": kind,
                                 "label": label, "label_source": src, "masked": True, "source": d.get("source", "?")})
    return rows


def build_spans_from_labels(docs_by_split: dict[str, list[dict]], labels: Iterable[dict], *, seed: int = 42,
                            mask_rate: float = 0.30) -> dict[str, list[dict]]:
    """`category_labels_v1.jsonl` の行から、分割ごとの学習用スパンを作る（v2）。

    `doc_id`（"train:<行番号>" / "valid:<行番号>"）で `docs_by_split` の本文を引き、`split`
    （head_train / head_dev / head_judge）ごとにスパンを振り分ける。Qwen ラベルの行は
    `no_type_word=True`（種別語の無いスパン）として扱い、写しは作らない。規則ラベルの行は
    `mask_rate` の割合で種別語を削った写し（`masked=True` / `no_type_word=True`）を足す。
    """
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {s: [] for s in HEAD_SPLITS}
    for lab in labels:
        split = lab["split"]
        if split not in out:
            raise ValueError(f"unknown split {split!r} (expected one of {HEAD_SPLITS})")
        kind, label, src = lab["kind"], lab["category"], lab["label_source"]
        if kind not in CATEGORIES or label not in CATEGORIES[kind]:
            raise ValueError(f"invalid label {label!r} for kind {kind!r}")
        if src not in ("rule", "qwen"):
            raise ValueError(f"unknown label_source {src!r}")
        data_split, di = lab["doc_id"].split(":")
        text = docs_by_split[data_split][int(di)]["text"]
        s, e = lab["start"], lab["end"]
        base = {"doc_id": lab["doc_id"], "kind": kind, "label": label, "label_source": src, "split": split,
                "source": lab["source"]}
        out[split].append({**base, "text": text, "start": s, "end": e, "masked": False, "no_type_word": src == "qwen"})
        if src == "rule" and rng.random() < mask_rate:
            m = mask_type_word(text, s, e, kind, label)
            if m is not None:
                out[split].append({**base, "doc_id": lab["doc_id"] + "#masked", "text": m[0], "start": m[1],
                                   "end": m[2], "masked": True, "no_type_word": True})
    return out


def _base_doc(doc_id: str) -> str:
    """`doc_id` から `#masked` サフィックスを除いた基底文書 ID（A/B の割り振り単位）。"""
    return doc_id.split("#", 1)[0]


def split_judge(rows_by_split: dict[str, list[dict]], *, seed: int = 42) -> tuple[dict[str, list[dict]], dict]:
    """案 A（設計 2026-10-01 §0）: head_judge を文書単位で A / B に分け、head_dev はヘッドの学習データに回す。

    `head_judge` の行を基底文書（`doc_id` の `#masked` を除いた部分）単位で `source` ごとに整列・
    シード付きシャッフルし、交互に A（dev）/ B（judge）へ割り振る（A から始める）。種別語を隠した
    写し（`#masked`）は基底文書と同じ側に付く。`head_dev`（旧 v2 の split）は丸ごと `head_train` へ
    移す。戻り値は `(out, meta)` で、`out` は `{"head_train", "head_dev", "head_judge"}`、`meta` は
    `{"seed", "docs", "rows", "by_source", "table"}`（`table` は `masked` でない行だけの kind/label 別件数）。
    """
    rng = random.Random(seed)
    judge = rows_by_split.get("head_judge", [])
    by_source: dict[str, list[str]] = {}
    for r in judge:
        docs = by_source.setdefault(r["source"], [])
        d = _base_doc(r["doc_id"])
        if d not in docs:
            docs.append(d)
    side: dict[str, str] = {}
    per_source: dict[str, dict[str, int]] = {}
    for source in sorted(by_source):
        docs = sorted(by_source[source], key=lambda x: (x.split(":")[0], int(x.split(":")[1])))
        rng.shuffle(docs)
        per_source[source] = {"a": 0, "b": 0}
        for i, d in enumerate(docs):
            s = "a" if i % 2 == 0 else "b"
            side[d] = s
            per_source[source][s] += 1
    a = [dict(r, split="judge_a") for r in judge if side[_base_doc(r["doc_id"])] == "a"]
    b = [dict(r, split="judge_b") for r in judge if side[_base_doc(r["doc_id"])] == "b"]
    table: dict[str, dict[str, int]] = {}
    for tag, rows in (("a", a), ("b", b)):
        for r in rows:
            if r["masked"]:
                continue
            key = f"{r['kind']}/{r['label']}"
            table.setdefault(key, {"a": 0, "b": 0})[tag] += 1
    out = {"head_train": list(rows_by_split.get("head_train", [])) + [dict(r, split="head_train") for r in rows_by_split.get("head_dev", [])],
           "head_dev": a, "head_judge": b}
    meta = {"seed": seed, "docs": {"a": sum(v["a"] for v in per_source.values()), "b": sum(v["b"] for v in per_source.values())},
            "rows": {"a": len(a), "b": len(b)}, "by_source": per_source, "table": dict(sorted(table.items()))}
    return out, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def check_labels_manifest(labels_path: str) -> dict:
    """`labels_path` の隣の `category_labels_v1.manifest.json` を読み、スパンを作ってよいかを確かめる（v2）。

    `tools/category_labels.py filter` の門（`agreement_ok`）が true でなければ拒否する（不合格の選別結果を
    学習に使わない）。さらに `files.labels.sha256` と `labels_path` の SHA-256 が一致しなければ拒否する
    （別の実行の manifest との取り違え・ラベルファイルの書き換えを防ぐ）。manifest が無い・JSON として読めない
    ときも拒否する（fail-closed）。いずれも ``SystemExit``。返り値は読んだ manifest。
    """
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(labels_path)), LABELS_MANIFEST)
    if not os.path.exists(manifest_path):
        raise SystemExit(f"labels manifest not found next to --labels: {manifest_path}; "
                         "run 'tools/category_labels.py filter' to produce it")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except ValueError as exc:
        raise SystemExit(f"labels manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"labels manifest must be a JSON object: {manifest_path}")
    ok = manifest.get("agreement_ok")
    if ok is False:
        raise SystemExit("labels manifest reports agreement_ok=false; refusing to build spans")
    if ok is not True:
        raise SystemExit(f"labels manifest does not report agreement_ok=true (got {ok!r}); refusing to build spans")
    files = manifest.get("files")
    entry = files.get("labels") if isinstance(files, dict) else None
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    actual = _sha256(labels_path)
    if expected != actual:
        raise SystemExit(f"--labels sha256 {actual} does not match files.labels.sha256 {expected!r} in {manifest_path}; "
                         "refusing to build spans")
    return manifest


def _counts(rows: list[dict]) -> dict:
    return {" / ".join(str(x) for x in k): v for k, v in
            Counter((r["kind"], r["label"], r["label_source"], r["masked"], r.get("no_type_word", r["masked"]))
                    for r in rows).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data_v11 ディレクトリ（train_v11.jsonl / valid_v11.jsonl）")
    ap.add_argument("--templates", required=False, help="synth_v11/templates_v1/templates.jsonl（--labels 無しのとき必須）")
    ap.add_argument("--labels", required=False, help="category_labels_v1.jsonl（tools/category_labels.py filter の出力。v2）")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-rate", type=float, default=0.30)
    ap.add_argument("--judge-split", action="store_true",
                     help="head_judge を文書単位で A/B に分け、A を dev、B を judge にする（head_dev は学習へ）")
    args = ap.parse_args(argv)

    if not args.labels and not args.templates:
        ap.error("--templates is required without --labels")
    if args.judge_split and not args.labels:
        ap.error("--judge-split requires --labels")

    train_path = os.path.join(args.data, TRAIN_FILE)
    valid_path = os.path.join(args.data, VALID_FILE)

    if args.labels:
        for p in (train_path, valid_path, args.labels):
            if not os.path.exists(p):
                raise SystemExit(f"input not found: {p}")
        check_labels_manifest(args.labels)          # 門が不合格・manifest が無い・SHA 不一致なら SystemExit
        docs = {"train": _read_jsonl(train_path), "valid": _read_jsonl(valid_path)}
        labels = _read_jsonl(args.labels)
        by_split = build_spans_from_labels(docs, labels, seed=args.seed, mask_rate=args.mask_rate)
        judge_meta = None
        if args.judge_split:
            by_split, judge_meta = split_judge(by_split, seed=args.seed)

        os.makedirs(args.out, exist_ok=True)
        files = {}
        for split, fn in SPLIT_FILES.items():
            path = os.path.join(args.out, fn)
            _write_jsonl(path, by_split[split])
            files[split.removeprefix("head_")] = {"name": fn, "sha256": _sha256(path), "rows": len(by_split[split])}

        manifest = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "mask_rate": args.mask_rate,
            "mode": "labels+judge-split" if args.judge_split else "labels",
            "inputs": {
                "train": {"path": train_path, "sha256": _sha256(train_path)},
                "valid": {"path": valid_path, "sha256": _sha256(valid_path)},
                "labels": {"path": args.labels, "sha256": _sha256(args.labels)},
            },
            "files": files,
            "counts": {split.removeprefix("head_"): _counts(rows) for split, rows in by_split.items()},
        }
        if judge_meta is not None:
            manifest["judge_split"] = judge_meta
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(json.dumps({"files": manifest["files"], "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
        return 0

    for p in (train_path, valid_path, args.templates):
        if not os.path.exists(p):
            raise SystemExit(f"input not found: {p}")

    templates = {r["template_id"]: r for r in _read_jsonl(args.templates)}
    train_docs = _read_jsonl(train_path)
    valid_docs = _read_jsonl(valid_path)

    train_rows = build_spans(train_docs, templates, seed=args.seed, mask_rate=args.mask_rate)
    valid_rows = build_spans(valid_docs, templates, seed=args.seed, mask_rate=args.mask_rate)

    os.makedirs(args.out, exist_ok=True)
    out_train = os.path.join(args.out, "spans_train.jsonl")
    out_valid = os.path.join(args.out, "spans_valid.jsonl")
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_valid, valid_rows)

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "mask_rate": args.mask_rate,
        "inputs": {
            "train": {"path": train_path, "sha256": _sha256(train_path)},
            "valid": {"path": valid_path, "sha256": _sha256(valid_path)},
            "templates": {"path": args.templates, "sha256": _sha256(args.templates)},
        },
        "files": {
            "train": {"name": "spans_train.jsonl", "sha256": _sha256(out_train), "rows": len(train_rows)},
            "valid": {"name": "spans_valid.jsonl", "sha256": _sha256(out_valid), "rows": len(valid_rows)},
        },
        "counts": {"train": _counts(train_rows), "valid": _counts(valid_rows)},
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"files": manifest["files"], "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

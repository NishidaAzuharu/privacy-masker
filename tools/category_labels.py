"""分類ヘッド v2（設計は開発記録、非公開）:
実文書（data_v11 の out_ner / gsd_train）の ORG / FACILITY / LOCATION スパンに Qwen がカテゴリを付けるための
依頼計画（plan）、同じ Qwen による監査の計画（audit-plan）、機械検査と分割（filter）。

分担は負例 v8・合成 v11 と同じ: 依頼文・system prompt・出力スキーマはここにあり、Colab のノートブック
（tools/make_category_labels_notebook.py が生成）は計画を読んで vLLM で生成するだけ。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/category_labels.py plan --data <Drive>/data_v11 [--out-dir <Drive>/category_labels_v1] [--seed 42]
  PYTHONPATH=. .venv/bin/python tools/category_labels.py audit-plan --plan-dir <Drive>/category_labels_v1 --label-raw <raw.jsonl または Colab の出力フォルダ>
  PYTHONPATH=. .venv/bin/python tools/category_labels.py filter --data <Drive>/data_v11 --plan-dir <Drive>/category_labels_v1 \
      --label-raw <...> --audit-raw <...> [--out-dir <plan-dir>] [--seed 42] [--dev-fraction 0.10] [--max-error 0.10]

filter は門（監査で裁定した Qwen の誤り率の上限 ≤ --max-error）が不合格なら exit code 2 で、選別結果を正式名
`category_labels_v1.jsonl` ではなく `category_labels_v1.REJECTED.jsonl` に書く（manifest と report は書く）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _paths import drive_dir  # noqa: E402
from negatives_v8 import file_sha256, read_jsonl, sha256_short, write_jsonl  # noqa: E402
from privacy_masker.normalize import default_normalize  # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES, OTHER, RuleCategorizer  # noqa: E402

PLAN_VERSION = "category_labels_v1"
PLAN_TAG = "plan_v1"
PROMPT_VERSION = "category_labels_prompt_v1"
AUDIT_PROMPT_VERSION = "category_audit_prompt_v1"
OUT_DIR_NAME = "category_labels_v1"
LABEL_PLAN_FILE = "plan_label.jsonl"
LABEL_PLAN_MANIFEST = "plan_label.manifest.json"
AUDIT_PLAN_FILE = "plan_audit.jsonl"
AUDIT_PLAN_MANIFEST = "plan_audit.manifest.json"
LABEL_RAW_FILE = "qwen_labels_raw.jsonl"
AUDIT_RAW_FILE = "qwen_audits_raw.jsonl"
LABELS_FILE = "category_labels_v1.jsonl"
# 門（agreement_ok）が不合格の選別結果の書き出し先。正式名 LABELS_FILE では書かない（下流の取り違え防止）
LABELS_REJECTED_FILE = "category_labels_v1.REJECTED.jsonl"
LABELS_MANIFEST = "category_labels_v1.manifest.json"
REPORT_FILE = "filter_report.md"
TRAIN_FILE = "train_v11.jsonl"
VALID_FILE = "valid_v11.jsonl"

REAL_SOURCES: tuple[str, ...] = ("out_ner", "gsd_train")  # concat は合成テンプレート文書と GSD 文書の連結で実文書ではない（使わない）
KINDS: dict[str, str] = {"ORG": "org", "FACILITY": "facility", "LOCATION": "location"}
KIND_LABEL: dict[str, str] = {"org": "ORG（組織）", "facility": "FACILITY（施設）", "location": "LOCATION（地名）"}
MAX_SPANS = 8
CONTEXT_CHARS = 300
CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")
VERDICTS: tuple[str, ...] = ("ok", "wrong", "unsure")

# 設計 §4（2026-09-24-semantic-surrogate-design.md）の表を 1 行定義にしたもの。CATEGORIES と 1:1（テストで固定）
CATEGORY_DEFS: dict[str, dict[str, str]] = {
    "org": {
        "manufacturing": "製造業の企業（電機・機械・化学・食品・素材などを作る会社）",
        "it": "IT・通信・ソフトウェア・インターネットサービスの企業",
        "finance": "金融・保険の企業（銀行・信用金庫・証券・保険・カード・投資）",
        "retail": "小売・飲食・販売の企業（商店・ストア・百貨店・スーパー・飲食チェーン）",
        "construction": "建設・不動産・工務店・住宅の企業",
        "transport": "運輸・物流・交通・海運・航空・鉄道の企業",
        "service": "その他の企業・会社（上のどれにも当たらない会社。業種が分からない会社もここ）",
        "medical_corp": "医療法人・社会福祉法人・介護事業者（法人としての名前。病院や施設そのものは FACILITY）",
        "school_corp": "学校法人・教育事業者・予備校の運営会社（法人としての名前。学校そのものは FACILITY）",
        "government": "官公庁・自治体・公的機関・国際機関（省庁・県庁・市役所・委員会・裁判所・警察・軍・国連機関）",
        "association": "団体（協会・連盟・組合・財団・NPO・学会・宗教団体・労働組合・同好会）",
        "media": "メディア・出版・芸能・スポーツ球団・音楽グループ（新聞社・放送局・出版社・映画会社・球団・バンド・劇団）",
        "party": "政党・政治団体・議員グループ・会派",
        "department": "部署・課・室・委員会など、組織の内部の一部門（法人格の無い名前）",
    },
    "facility": {
        "medical": "医療施設（病院・クリニック・医院・歯科・薬局・保健所・診療所）",
        "education": "教育・研究施設（大学・高校・中学校・小学校・幼稚園・保育園・塾・研究所・学部）",
        "transport": "交通施設（駅・空港・港・バス停・IC・SA・道路・橋・路線・停留所）",
        "commerce": "商業施設（ホテル・旅館・店舗・モール・飲食店・市場・銀行の支店・オフィス）",
        "public": "公共施設（市役所・図書館・公民館・文化会館・体育館・公園・競技場・神社・寺・教会）",
        "building": "建物・住宅・構造物（ビル・マンション・タワー・団地・城・塔・ダム・工場・発電所）",
        "culture": "文化・観光施設（美術館・博物館・劇場・温泉・遊園地・動物園・史跡・庭園・記念館）",
        "other": "その他の施設（上のどれにも当たらない、または分からない）",
    },
    "location": {
        "country": "国・国家",
        "prefecture": "日本の都道府県",
        "city": "日本の市区町村（郡・政令市の区を含む）",
        "town": "日本の町域・丁目・大字・字",
        "district": "地区・通称地名・地域名（〜地方・〜平野・駅前・繁華街・旧国名など、行政区分でないもの）",
        "foreign": "外国の都市・州・地域・地名",
        "nature": "自然地形（山・川・湖・島・岬・湾・海・砂漠・大陸・高原）",
        "other": "その他の地名（上のどれにも当たらない、または分からない）",
    },
}

LABEL_SYSTEM_PROMPT = """あなたは日本語の固有表現にカテゴリを付ける注釈者です。本文中で『 』で囲まれ、直後に (#番号) が付いた固有表現（組織名・施設名・地名）について、文脈から最も適切なカテゴリを 1 つ選びます。

規則:
1. 判断には本文の文脈と一般知識の両方を使う。名前に種別語（株式会社・病院・市 など）が無くても、文脈（何をしている組織か、どんな場所か）から決める。
2. カテゴリは依頼文に列挙された、その型（ORG / FACILITY / LOCATION）の候補からだけ選ぶ。別の型のカテゴリは選ばない。
3. どれにも当てはまらない、または判断できないときは、その型の「その他」（ORG は service、FACILITY と LOCATION は other）を選び、confidence を low にする。
4. confidence は high（名前か文脈から明らか）/ medium（文脈から推測できる）/ low（根拠が弱い）。
5. reason は 60 字以内で、判断の根拠となった本文中の語や知識を日本語で書く。
6. 出力は {"labels": [{"index": 番号, "category": "カテゴリ名", "confidence": "high|medium|low", "reason": "根拠"}, ...]} の JSON だけ。対象の番号すべてについて 1 件ずつ、番号順に出す。説明や前置きは書かない。
"""

AUDIT_SYSTEM_PROMPT = """あなたは固有表現のカテゴリ注釈を点検する監査役です。別の注釈者が、本文中で『 』で囲まれ (#番号) が付いた固有表現（組織名・施設名・地名）に付けたカテゴリが妥当かを、本文の文脈と一般知識で点検します。

規則:
1. 各対象について verdict を ok（妥当）/ wrong（別のカテゴリが明らかに正しい）/ unsure（判断できない・根拠が弱い）から選ぶ。
2. wrong のときだけ suggested_category に正しいカテゴリ（依頼文に列挙された同じ型の候補から）を入れる。ok / unsure では null にする。
3. 名前に種別語が無くても文脈から判断する。注釈者に甘くならず、疑わしければ unsure にする。
4. reason は 60 字以内で、判断の根拠を日本語で書く。
5. 出力は {"audits": [{"index": 番号, "verdict": "ok|wrong|unsure", "suggested_category": "カテゴリ名 または null", "reason": "根拠"}, ...]} の JSON だけ。対象の番号すべてについて 1 件ずつ、番号順に出す。説明や前置きは書かない。
"""


def _all_categories() -> list[str]:
    return sorted({c for kind in KINDS.values() for c in CATEGORIES[kind]})


LABEL_SCHEMA: dict = {
    "type": "object",
    "properties": {"labels": {"type": "array", "minItems": 1, "maxItems": MAX_SPANS, "items": {
        "type": "object",
        "properties": {"index": {"type": "integer", "minimum": 1, "maximum": MAX_SPANS},
                       "category": {"type": "string", "enum": _all_categories()},
                       "confidence": {"type": "string", "enum": list(CONFIDENCES)},
                       "reason": {"type": "string", "maxLength": 120}},
        "required": ["index", "category", "confidence", "reason"], "additionalProperties": False}}},
    "required": ["labels"], "additionalProperties": False,
}

AUDIT_SCHEMA: dict = {
    "type": "object",
    "properties": {"audits": {"type": "array", "minItems": 1, "maxItems": MAX_SPANS, "items": {
        "type": "object",
        "properties": {"index": {"type": "integer", "minimum": 1, "maximum": MAX_SPANS},
                       "verdict": {"type": "string", "enum": list(VERDICTS)},
                       "suggested_category": {"anyOf": [{"type": "string", "enum": _all_categories()}, {"type": "null"}]},
                       "reason": {"type": "string", "maxLength": 120}},
        "required": ["index", "verdict", "suggested_category", "reason"], "additionalProperties": False}}},
    "required": ["audits"], "additionalProperties": False,
}

SAMPLING_LABEL = {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
SAMPLING_AUDIT = dict(SAMPLING_LABEL, temperature=0.5)


def max_tokens_for(n_spans: int) -> int:
    """8 件で 600（設計 §1.2）。1 件あたり index + カテゴリ + confidence + 60 字の reason ≈ 65 トークン。"""
    return 80 + 65 * n_spans


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def mark_spans(text: str, lo: int, hi: int, group: Sequence[tuple[int, int, str]]) -> tuple[str, list[dict]]:
    """`text[lo:hi]` の中で `group` のスパンを `『表層』(#番号)` に置き換えた本文と、番号付きのスパン一覧を返す。
    番号は 1 から（リクエストごとに振り直す）。"""
    out: list[str] = []
    spans: list[dict] = []
    cur = lo
    for i, (s, e, kind) in enumerate(group, start=1):
        out.append(text[cur:s])
        out.append(f"『{text[s:e]}』(#{i})")
        spans.append({"index": i, "kind": kind, "start": s, "end": e})
        cur = e
    out.append(text[cur:hi])
    return "".join(out), spans


def extract_requests(docs: Sequence[dict], split: str, *, max_spans: int = MAX_SPANS,
                     context_chars: int = CONTEXT_CHARS) -> list[dict]:
    """実文書（`REAL_SOURCES`）の ORG / FACILITY / LOCATION スパンを、文書ごと・`max_spans` 件ずつのリクエストにする。
    `doc_id` は `f"{split}:{行番号}"`（`tools/category_head_data.py` と選別ツールがこの id で本文を引く）。"""
    reqs: list[dict] = []
    for di, d in enumerate(docs):
        if d.get("source") not in REAL_SOURCES:
            continue
        text = d["text"]
        targets = sorted((s, e, KINDS[lab]) for s, e, lab in d["entities"] if lab in KINDS)
        for gi in range(0, len(targets), max_spans):
            group = targets[gi:gi + max_spans]
            lo = max(0, group[0][0] - context_chars)
            hi = min(len(text), group[-1][1] + context_chars)
            marked, spans = mark_spans(text, lo, hi, group)
            reqs.append({"doc_id": f"{split}:{di}", "split": split, "source": d["source"], "window": [lo, hi],
                         "truncated": [lo > 0, hi < len(text)], "spans": spans, "marked_text": marked})
    return reqs


def _definitions_block(kinds_present: set[str]) -> list[str]:
    lines: list[str] = []
    for kind in ("org", "facility", "location"):
        if kind in kinds_present:
            lines.append(f"## {KIND_LABEL[kind]}のカテゴリ")
            lines += [f"- {cat}: {desc}" for cat, desc in CATEGORY_DEFS[kind].items()]
            lines.append("")
    return lines


def build_label_prompt(req: dict) -> str:
    kinds_present = {s["kind"] for s in req["spans"]}
    lines = ["次の本文の中で『 』で囲まれ (#番号) が付いた固有表現に、カテゴリを付けてください。", ""]
    lines += _definitions_block(kinds_present)
    lines.append("## 対象")
    lines += [f"- #{s['index']}: {KIND_LABEL[s['kind']]}" for s in req["spans"]]
    lines += ["", "## 本文", req["marked_text"]]
    return "\n".join(lines)


def _interleave_by_source(reqs: list[dict], rng: random.Random) -> list[dict]:
    """source ごとにシャッフルしてから、source の比率を保つ重み付きラウンドロビンで並べる
    （`negatives_v8.build_plan` のジャンル均衡と同じ考え方。先頭 k 件を pilot にしても比率が保たれる）。"""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in reqs:
        by_source[r["source"]].append(r)
    for rows in by_source.values():
        rng.shuffle(rows)
    total = len(reqs)
    weight = {s: len(rows) / total for s, rows in by_source.items()}
    credit = {s: 0.0 for s in by_source}
    cursor = {s: 0 for s in by_source}
    out: list[dict] = []
    while len(out) < total:
        for s in by_source:
            credit[s] += weight[s]
        s = max((s for s in by_source if cursor[s] < len(by_source[s])), key=lambda k: (credit[k], k))
        credit[s] -= 1.0
        out.append(by_source[s][cursor[s]])
        cursor[s] += 1
    return out


def build_label_plan(docs_by_split: dict[str, Sequence[dict]], *, seed: int = 42, max_spans: int = MAX_SPANS,
                     context_chars: int = CONTEXT_CHARS) -> list[dict]:
    reqs: list[dict] = []
    for split in ("train", "valid"):
        reqs += extract_requests(docs_by_split.get(split, []), split, max_spans=max_spans, context_chars=context_chars)
    rows = _interleave_by_source(reqs, random.Random(seed))
    for i, r in enumerate(rows):
        r.update({"request_id": f"lab_{i:06d}", "index": i, "stage": "label", "max_tokens": max_tokens_for(len(r["spans"])),
                  "plan_version": PLAN_VERSION, "prompt_version": PROMPT_VERSION})
        r["prompt"] = build_label_prompt(r)
        r["prompt_sha256"] = sha256_short(LABEL_SYSTEM_PROMPT + "\n" + r["prompt"])
    return rows


def label_plan_manifest(rows: Sequence[dict], seed: int, plan_sha256: str, data_sha: dict) -> dict:
    return {
        "plan_version": PLAN_VERSION, "plan_tag": PLAN_TAG, "prompt_version": PROMPT_VERSION, "stage": "label",
        "seed": seed, "n": len(rows), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/category_labels.py plan", "plan_file": LABEL_PLAN_FILE, "plan_sha256": plan_sha256,
        "inputs": data_sha, "max_spans": MAX_SPANS, "context_chars": CONTEXT_CHARS,
        "system_prompt": LABEL_SYSTEM_PROMPT, "output_schema": LABEL_SCHEMA, "sampling_recommended": SAMPLING_LABEL,
        "requests_by_split": dict(Counter(r["split"] for r in rows)),
        "requests_by_source": dict(Counter(r["source"] for r in rows)),
        "spans_by_kind": dict(Counter(s["kind"] for r in rows for s in r["spans"])),
        "spans_by_split_source_kind": _spans_table(rows),
        "truncated_requests": sum(1 for r in rows if any(r["truncated"])),
    }


def _spans_table(rows: Sequence[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        for s in r["spans"]:
            c[f"{r['split']}/{r['source']}/{s['kind']}"] += 1
    return dict(sorted(c.items()))


def _load_docs(data_dir: str) -> dict[str, list[dict]]:
    out = {}
    for split, fn in (("train", TRAIN_FILE), ("valid", VALID_FILE)):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            raise SystemExit(f"input not found: {path}")
        out[split] = read_jsonl(path)
    return out


def _data_sha(data_dir: str) -> dict:
    return {split: {"path": os.path.join(data_dir, fn), "sha256": file_sha256(os.path.join(data_dir, fn))}
            for split, fn in (("train", TRAIN_FILE), ("valid", VALID_FILE))}


def cmd_plan(args) -> int:
    docs = _load_docs(args.data)
    rows = build_label_plan(docs, seed=args.seed, max_spans=args.max_spans, context_chars=args.context_chars)
    out_dir = args.out_dir or os.path.join(drive_dir(), OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, LABEL_PLAN_FILE)
    write_jsonl(plan_path, rows)
    m = label_plan_manifest(rows, args.seed, file_sha256(plan_path), _data_sha(args.data))
    with open(os.path.join(out_dir, LABEL_PLAN_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: m[k] for k in ("n", "requests_by_split", "requests_by_source", "spans_by_kind", "truncated_requests")},
                     ensure_ascii=False, indent=2))
    print("wrote", plan_path)
    return 0


# ---------------------------------------------------------------------------
# audit-plan
# ---------------------------------------------------------------------------

def _raw_by_request(raw_rows: Sequence[dict]) -> dict[str, dict]:
    """同じ request_id が複数あれば最後の行（RESUME で後から書かれた方）を採る。"""
    out: dict[str, dict] = {}
    for r in raw_rows:
        out[r["request_id"]] = r
    return out


def _matching_raw(raw_by: dict[str, dict], req: dict) -> dict | None:
    """`request_id` で raw を引き、`prompt_sha256` が計画行と一致しない raw は無し扱いにする
    （古い raw ファイルを取り違えて join したときの事故を防ぐ）。"""
    raw = raw_by.get(req["request_id"])
    if raw is None or raw.get("prompt_sha256") != req.get("prompt_sha256"):
        return None
    return raw


def parsed_items(raw: dict | None, key: str, expected: Sequence[int]) -> dict[int, dict] | None:
    """raw の `parsed[key]` を index → 項目の辞書にする。番号の欠落・重複・範囲外、parsed 無しは None（構造化失敗扱い）。"""
    if not raw or not isinstance(raw.get("parsed"), dict):
        return None
    items = raw["parsed"].get(key)
    if not isinstance(items, list):
        return None
    by: dict[int, dict] = {}
    for it in items:
        idx = it.get("index") if isinstance(it, dict) else None
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in by:
            return None
        by[idx] = it
    return by if sorted(by) == sorted(expected) else None


def build_audit_prompt(req: dict, items: dict[int, dict]) -> str:
    kinds_present = {s["kind"] for s in req["spans"]}
    lines = ["次の本文の中で『 』で囲まれ (#番号) が付いた固有表現に付いたカテゴリを点検してください。", ""]
    lines += _definitions_block(kinds_present)
    lines.append("## 対象と付いたカテゴリ")
    lines += [f"- #{s['index']}: {KIND_LABEL[s['kind']]} → {items[s['index']]['category']}" for s in req["spans"]]
    lines += ["", "## 本文", req["marked_text"]]
    return "\n".join(lines)


def build_audit_plan(label_rows: Sequence[dict], raw_rows: Sequence[dict]) -> list[dict]:
    """label の raw から監査の計画を作る。構造化失敗（番号の欠落など）のリクエストは監査しない（filter で label_structured）。"""
    raw_by = _raw_by_request(raw_rows)
    out: list[dict] = []
    for r in label_rows:
        items = parsed_items(_matching_raw(raw_by, r), "labels", [s["index"] for s in r["spans"]])
        if items is None:
            continue
        row = {k: r[k] for k in ("doc_id", "split", "source", "window", "truncated", "spans", "marked_text")}
        row.update({"label_request_id": r["request_id"], "stage": "audit",
                    "labels": {str(i): {"category": it["category"], "confidence": it["confidence"]} for i, it in sorted(items.items())},
                    "max_tokens": max_tokens_for(len(r["spans"])), "plan_version": PLAN_VERSION, "prompt_version": AUDIT_PROMPT_VERSION})
        row["prompt"] = build_audit_prompt(row, items)
        row["prompt_sha256"] = sha256_short(AUDIT_SYSTEM_PROMPT + "\n" + row["prompt"])
        out.append(row)
    for i, row in enumerate(out):
        row["request_id"] = f"aud_{i:06d}"
        row["index"] = i
    return out


def audit_plan_manifest(rows: Sequence[dict], plan_sha256: str, label_raw_sha: str) -> dict:
    return {
        "plan_version": PLAN_VERSION, "plan_tag": PLAN_TAG, "prompt_version": AUDIT_PROMPT_VERSION, "stage": "audit",
        "n": len(rows), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/category_labels.py audit-plan", "plan_file": AUDIT_PLAN_FILE, "plan_sha256": plan_sha256,
        "label_raw_sha256": label_raw_sha, "system_prompt": AUDIT_SYSTEM_PROMPT, "output_schema": AUDIT_SCHEMA,
        "sampling_recommended": SAMPLING_AUDIT,
        "requests_by_split": dict(Counter(r["split"] for r in rows)),
        "spans_by_kind": dict(Counter(s["kind"] for r in rows for s in r["spans"])),
    }


def _load_raw(paths: Sequence[str]) -> list[dict]:
    """raw の jsonl か、Colab の出力フォルダ（中の `*_raw.jsonl` を全部）を読む。"""
    rows: list[dict] = []
    for p in paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.endswith("_raw.jsonl"):
                    rows += read_jsonl(os.path.join(p, fn))
        else:
            rows += read_jsonl(p)
    return rows


def _raw_sha(paths: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        files = [os.path.join(p, fn) for fn in sorted(os.listdir(p)) if fn.endswith("_raw.jsonl")] if os.path.isdir(p) else [p]
        for f in files:
            out[f] = file_sha256(f)
    return out


def cmd_audit_plan(args) -> int:
    label_rows = read_jsonl(os.path.join(args.plan_dir, LABEL_PLAN_FILE))
    raw_rows = _load_raw(args.label_raw)
    rows = build_audit_plan(label_rows, raw_rows)
    out_dir = args.out_dir or args.plan_dir
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, AUDIT_PLAN_FILE)
    write_jsonl(plan_path, rows)
    m = audit_plan_manifest(rows, file_sha256(plan_path), json.dumps(_raw_sha(args.label_raw), sort_keys=True))
    m["skipped_label_requests"] = len(label_rows) - len(rows)
    with open(os.path.join(out_dir, AUDIT_PLAN_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: m[k] for k in ("n", "skipped_label_requests", "requests_by_split", "spans_by_kind")}, ensure_ascii=False, indent=2))
    print("wrote", plan_path)
    return 0


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

DROP_REASONS: tuple[str, ...] = ("label_structured", "prompt_mismatch", "wrong_kind_category", "forbidden_surface",
                                 "low_confidence", "audit_missing", "audit_structured", "audit_stale", "audit_not_ok",
                                 "inconsistent_surface")
HEAD_SPLITS: tuple[str, ...] = ("head_train", "head_dev", "head_judge")
RARE_CLASS_THRESHOLD = 150


def _forbidden_surfaces() -> set[str]:
    """禁止表層の正本（データフォルダの blind_eval_v1/eval_surfaces.json。tools/build_data_v7.py の EVAL_SURFACES が読む）。
    リポジトリには平文で置かない。ファイルが無い環境では空集合（検査は素通り）。
    呼び出し側は表層そのものと正規化表層の両方をこの集合に照合する。"""
    from build_data_v7 import EVAL_SURFACES
    return set(EVAL_SURFACES)


def collect_spans(label_rows: Sequence[dict], label_raw: Sequence[dict], audit_rows: Sequence[dict],
                  audit_raw: Sequence[dict], docs_by_split: dict[str, Sequence[dict]], forbidden: set[str]) -> list[dict]:
    """計画・raw・監査を結合し、スパン 1 件 = 1 dict にする。落とす理由は `drop`（設計 §1.4）。

    規則でカテゴリが決まるスパン（`rule` が None でない）は常に残す（Qwen の失敗・不一致・監査は `drop` にしない。
    Qwen の値は一致率の計測にだけ使う）。例外は禁止表層だけ。

    raw は `request_id` だけでなく `prompt_sha256` が計画行と一致するものだけを使う（`_matching_raw`）。
    一致しない label raw は `prompt_mismatch` で落とす。監査は「一致する raw が無い」「あるが構造化に失敗」
    「あるが計画時と違うカテゴリを監査した（後で label をやり直した）」を、それぞれ
    `audit_missing` / `audit_structured` / `audit_stale` として区別する。
    """
    lraw = _raw_by_request(label_raw)
    araw = _raw_by_request(audit_raw)
    audit_by_label = {a["label_request_id"]: a for a in audit_rows}
    spans: list[dict] = []
    for r in label_rows:
        expected = [s["index"] for s in r["spans"]]
        matched_lraw = _matching_raw(lraw, r)
        items = parsed_items(matched_lraw, "labels", expected)
        prompt_mismatch = r["request_id"] in lraw and matched_lraw is None
        arow = audit_by_label.get(r["request_id"])
        matched_araw = _matching_raw(araw, arow) if arow else None
        audits = parsed_items(matched_araw, "audits", expected) if arow else None
        split, di = r["doc_id"].split(":")
        text = docs_by_split[split][int(di)]["text"]
        for s in r["spans"]:
            kind = s["kind"]
            surface = text[s["start"]:s["end"]]
            norm = default_normalize(surface)
            rule = RuleCategorizer.categorize(kind, norm)
            row = {"doc_id": r["doc_id"], "split": split, "source": r["source"], "kind": kind, "start": s["start"], "end": s["end"],
                   "norm": norm, "surface_sha": sha256_short(surface), "rule": (rule if rule != OTHER[kind] else None),
                   "qwen": None, "confidence": None, "verdict": None, "suggested": None, "drop": None}
            it = items.get(s["index"]) if items else None
            au, au_state = None, None                # au_state: None（有効）/ "missing" / "structured" / "stale"
            if arow is None:
                au_state = "missing"
            elif audits is None:
                au_state = "structured"
            else:
                cand = audits.get(s["index"])
                recorded = arow["labels"].get(str(s["index"]))
                if cand is None:
                    au_state = "structured"
                elif it is None or recorded is None or recorded.get("category") != it.get("category"):
                    au_state = "stale"           # 監査が点検したのは今と違うラベル（label をやり直した後など）
                else:
                    au = cand
            if it is not None:
                row["qwen"], row["confidence"] = it.get("category"), it.get("confidence")
            if au is not None:
                row["verdict"], row["suggested"] = au.get("verdict"), au.get("suggested_category")
            if surface in forbidden or norm in forbidden:
                row["drop"] = "forbidden_surface"
            elif row["rule"] is None:                       # Qwen ラベルで残す候補にだけ掛かる条件
                if it is None:
                    row["drop"] = "prompt_mismatch" if prompt_mismatch else "label_structured"
                elif row["qwen"] not in CATEGORIES[kind]:
                    row["drop"] = "wrong_kind_category"
                elif row["confidence"] == "low":
                    row["drop"] = "low_confidence"
                elif au_state == "missing":
                    row["drop"] = "audit_missing"
                elif au_state == "structured":
                    row["drop"] = "audit_structured"
                elif au_state == "stale":
                    row["drop"] = "audit_stale"
                elif row["verdict"] != "ok":
                    row["drop"] = "audit_not_ok"
            spans.append(row)
    return spans


def apply_consistency(spans: list[dict]) -> int:
    """同じ (kind, 正規化表層) が文書をまたいで別カテゴリになった Qwen ラベルを検査する。
    多数派が 2/3 以上ならその表層は多数派カテゴリのままにし、少数派の出現だけ `inconsistent_surface` で落とす
    （少数派を多数派に書き換えることはしない。監査の `ok` は書き換え後のカテゴリに対する判定ではないため）。
    2/3 未満ならその表層の出現を全部落とす。規則ラベル付き・既に落ちた行は対象外。返り値は落とした件数。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in spans:
        if s["drop"] is None and s["rule"] is None and s["qwen"] is not None:
            groups[(s["kind"], s["norm"])].append(s)
    dropped = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        top, n_top = Counter(r["qwen"] for r in rows).most_common(1)[0]
        if n_top * 3 < len(rows) * 2:
            for r in rows:
                r["drop"] = "inconsistent_surface"
                dropped += 1
            continue
        for r in rows:
            if r["qwen"] != top:
                r["drop"] = "inconsistent_surface"
                dropped += 1
    return dropped


def agreement(spans: Sequence[dict]) -> dict[str, dict]:
    """規則ラベル付きスパンでの Qwen との一致・不一致を kind 別に集計する（設計 §1.4、ユーザー決定 2026-09-29 案 1）。

    一致率（`rate` / `rate_excl_low` / confidence 別 `by_confidence`）は診断用にそのまま出す（規則は経験則なので
    低くても規則側の癖が主因になりうる。門には使わない）。**門に使うのは `error_rate_upper`**: 不一致
    （`qwen != rule`、件数は `disagree`）のうち、監査が `wrong` で `suggested` が規則と一致するもの
    （`adjudicated_errors` = 監査が Qwen の誤りだと裁定した件数）を誤りとして数え、監査が無い・構造化に失敗した
    不一致（`unadjudicated`）も fail-closed で誤りに含めた上限。`disagree_audit_ok`（監査 ok = 規則側の癖と判定）・
    `disagree_unsure`・`disagree_wrong_other`（wrong だが提案が規則と違う）は内訳の記録用（門には使わない）。

    `n_rule_total` は Qwen ラベルの有無に関係なく、その kind の規則ラベル付きスパンの総数（ゲートの判定不能・不合格の
    切り分けに使う。§ `_stats`）。規則ラベル付きスパンが 1 件も無い kind はキー自体を作らない。Qwen ラベルが
    1 件も無い（すべて構造化失敗・prompt_mismatch など）場合は `n`・`rate` 系・`error_rate` 系が 0 / None になる。"""
    out: dict[str, dict] = {}
    for kind in ("org", "facility", "location"):
        rule_rows = [s for s in spans if s["kind"] == kind and s["rule"] is not None]
        if not rule_rows:
            continue
        rows = [s for s in rule_rows if s["qwen"] is not None]
        excl = [s for s in rows if s["confidence"] != "low"]
        sug = [s for s in rows if s["verdict"] == "wrong" and s["suggested"]]
        disagree = [s for s in rows if s["qwen"] != s["rule"]]
        adjudicated_errors = [s for s in disagree if s["verdict"] == "wrong" and s["suggested"] == s["rule"]]
        unadjudicated = [s for s in disagree if s["verdict"] is None]
        disagree_audit_ok = [s for s in disagree if s["verdict"] == "ok"]
        disagree_unsure = [s for s in disagree if s["verdict"] == "unsure"]
        disagree_wrong_other = [s for s in disagree if s["verdict"] == "wrong" and s["suggested"] != s["rule"]]
        n = len(rows)
        out[kind] = {
            "n_rule_total": len(rule_rows),
            "n": n, "rate": (sum(1 for s in rows if s["qwen"] == s["rule"]) / n) if rows else None,
            "n_excl_low": len(excl), "rate_excl_low": (sum(1 for s in excl if s["qwen"] == s["rule"]) / len(excl)) if excl else None,
            "audit_suggested_n": len(sug),
            "audit_suggested_rate": (sum(1 for s in sug if s["suggested"] == s["rule"]) / len(sug)) if sug else None,
            "by_confidence": {c: [sum(1 for s in rows if s["confidence"] == c and s["qwen"] == s["rule"]), sum(1 for s in rows if s["confidence"] == c)]
                              for c in CONFIDENCES},
            "disagree": len(disagree), "adjudicated_errors": len(adjudicated_errors), "unadjudicated": len(unadjudicated),
            "disagree_audit_ok": len(disagree_audit_ok), "disagree_unsure": len(disagree_unsure), "disagree_wrong_other": len(disagree_wrong_other),
            "error_rate": (len(adjudicated_errors) / n) if n else None,
            "error_rate_upper": ((len(adjudicated_errors) + len(unadjudicated)) / n) if n else None,
        }
    return out


def assign_splits(docs_by_split: dict[str, Sequence[dict]], seed: int = 42, dev_fraction: float = 0.10) -> dict[str, str]:
    """文書単位の分割。train の実文書は source ごとに並べてシャッフルし、先頭 `dev_fraction`（切り上げ）を head_dev、残りを head_train。
    valid の実文書は全部 head_judge。"""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    by_source: dict[str, list[str]] = defaultdict(list)
    for di, d in enumerate(docs_by_split.get("train", [])):
        if d.get("source") in REAL_SOURCES:
            by_source[d["source"]].append(f"train:{di}")
    for source in sorted(by_source):
        ids = sorted(by_source[source], key=lambda x: int(x.split(":")[1]))
        rng.shuffle(ids)
        n_dev = int(len(ids) * dev_fraction + 0.999999)
        for i, doc_id in enumerate(ids):
            out[doc_id] = "head_dev" if i < n_dev else "head_train"
    for di, d in enumerate(docs_by_split.get("valid", [])):
        if d.get("source") in REAL_SOURCES:
            out[f"valid:{di}"] = "head_judge"
    return out


def finalize(spans: Sequence[dict], head_split: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for s in spans:
        if s["drop"] is not None:
            continue
        is_rule = s["rule"] is not None
        rows.append({"doc_id": s["doc_id"], "split": head_split[s["doc_id"]], "source": s["source"], "kind": s["kind"],
                     "start": s["start"], "end": s["end"], "category": (s["rule"] if is_rule else s["qwen"]),
                     "label_source": ("rule" if is_rule else "qwen"), "confidence": (None if is_rule else s["confidence"]),
                     "surface_sha": s["surface_sha"]})
    return rows


def _stats(spans: Sequence[dict], final: Sequence[dict], agree: dict, max_error: float, min_n: int) -> dict:
    kinds = ("org", "facility", "location")
    assert all(s["drop"] in DROP_REASONS for s in spans if s["drop"] is not None), \
        sorted({s["drop"] for s in spans if s["drop"] is not None} - set(DROP_REASONS))
    doc_split = {r["doc_id"]: r["split"] for r in final}
    drops = {k: dict(Counter(s["drop"] for s in spans if s["kind"] == k and s["drop"])) for k in kinds}
    conf = {k: dict(Counter(s["confidence"] for s in spans if s["kind"] == k and s["rule"] is None and s["confidence"])) for k in kinds}
    verd = {k: dict(Counter(s["verdict"] for s in spans if s["kind"] == k and s["rule"] is None and s["verdict"])) for k in kinds}
    table: Counter = Counter((r["kind"], r["category"], r["split"]) for r in final)
    train_counts = {k: Counter(r["category"] for r in final if r["kind"] == k and r["split"] == "head_train") for k in kinds}
    rare = {k: sorted(c for c in CATEGORIES[k] if train_counts[k][c] < RARE_CLASS_THRESHOLD) for k in kinds}
    norms_by = {(k, sp): {s["norm"] for s in spans if s["drop"] is None and s["kind"] == k and doc_split.get(s["doc_id"]) == sp}
                for k in kinds for sp in ("head_train", "head_judge")}
    overlap = {}
    for k in kinds:
        tr, ju = norms_by[(k, "head_train")], norms_by[(k, "head_judge")]
        overlap[k] = {"judge_norms": len(ju), "overlap_with_train": len(tr & ju), "rate": (len(tr & ju) / len(ju)) if ju else None}
    gate = {}
    for k in kinds:
        a = agree.get(k)
        if a is None:
            gate[k] = None                                   # 判定不能（この kind の規則ラベル付きスパンが無い）
        elif a["n_rule_total"] < min_n:
            gate[k] = None                                   # 判定不能（規則ラベル付きスパンの総数が少なすぎる）
        elif a["error_rate_upper"] is None:
            gate[k] = False                                  # 規則ラベル付きスパンは十分あるが使える Qwen ラベルが 1 件も無い → 不合格
        else:
            gate[k] = a["error_rate_upper"] <= max_error
    return {"agreement": agree, "agreement_gate": gate, "agreement_ok": all(v is not False for v in gate.values()),
            "max_error": max_error, "min_agreement_n": min_n,
            "counts": {"spans": len(spans), "kept": len(final), "dropped": sum(1 for s in spans if s["drop"]),
                       "kept_by_source": {k: v for k, v in Counter(r["label_source"] for r in final).items()},
                       "kept_by_split": dict(Counter(r["split"] for r in final))},
            "drops": drops, "confidence": conf, "verdict": verd,
            "table": {f"{k}/{c}/{sp}": n for (k, c, sp), n in sorted(table.items())},
            "rare_classes_in_train": rare, "surface_overlap": overlap}


def build_report(st: dict) -> str:
    """表層を一切含めない Markdown。"""
    L = ["# 分類ヘッド v2 ラベル選別レポート", "", f"作成: {dt.datetime.now().isoformat(timespec='seconds')}", "",
         f"- スパン {st['counts']['spans']}、残した {st['counts']['kept']}、落とした {st['counts']['dropped']}",
         f"- 残した内訳: {st['counts']['kept_by_source']}、分割: {st['counts']['kept_by_split']}", "",
         f"## 規則との一致率と監査による裁定（規則ラベル付きスパン。判断は 誤り率(上限) ≤ {st['max_error']:.2f}、n(規則ラベル付き総数) ≥ {st['min_agreement_n']}）", "",
         "| kind | n | 一致率 | 不一致 | 監査 ok | unsure | wrong→規則 | wrong→他 | 未裁定 | 誤り率(上限) | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in st["agreement"].items():
        g = st["agreement_gate"].get(k)
        f = lambda v: "-" if v is None else f"{v:.3f}"
        L.append(f"| {k} | {a['n']} | {f(a['rate'])} | {a['disagree']} | {a['disagree_audit_ok']} | {a['disagree_unsure']} | "
                 f"{a['adjudicated_errors']} | {a['disagree_wrong_other']} | {a['unadjudicated']} | {f(a['error_rate_upper'])} | "
                 f"{'○' if g else ('×' if g is False else '判定不能')} |")
    L += ["", "## confidence 別の一致率（規則ラベル付き・Qwen ラベルありスパン、correct/total）", ""]
    for k, a in st["agreement"].items():
        parts = "、".join(f"{c}: {a['by_confidence'][c][0]}/{a['by_confidence'][c][1]}" for c in CONFIDENCES)
        L.append(f"- {k}: {parts}")
    L += ["", "## confidence / 監査 verdict の分布（Qwen ラベルのスパン）", ""]
    for k in ("org", "facility", "location"):
        L.append(f"- {k}: confidence {st['confidence'][k]}、verdict {st['verdict'][k]}")
    L += ["", "## 落とした理由", ""]
    for k in ("org", "facility", "location"):
        L.append(f"- {k}: {st['drops'][k]}")
    L += ["", "## kind × カテゴリ × 分割（残した件数）", "", "| kind | category | head_train | head_dev | head_judge |", "|---|---|---|---|---|"]
    for k in ("org", "facility", "location"):
        for c in CATEGORIES[k]:
            cells = [st["table"].get(f"{k}/{c}/{sp}", 0) for sp in HEAD_SPLITS]
            if any(cells):
                L.append(f"| {k} | {c} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L += ["", f"## head_train で {RARE_CLASS_THRESHOLD} 件未満のクラス", ""]
    for k in ("org", "facility", "location"):
        L.append(f"- {k}: {', '.join(st['rare_classes_in_train'][k]) or '（なし）'}")
    L += ["", "## 表層の重なり（head_judge の正規化表層のうち head_train にもあるもの）", ""]
    for k, o in st["surface_overlap"].items():
        rate = "-" if o["rate"] is None else f"{o['rate']:.3f}"
        L.append(f"- {k}: {o['overlap_with_train']} / {o['judge_norms']}（{rate}）")
    return "\n".join(L) + "\n"


def cmd_filter(args) -> int:
    docs = _load_docs(args.data)
    out_dir = args.out_dir or args.plan_dir
    label_rows = read_jsonl(os.path.join(args.plan_dir, LABEL_PLAN_FILE))
    audit_plan_path = os.path.join(args.plan_dir, AUDIT_PLAN_FILE)
    if not os.path.exists(audit_plan_path) and out_dir != args.plan_dir:
        alt = os.path.join(out_dir, AUDIT_PLAN_FILE)
        if os.path.exists(alt):
            audit_plan_path = alt
    if args.audit_raw and not os.path.exists(audit_plan_path):
        raise SystemExit(f"--audit-raw was given but no audit plan was found in {args.plan_dir} or {out_dir}; "
                         "run 'audit-plan' first")
    audit_rows = read_jsonl(audit_plan_path) if os.path.exists(audit_plan_path) else []
    label_raw = _load_raw(args.label_raw)
    if not label_raw:
        raise SystemExit(f"no label raw rows found in {args.label_raw}")
    audit_raw = _load_raw(args.audit_raw) if args.audit_raw else []
    spans = collect_spans(label_rows, label_raw, audit_rows, audit_raw, docs, _forbidden_surfaces())
    apply_consistency(spans)
    agree = agreement(spans)
    head_split = assign_splits(docs, seed=args.seed, dev_fraction=args.dev_fraction)
    final = finalize(spans, head_split)
    st = _stats(spans, final, agree, args.max_error, args.min_agreement_n)
    os.makedirs(out_dir, exist_ok=True)
    # 門が不合格なら正式名（LABELS_FILE）では書かず LABELS_REJECTED_FILE に書く。下流の
    # tools/category_head_data.py --labels は隣の manifest の agreement_ok と SHA も確かめて拒否する（二重の防止）
    labels_name = LABELS_FILE if st["agreement_ok"] else LABELS_REJECTED_FILE
    labels_path = os.path.join(out_dir, labels_name)
    write_jsonl(labels_path, final)
    with open(os.path.join(out_dir, REPORT_FILE), "w", encoding="utf-8") as f:
        f.write(build_report(st))
    manifest = {"created_at": dt.datetime.now().isoformat(timespec="seconds"), "generator": "tools/category_labels.py filter",
                "seed": args.seed, "dev_fraction": args.dev_fraction, "plan_version": PLAN_VERSION,
                "inputs": {"data": _data_sha(args.data), "label_plan": file_sha256(os.path.join(args.plan_dir, LABEL_PLAN_FILE)),
                           "audit_plan": (file_sha256(audit_plan_path) if audit_rows else None),
                           "label_raw": _raw_sha(args.label_raw), "audit_raw": _raw_sha(args.audit_raw or [])},
                "files": {"labels": {"name": labels_name, "sha256": file_sha256(labels_path), "rows": len(final)}, "report": REPORT_FILE},
                **st}
    with open(os.path.join(out_dir, LABELS_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"agreement": agree, "gate": st["agreement_gate"], "counts": st["counts"], "rare_classes_in_train": st["rare_classes_in_train"]},
                     ensure_ascii=False, indent=2))
    print("wrote", labels_path)
    if not st["agreement_ok"]:
        failed = [k for k, v in st["agreement_gate"].items() if v is False]
        print(f"監査で裁定した Qwen の誤り率が {args.max_error:.2f} を超える kind があります: {failed}。"
              f"選別結果は正式名 {LABELS_FILE} ではなく {labels_name} に書きました。"
              "学習に進まず、依頼文を直して pilot からやり直してください", file=sys.stderr)
        if os.path.exists(os.path.join(out_dir, LABELS_FILE)):
            print(f"{out_dir} に以前の実行の {LABELS_FILE} が残っていますが、manifest は今回の不合格（agreement_ok=false）"
                  "を記録しているので tools/category_head_data.py --labels はそれを読みません", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", required=True, help="data_v11 ディレクトリ（train_v11.jsonl / valid_v11.jsonl）")
    p.add_argument("--out-dir", default=None, help=f"既定は <Drive>/{OUT_DIR_NAME}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-spans", type=int, default=MAX_SPANS)
    p.add_argument("--context-chars", type=int, default=CONTEXT_CHARS)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("audit-plan")
    p.add_argument("--plan-dir", required=True, help="plan_label.jsonl のあるディレクトリ")
    p.add_argument("--label-raw", required=True, action="append", help="label 段の raw jsonl か Colab の出力フォルダ（繰り返し可）")
    p.add_argument("--out-dir", default=None, help="既定は --plan-dir")
    p.set_defaults(func=cmd_audit_plan)
    p = sub.add_parser("filter")
    p.add_argument("--data", required=True)
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--label-raw", required=True, action="append")
    p.add_argument("--audit-raw", action="append", default=None, help="audit 段の raw（無ければ Qwen ラベルは全部 audit_missing で落ちる）")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.10)
    p.add_argument("--max-error", type=float, default=0.10,
                   help="監査で裁定した Qwen の誤り率（上限、error_rate_upper）がこれを超える kind があれば学習に進まない（exit code 2）")
    p.add_argument("--min-agreement-n", type=int, default=30, help="規則ラベル付きスパンの総数がこれ未満の kind は判定不能（門を掛けない）")
    p.set_defaults(func=cmd_filter)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

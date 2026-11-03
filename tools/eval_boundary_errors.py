"""境界誤りの内訳評価: NER 5 ラベルの gold ごとに exact / boundary / type / miss を判定し、boundary のずれ幅、
決定的トークン分割との関係（gold の端がピース内部に落ちるか）、敬称直後サブセットの成績を集計する。

背景: 「田中さん」が語彙で 1 ピースになると文字ヘッドが PERSON を『田中さ』（1 文字ずれ）と出すことがある。
coverage 系の指標（tools/eval_all.py）ではこの種の 1 文字ずれは「1 字以上漏れ」に混ざって見えないので、
ずれの方向・幅とトークン境界の関係を直接数える。本文は出力しない（位置とラベルだけ。eval_all.py と同じ方針）。

判定（gold 1 件ごと。gold・予測とも端の空白・中黒・句読点は privacy_masker.evaluation.IGNORE_CHARS と同様に無視する）:
  exact     型が一致し範囲も一致
  boundary  型が一致する予測と重なるが範囲がずれる（最大重なりの予測を対応付け、start_delta = pred.start - gold.start、
            end_delta = pred.end - gold.end を記録。負 = 予測が短い側に欠ける / 正 = はみ出す）
  type      重なる予測はあるが型が違う（型一致の重なりが無い）
  miss      重なる予測が無い
  spurious  どの gold（全ラベル）とも重ならない予測（neutral の中に収まるものは別に数える）
属性（gold ごと）:
  straddle_start / straddle_end  決定的トークン分割（tokenizer の offset_mapping、文書全体）で gold の start / end が
                                 ピースの内部に落ちる（= ピース境界と一致しない）
  hon_after                      gold の直後が敬称（さん/様/氏/さま/くん/ちゃん/君/殿）のときその敬称
  ※ 推論はチャンク単位で分割するので、長い文書ではチャンク境界付近のピース割りが文書全体の分割と少し違いうる

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/eval_boundary_errors.py --gold valid=<データ>/data_v11/valid_v11.jsonl \\
      --model az225/privacy-masker-ja-pii-ner-30m [--with-refiner] [--with-morph-refiner] [--limit 300] \\
      [--out report.json] [--save-pred-dir DIR]
  PYTHONPATH=. .venv/bin/python tools/eval_boundary_errors.py --gold valid=... --pred valid=preds.jsonl [--tokenizer <model_dir>]

  --gold NAME=PATH   gold JSONL（text / entities [[start, end, label], ...] / neutral [[s, e], ...]）。複数可
  --model            モデルフォルダか Hub ID（CharHeadNerDetector）。--pred と併用可（両方の変種を併記）
  --pred NAME=PATH   既存の予測 JSONL（1 行 = [[start, end, label], ...]、gold と同じ行順）。NAME は --gold の NAME に合わせる
  --with-refiner     SpanRefiner（規則式）を掛けた結果も併記   --with-morph-refiner  MorphologicalRefiner（本番設定）も併記
  --save-pred-dir    モデルの生の予測を <DIR>/<NAME>_pred.jsonl に保存（--pred の入力形式）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_masker.evaluation import IGNORE_CHARS  # noqa: E402
from privacy_masker.types import Entity  # noqa: E402

LABELS = ("PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION")
HONORIFICS = ("ちゃん", "さん", "さま", "くん", "様", "氏", "君", "殿")   # 長い順（先頭一致なので ちゃん を先に）
OUTCOMES = ("exact", "boundary", "type", "miss")
DELTA_BINS = ("<=-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", ">=+4")
STRADDLE_KINDS = ("none", "start", "end", "both")
N_EXAMPLES = 10
DEFAULT_TOKENIZER = "cl-nagoya/ruri-v3-30m"


# ----------------------------------------------------------------------------- 入出力
def parse_named(items: list[str], flag: str) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{flag} は NAME=PATH の形で指定してください: {it}")
        name, path = it.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"{flag} {name}: ファイルが見つかりません: {path}")
        out[name] = path
    return out


def load_jsonl(path: str, limit: int | None = None) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def norm_label(label: str) -> str:
    return label[:-len("_PUBLIC")] if label.endswith("_PUBLIC") else label


def to_entities(rows: list) -> list[Entity]:
    return [Entity(str(lab), int(s), int(e), 1.0, "pred") for s, e, lab in rows]


def strip_edges(text: str, s: int, e: int) -> tuple[int, int]:
    """端の空白・中黒・句読点を除いた範囲。全部が無視文字なら元のまま。"""
    a, b = s, e
    while a < b and text[a] in IGNORE_CHARS:
        a += 1
    while a < b and text[b - 1] in IGNORE_CHARS:
        b -= 1
    return (a, b) if a < b else (s, e)


def delta_bin(d: int) -> str:
    if d <= -4:
        return "<=-4"
    if d >= 4:
        return ">=+4"
    return f"{d:+d}" if d else "0"


def straddle_kind(rec: dict) -> str:
    if rec["straddle_start"] and rec["straddle_end"]:
        return "both"
    if rec["straddle_start"]:
        return "start"
    if rec["straddle_end"]:
        return "end"
    return "none"


# ----------------------------------------------------------------------------- 判定
def piece_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    return [(int(a), int(b)) for a, b in enc["offset_mapping"] if b > a]


def judge_doc(text: str, gold: list, preds: list[Entity], neutral: list, offsets: list[tuple[int, int]], doc_idx: int):
    """1 文書分の gold ごとの判定レコードと spurious 予測を返す。"""
    pr = []
    for p in preds:
        lab = norm_label(p.entity_type)
        if lab not in LABELS:
            continue
        s, e = strip_edges(text, p.start, p.end)
        pr.append((s, e, lab))
    gl = [(strip_edges(text, int(s), int(e)), str(lab)) for s, e, lab in gold]
    records = []
    matched_pred: set[int] = set()
    for (gs, ge), lab in gl:
        overl = [(min(ge, pe) - max(gs, ps), i) for i, (ps, pe, pl) in enumerate(pr) if min(ge, pe) - max(gs, ps) > 0]
        for _, i in overl:
            matched_pred.add(i)
        if lab not in LABELS:
            continue
        rec = {"doc": doc_idx, "label": lab, "gold": [gs, ge],
               "straddle_start": any(a < gs < b for a, b in offsets),
               "straddle_end": any(a < ge < b for a, b in offsets),
               "hon_after": next((h for h in HONORIFICS if text.startswith(h, ge)), None),
               "pred": None, "pred_label": None, "start_delta": None, "end_delta": None}
        same = [(ov, i) for ov, i in overl if pr[i][2] == lab]
        if same:
            # 最大重なり。同点なら範囲のずれの合計が小さい方
            _, i = max(same, key=lambda t: (t[0], -(abs(pr[t[1]][0] - gs) + abs(pr[t[1]][1] - ge))))
            ps, pe, _ = pr[i]
            rec.update({"pred": [ps, pe], "pred_label": lab, "start_delta": ps - gs, "end_delta": pe - ge,
                        "outcome": "exact" if (ps, pe) == (gs, ge) else "boundary"})
        elif overl:
            _, i = max(overl)
            ps, pe, pl = pr[i]
            rec.update({"pred": [ps, pe], "pred_label": pl, "start_delta": ps - gs, "end_delta": pe - ge, "outcome": "type"})
        else:
            rec["outcome"] = "miss"
        records.append(rec)
    neutral_chars: set[int] = set()
    for s, e in neutral or []:
        neutral_chars.update(range(int(s), int(e)))
    spurious = []
    for i, (ps, pe, pl) in enumerate(pr):
        if i in matched_pred:
            continue
        spurious.append({"doc": doc_idx, "label": pl, "pred": [ps, pe],
                         "in_neutral": bool(neutral_chars) and set(range(ps, pe)) <= neutral_chars})
    return records, spurious


def judge_set(docs: list, preds: list[list[Entity]], offsets_list: list[list[tuple[int, int]]]):
    records, spurious = [], []
    for i, (d, p, offs) in enumerate(zip(docs, preds, offsets_list)):
        r, s = judge_doc(d["text"], d.get("entities", []), p, d.get("neutral", []), offs, i)
        records.extend(r)
        spurious.extend(s)
    return records, spurious


# ----------------------------------------------------------------------------- 集計
def aggregate(records: list[dict], spurious: list[dict]) -> dict:
    by_label = {}
    for lab in LABELS:
        rs = [r for r in records if r["label"] == lab]
        if not rs:
            continue
        c = Counter(r["outcome"] for r in rs)
        by_label[lab] = {"total": len(rs), **{o: c[o] for o in OUTCOMES},
                         "spurious": sum(1 for s in spurious if s["label"] == lab)}
    total = {"total": len(records), **{o: sum(1 for r in records if r["outcome"] == o) for o in OUTCOMES},
             "spurious": len(spurious), "spurious_in_neutral": sum(1 for s in spurious if s["in_neutral"])}

    def hist(rs, key):
        return {b: 0 for b in DELTA_BINS} | Counter(delta_bin(r[key]) for r in rs)

    bnd = [r for r in records if r["outcome"] == "boundary"]
    deltas = {"all": {"start": hist(bnd, "start_delta"), "end": hist(bnd, "end_delta")}}
    for lab in by_label:
        rs = [r for r in bnd if r["label"] == lab]
        deltas[lab] = {"start": hist(rs, "start_delta"), "end": hist(rs, "end_delta")}

    straddle = {}
    for lab in ["all"] + list(by_label):
        rs = records if lab == "all" else [r for r in records if r["label"] == lab]
        straddle[lab] = {}
        for kind in STRADDLE_KINDS:
            sub = [r for r in rs if straddle_kind(r) == kind]
            if sub:
                c = Counter(r["outcome"] for r in sub)
                straddle[lab][kind] = {"total": len(sub), **{o: c[o] for o in OUTCOMES}}

    hon = [r for r in records if r["label"] == "PERSON" and r["hon_after"]]
    hon_stats = {"total": len(hon), **{o: sum(1 for r in hon if r["outcome"] == o) for o in OUTCOMES},
                 "end_delta": hist([r for r in hon if r["outcome"] == "boundary"], "end_delta"), "by_honorific": {}, "by_merged": {}}
    for h in HONORIFICS:
        sub = [r for r in hon if r["hon_after"] == h]
        if sub:
            c = Counter(r["outcome"] for r in sub)
            hon_stats["by_honorific"][h] = {"total": len(sub), **{o: c[o] for o in OUTCOMES},
                                            "end_delta": hist([r for r in sub if r["outcome"] == "boundary"], "end_delta")}
    for merged in (True, False):
        sub = [r for r in hon if r["straddle_end"] == merged]
        if sub:
            c = Counter(r["outcome"] for r in sub)
            hon_stats["by_merged"]["merged" if merged else "split"] = {
                "total": len(sub), **{o: c[o] for o in OUTCOMES},
                "end_delta": hist([r for r in sub if r["outcome"] == "boundary"], "end_delta")}

    def pick(rs, n=N_EXAMPLES):
        return [{k: r[k] for k in ("doc", "label", "gold", "pred", "pred_label", "start_delta", "end_delta",
                                  "straddle_start", "straddle_end", "hon_after")} for r in rs[:n]]

    examples = {"boundary_honorific": pick([r for r in bnd if r["hon_after"]]),
                "boundary_other": pick([r for r in bnd if not r["hon_after"]]),
                "type": pick([r for r in records if r["outcome"] == "type"]),
                "miss": pick([r for r in records if r["outcome"] == "miss"]),
                "spurious": spurious[:N_EXAMPLES]}
    return {"total": total, "by_label": by_label, "deltas": deltas, "straddle": straddle, "honorific": hon_stats,
            "examples": examples}


# ----------------------------------------------------------------------------- Markdown
def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "-"


def md_outcomes(agg: dict) -> list[str]:
    lines = ["| ラベル | gold | exact | boundary | type | miss | spurious |", "|---|---|---|---|---|---|---|"]
    rows = list(agg["by_label"].items()) + [("合計", agg["total"])]
    for lab, s in rows:
        t = s["total"]
        lines.append(f"| {lab} | {t} | {s['exact']} ({pct(s['exact'], t)}) | {s['boundary']} ({pct(s['boundary'], t)}) | "
                     f"{s['type']} ({pct(s['type'], t)}) | {s['miss']} ({pct(s['miss'], t)}) | {s['spurious']} |")
    lines.append(f"\nspurious {agg['total']['spurious']} 件のうち neutral 内 {agg['total']['spurious_in_neutral']} 件")
    return lines


def md_deltas(agg: dict) -> list[str]:
    lines = []
    for key, title in (("start", "start_delta（pred.start - gold.start）"), ("end", "end_delta（pred.end - gold.end）")):
        lines += [f"\n**boundary の {title}**\n", "| ラベル | " + " | ".join(DELTA_BINS) + " |", "|---|" + "---|" * len(DELTA_BINS)]
        for lab, d in agg["deltas"].items():
            h = d[key]
            if sum(h.values()) == 0:
                continue
            lines.append(f"| {lab} | " + " | ".join(str(h[b]) for b in DELTA_BINS) + " |")
    return lines


def md_straddle(agg: dict) -> list[str]:
    lines = ["| ラベル | gold 端の straddle | gold | exact 率 | boundary | type | miss |", "|---|---|---|---|---|---|---|"]
    for lab, kinds in agg["straddle"].items():
        for kind, s in kinds.items():
            lines.append(f"| {lab} | {kind} | {s['total']} | {pct(s['exact'], s['total'])} | {s['boundary']} | {s['type']} | {s['miss']} |")
    return lines


def md_honorific(agg: dict) -> list[str]:
    h = agg["honorific"]
    lines = [f"PERSON gold のうち直後が敬称: {h['total']} 件  exact {h['exact']} ({pct(h['exact'], h['total'])})  "
             f"boundary {h['boundary']}  type {h['type']}  miss {h['miss']}", "",
             "| 区分 | gold | exact 率 | boundary | " + " | ".join(f"end {b}" for b in DELTA_BINS) + " |",
             "|---|---|---|---|" + "---|" * len(DELTA_BINS)]
    for name, s in list(h["by_honorific"].items()) + [(f"ピース: {k}", v) for k, v in h["by_merged"].items()]:
        lines.append(f"| {name} | {s['total']} | {pct(s['exact'], s['total'])} | {s['boundary']} | "
                     + " | ".join(str(s["end_delta"][b]) for b in DELTA_BINS) + " |")
    lines.append("\n「ピース: merged」= gold 末尾の文字を含むピースが敬称まで伸びている（straddle_end）。「split」= 名前と敬称のピースが分かれている")
    return lines


def md_examples(agg: dict) -> list[str]:
    lines = []
    for key, title in (("boundary_honorific", "boundary（敬称直後）"), ("boundary_other", "boundary（その他）"),
                       ("type", "type"), ("miss", "miss")):
        ex = agg["examples"][key]
        if not ex:
            continue
        lines += [f"\n**{title} の例（doc / label / gold / pred / Δstart / Δend / straddle / 敬称）**\n"]
        for r in ex:
            st = straddle_kind(r)
            pred = f"{r['pred'][0]}-{r['pred'][1]}" + (f" ({r['pred_label']})" if r["pred_label"] not in (None, r["label"]) else "") if r["pred"] else "-"
            lines.append(f"- doc {r['doc']} {r['label']} gold {r['gold'][0]}-{r['gold'][1]} pred {pred} "
                         f"Δstart {r['start_delta'] if r['start_delta'] is not None else '-'} Δend {r['end_delta'] if r['end_delta'] is not None else '-'} "
                         f"straddle {st} 敬称 {r['hon_after'] or '-'}")
    if agg["examples"]["spurious"]:
        lines += ["\n**spurious の例（doc / label / pred / neutral 内）**\n"]
        for s in agg["examples"]["spurious"]:
            lines.append(f"- doc {s['doc']} {s['label']} pred {s['pred'][0]}-{s['pred'][1]} neutral {'yes' if s['in_neutral'] else 'no'}")
    return lines


def to_markdown(results: dict) -> str:
    lines = ["# 境界誤りの内訳評価", ""]
    for set_name, variants in results.items():
        for variant, res in variants.items():
            agg = res["aggregate"]
            lines += [f"## {set_name} / {variant}（文書 {res['docs']}、NER gold {agg['total']['total']}）", ""]
            lines += md_outcomes(agg)
            lines += md_deltas(agg)
            lines += ["", "**gold 端のトークン境界（決定的分割）別**", ""]
            lines += md_straddle(agg)
            lines += ["", "**敬称直後サブセット（PERSON）**", ""]
            lines += md_honorific(agg)
            lines += md_examples(agg)
            lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def load_tokenizer(spec: str):
    from transformers import AutoTokenizer

    from privacy_masker.exceptions import ConfigurationError
    from privacy_masker.hub import resolve_model_dir

    try:
        path = resolve_model_dir(spec)
    except ConfigurationError:
        path = spec   # privacy-masker のモデルではない Hub ID（素の encoder など）はそのまま transformers に渡す
    return AutoTokenizer.from_pretrained(path, use_fast=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", default=[], metavar="NAME=PATH", required=True)
    ap.add_argument("--model", default=None, help="モデルフォルダか Hub ID（CharHeadNerDetector）")
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=PATH", help="既存の予測 JSONL（1 行 = [[s,e,label],...]）")
    ap.add_argument("--tokenizer", default=None,
                    help=f"straddle の計算に使う tokenizer（既定: --model のもの、無ければ {DEFAULT_TOKENIZER}）")
    ap.add_argument("--with-refiner", action="store_true", help="SpanRefiner（規則式）を掛けた結果も併記")
    ap.add_argument("--with-morph-refiner", action="store_true", help="MorphologicalRefiner（本番設定）を掛けた結果も併記")
    ap.add_argument("--limit", type=int, default=None, help="各 gold セットの先頭 N 文書だけ使う")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="集計と gold ごとの判定レコード（位置のみ）を JSON で保存")
    ap.add_argument("--save-pred-dir", default=None, help="モデルの生の予測を <DIR>/<NAME>_pred.jsonl に保存")
    args = ap.parse_args()
    if not args.model and not args.pred:
        raise SystemExit("--model か --pred のどちらかは必要です")

    golds = parse_named(args.gold, "--gold")
    preds_in = parse_named(args.pred, "--pred")
    for name in preds_in:
        if name not in golds:
            raise SystemExit(f"--pred {name}: 対応する --gold {name}=... がありません")

    detector = None
    if args.model:
        from privacy_masker.detectors import CharHeadNerDetector

        detector = CharHeadNerDetector(model=args.model, device=args.device, batch_size=args.batch_size, span_vectors=False)
        tokenizer = detector.tokenizer if args.tokenizer is None else load_tokenizer(args.tokenizer)
    else:
        tokenizer = load_tokenizer(args.tokenizer or DEFAULT_TOKENIZER)

    refiners = []
    if args.with_refiner:
        from privacy_masker.refiner import SpanRefiner

        refiners.append(("refiner", SpanRefiner()))
    if args.with_morph_refiner:
        from privacy_masker.refiner_janome import MorphologicalRefiner

        refiners.append(("morph", MorphologicalRefiner()))

    results: dict[str, dict] = {}
    for name, path in golds.items():
        docs = load_jsonl(path, args.limit)
        texts = [d["text"] for d in docs]
        offsets_list = [piece_offsets(tokenizer, t) for t in texts]
        variants: dict[str, list[list[Entity]]] = {}
        if detector is not None:
            raw = detector.detect_batch(texts)
            variants["raw"] = raw
            if args.save_pred_dir:
                os.makedirs(args.save_pred_dir, exist_ok=True)
                p = os.path.join(args.save_pred_dir, f"{name}_pred.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ents in raw:
                        f.write(json.dumps([[e.start, e.end, e.entity_type] for e in ents], ensure_ascii=False) + "\n")
                print(f"予測を保存: {p}", file=sys.stderr)
        if name in preds_in:
            rows = load_jsonl(preds_in[name], args.limit)
            if len(rows) != len(docs):
                raise SystemExit(f"--pred {name}: 行数 {len(rows)} が gold の文書数 {len(docs)} と合いません")
            variants["pred"] = [to_entities(r) for r in rows]
        for base in list(variants):
            for rname, ref in refiners:
                variants[f"{base}+{rname}"] = [ref.refine(t, ents) for t, ents in zip(texts, variants[base])]
        results[name] = {}
        for vname, preds in variants.items():
            records, spurious = judge_set(docs, preds, offsets_list)
            results[name][vname] = {"docs": len(docs), "aggregate": aggregate(records, spurious),
                                    "records": records, "spurious": spurious}

    print(to_markdown(results))
    if args.out:
        meta = {"model": args.model, "pred": preds_in, "gold": golds, "limit": args.limit,
                "refiners": [r for r, _ in refiners], "tokenizer": args.tokenizer or (args.model or DEFAULT_TOKENIZER)}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=1)
        print(f"\nJSON を保存: {args.out}")


if __name__ == "__main__":
    main()

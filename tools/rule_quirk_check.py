"""規則カテゴリ判定（RuleCategorizer）の癖の検証。Qwen ラベル付き実文書（category_labels_v1 の計画・raw・監査）から
規則ラベル付き / Qwen ラベル付きのスパンを組み立て、現行の判定（ベースライン JSON）と今の判定を比べる。
表層は出力しない: report / baseline に出すのは種別語（「電機」「委員会」「部」等）か、種別語の一致が名前全体に
及ぶときの "(whole)"、種別語が無いときの "(none)" だけで、名前全体は出さない（_type_word_label）。

usage:
  PYTHONPATH=. .venv/bin/python tools/rule_quirk_check.py dump   --data <data_v11> --plan-dir <category_labels_v1> --label-raw <dir> --audit-raw <dir> --out baseline.json
  PYTHONPATH=. .venv/bin/python tools/rule_quirk_check.py report --data <data_v11> --plan-dir <category_labels_v1> --label-raw <dir> --audit-raw <dir> --baseline baseline.json --out report.md [--json out.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections import Counter
from typing import Callable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import category_labels as cl  # noqa: E402
from privacy_masker.operators.semantic import OTHER, RuleCategorizer  # noqa: E402

KINDS = ("org", "facility", "location")
# 部署・委員会の規則（department の `^[^株合有]{1,8}(部|…|委員会)$`、government の自治体接頭の委員会）は文字列先頭から
# 一致するため、type_word_span の一致範囲が名前全体になる。その末尾の種別語だけをラベルにする
_DEPT_TYPE_WORD = re.compile(r"(部|課|室|チーム|係|委員会)$")


def build_rows(data_dir: str, plan_dir: str, label_raw: Sequence[str], audit_raw: Sequence[str]) -> list[dict]:
    docs = cl._load_docs(data_dir)
    label_rows = cl.read_jsonl(os.path.join(plan_dir, cl.LABEL_PLAN_FILE))
    audit_rows = cl.read_jsonl(os.path.join(plan_dir, cl.AUDIT_PLAN_FILE))
    spans = cl.collect_spans(label_rows, cl._load_raw(label_raw), audit_rows, cl._load_raw(audit_raw), docs, set())
    out = []
    for s in spans:
        if s["drop"] is not None and s["rule"] is None:
            continue
        split, di = s["doc_id"].split(":")
        text = docs[split][int(di)]["text"]
        out.append(dict(s, surface=text[s["start"]:s["end"]]))
    return out


def _type_word_label(kind: str, surface: str) -> str:
    """report / baseline に出す種別語のラベル。名前全体（表層）は返さない。

    type_word_span の一致範囲の末尾が部署・委員会の語（部|課|室|チーム|係|委員会）ならその語、そうでなく一致が
    名前全体に及ぶなら "(whole)"、それ以外は一致部分（種別語そのもの）。org / facility 以外や種別語が無ければ "(none)"。
    type_word_span の位置は strip 後の文字列上なので、切り出しも strip 後の文字列で行う。"""
    if kind not in ("org", "facility"):
        return "(none)"
    s = surface.strip()
    sp = RuleCategorizer.type_word_span(kind, s)
    if sp is None:
        return "(none)"
    m = _DEPT_TYPE_WORD.search(s[sp[0]:sp[1]])
    if m:
        return m.group(1)
    if sp == (0, len(s)):
        return "(whole)"
    return s[sp[0]:sp[1]]


def label_rows_with(rows: Sequence[dict], categorize: Callable[[str, str], str]) -> list[dict]:
    out = []
    for r in rows:
        cat = categorize(r["kind"], r["norm"])
        out.append(dict(r, rule_now=(None if cat == OTHER[r["kind"]] else cat), type_word=_type_word_label(r["kind"], r["surface"])))
    return out


def _key(r: Mapping) -> tuple:
    return (r["doc_id"], r["start"], r["end"])


def dump(rows: Sequence[dict], path: str) -> None:
    data = [{"doc_id": r["doc_id"], "start": r["start"], "end": r["end"], "kind": r["kind"], "rule": r["rule_now"], "type_word": r["type_word"]} for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_baseline(path: str) -> dict[tuple, dict]:
    with open(path, encoding="utf-8") as f:
        return {(d["doc_id"], d["start"], d["end"]): {"rule": d["rule"], "type_word": d["type_word"]} for d in json.load(f)}


def _cross(r: Mapping, rule: str | None) -> bool:
    return rule is not None and r["qwen"] is not None and r["qwen"] != rule and r["verdict"] == "ok"


def compare(rows_now: Sequence[dict], baseline: Mapping[tuple, dict]) -> dict:
    agreement: dict = {}
    for kind in KINDS:
        agreement[kind] = {}
        for split_name, pred in (("all", lambda r: True), ("train", lambda r: r["split"] == "train"), ("valid", lambda r: r["split"] == "valid")):
            rows = [r for r in rows_now if r["kind"] == kind and pred(r)]
            base_rows = [(r, baseline.get(_key(r), {}).get("rule")) for r in rows]
            n_b = sum(1 for r, b in base_rows if b is not None)
            x_b = sum(1 for r, b in base_rows if _cross(r, b))
            n_n = sum(1 for r in rows if r["rule_now"] is not None)
            x_n = sum(1 for r in rows if _cross(r, r["rule_now"]))
            by_tw_now = Counter(f"{r['rule_now']}/{r['type_word']}" for r in rows if _cross(r, r["rule_now"]))
            by_tw_base = Counter(f"{b}/{baseline.get(_key(r), {}).get('type_word')}" for r, b in base_rows if _cross(r, b))
            agreement[kind][split_name] = {"n_baseline": n_b, "cross_baseline": x_b, "n_now": n_n, "cross_now": x_n,
                                           "cross_by_rule_word_baseline": dict(by_tw_base.most_common(15)),
                                           "cross_by_rule_word_now": dict(by_tw_now.most_common(15))}
    newly: dict = {}
    gone: dict = {}
    other_rate: dict = {}
    for kind in KINDS:
        rows = [r for r in rows_now if r["kind"] == kind]
        new_rows = [r for r in rows if baseline.get(_key(r), {}).get("rule") is None and r["rule_now"] is not None]
        judged = [r for r in new_rows if r["qwen"] is not None and r["verdict"] == "ok"]
        mism = [r for r in judged if r["qwen"] != r["rule_now"]]
        newly[kind] = {"n": len(judged), "mismatch": len(mism), "mismatch_rate": (len(mism) / len(judged)) if judged else None,
                       "by_rule_word": {k: {"n": v, "mismatch": sum(1 for r in mism if f"{r['rule_now']}/{r['type_word']}" == k)}
                                        for k, v in Counter(f"{r['rule_now']}/{r['type_word']}" for r in judged).most_common(20)},
                       "mismatch_to": dict(Counter(f"{r['rule_now']}→{r['qwen']}" for r in mism).most_common(10))}
        gone_rows = [r for r in rows if baseline.get(_key(r), {}).get("rule") is not None and r["rule_now"] is None]
        gone[kind] = {"n": len(gone_rows), "qwen_dist": dict(Counter(r["qwen"] for r in gone_rows if r["qwen"]).most_common()),
                      "baseline_rule_dist": dict(Counter(baseline[_key(r)]["rule"] for r in gone_rows).most_common())}
        n_all = len(rows)
        other_rate[kind] = {"baseline": (sum(1 for r in rows if baseline.get(_key(r), {}).get("rule") is None) / n_all) if n_all else None,
                            "now": (sum(1 for r in rows if r["rule_now"] is None) / n_all) if n_all else None, "n": n_all}
    return {"agreement": agreement, "newly_firing": newly, "no_longer_firing": gone, "other_rate": other_rate}


def build_report(cmp: dict) -> str:
    L = ["# 規則カテゴリ判定の癖の検証", "", f"作成: {dt.datetime.now().isoformat(timespec='seconds')}", "",
         "## (a) 規則ラベル付きスパンでの監査済み不一致 ✗（Qwen ≠ 規則 かつ 監査 ok）", "",
         "| kind | split | n(旧) | ✗(旧) | n(新) | ✗(新) |", "|---|---|---|---|---|---|"]
    for kind, by_split in cmp["agreement"].items():
        for sp, a in by_split.items():
            L.append(f"| {kind} | {sp} | {a['n_baseline']} | {a['cross_baseline']} | {a['n_now']} | {a['cross_now']} |")
    L += ["", "### ✗ の発火語別（新規則、all）", ""]
    for kind, by_split in cmp["agreement"].items():
        L.append(f"- {kind}: {by_split['all']['cross_by_rule_word_now']}")
    L += ["", "## (b) 新しく発火するスパン（旧 OTHER → 新 カテゴリ）の Qwen（監査 ok）との不一致", "",
          "| kind | n | 不一致 | 不一致率 |", "|---|---|---|---|"]
    for kind, nf in cmp["newly_firing"].items():
        rate = "-" if nf["mismatch_rate"] is None else f"{nf['mismatch_rate']:.3f}"
        L.append(f"| {kind} | {nf['n']} | {nf['mismatch']} | {rate} |")
    for kind, nf in cmp["newly_firing"].items():
        L.append(f"- {kind} 発火語別: {nf['by_rule_word']}")
        L.append(f"- {kind} 不一致の向き: {nf['mismatch_to']}")
    L += ["", "## (c) 発火しなくなるスパン（旧 カテゴリ → 新 OTHER）", ""]
    for kind, g in cmp["no_longer_firing"].items():
        L.append(f"- {kind}: n {g['n']}、旧規則 {g['baseline_rule_dist']}、Qwen {g['qwen_dist']}")
    L += ["", "## OTHER 率（規則が決めないスパンの割合）", ""]
    for kind, o in cmp["other_rate"].items():
        b = "-" if o["baseline"] is None else f"{o['baseline']:.3f}"; n = "-" if o["now"] is None else f"{o['now']:.3f}"
        L.append(f"- {kind}: 旧 {b} → 新 {n}（n {o['n']}）")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "report"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True); p.add_argument("--plan-dir", required=True)
        p.add_argument("--label-raw", required=True, action="append"); p.add_argument("--audit-raw", required=True, action="append")
        p.add_argument("--out", required=True)
        if name == "report":
            p.add_argument("--baseline", required=True); p.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rows = label_rows_with(build_rows(args.data, args.plan_dir, args.label_raw, args.audit_raw), RuleCategorizer.categorize)
    if args.cmd == "dump":
        dump(rows, args.out); print("wrote", args.out, len(rows), "rows"); return 0
    cmp = compare(rows, load_baseline(args.baseline))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_report(cmp))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(cmp, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: {"n_baseline": vv["all"]["n_baseline"], "cross_baseline": vv["all"]["cross_baseline"], "n_now": vv["all"]["n_now"], "cross_now": vv["all"]["cross_now"]}
                          for kk, vv in cmp["agreement"].items()} if k == "agreement" else cmp[k] for k in ("agreement", "other_rate")}, ensure_ascii=False, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

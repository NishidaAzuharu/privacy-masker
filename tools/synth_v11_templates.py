"""Qwen の生出力（{"lines": [...]}）をテンプレートに戻し、受け入れ検査・近重複・枠を掛けて templates.jsonl にする。

usage:
  PYTHONPATH=. .venv/bin/python tools/synth_v11_templates.py filter \
      --raw <データフォルダ>/qwen3_8_27b_synth_v11_plan_v1_full_seed42 [--raw <pilot dir>] \
      [--plan <データフォルダ>/synth_v11/plan.jsonl] [--out-dir <データフォルダ>/synth_v11/templates_v1] [--target 7000]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
import random
import re

from negatives_v8 import (_weighted_choice, deterministic_reject_reasons, file_sha256, near_duplicate_pass,  # noqa: E402
                          normalize, opening_signature, parsed_text, read_jsonl, write_jsonl)
from synth_v11_plan import (ANY_BRACE_RE, FIRST_MENTION, GENRE_BY_KEY, GENRES, LABELED_TYPES, PLAN_DIR_NAME,  # noqa: E402
                            PLAN_FILE, RAW_FILE, ROLES_BY_GENRE, SLOT_RE)
from synth_v11_values import ADDRESS_FORMS, COUNTRIES, LOCATION_FORMS, ORG_FORMS, PERSON_SCRIPTS  # noqa: E402

RULE_VERSION = "synth_v11_filter_v2"
TEMPLATES_FILE = "templates.jsonl"
TEMPLATES_DIR_NAME = "templates_v1"
OPENING_CAP_PER_GENRE = 5
# 規則で見つけた PII 表層のうち、値をスロットに置き換えれば済むもの（型, ORG/ADDRESS の form）。人名は表層の切り出しが不安定なので修復しない
REPAIRABLE: dict[str, tuple[str, str | None]] = {
    "place_name": ("LOCATION", None), "department_name": ("ORG", "dept"), "known_org": ("ORG", "corporate"),
    "phone": ("PHONE", None), "email": ("EMAIL", None), "url": ("URL", None), "postal_code": ("POSTAL", None),
}
_DEPT_TAIL = "長員次課係"          # 「営業部長」の 営業部 は部署名ではなく肩書きなので置き換えない
MAX_REPAIRS = 3                    # これを超える修復が要るテンプレートはスロットの指示を無視しているので棄却（検出できない残りが疑わしい）
# 波括弧を落としたスロット名（「ORG1」・ PERSON1さん）。スロットのつもりで書いているので波括弧を補う
BARE_SLOT_RE = re.compile(r"(?<![{A-Za-z_])(PERSON|ADDRESS|LOCATION|ORG|FACILITY|PHONE|EMAIL|URL|POSTAL|DATE|ID)(\d+)(_(?:SEI|MEI|NICK))?(?![}A-Za-z0-9_])")


def template_from_raw(row: dict) -> str | None:
    return parsed_text(row.get("parsed"))


def strip_slots(template: str, placeholder: str = "") -> str:
    return SLOT_RE.sub(placeholder, template)


def load_raw_templates(paths: Sequence[str]) -> dict[str, dict]:
    """Colab の OUT_DIR かファイルを複数受け、request_id ごとに後勝ちで束ねる（negatives_v8.load_raw の RAW_FILE 違い）。"""
    rows: dict[str, dict] = {}
    for p in paths:
        path = os.path.join(p, RAW_FILE) if os.path.isdir(p) else p
        if not os.path.exists(path):
            raise SystemExit(f"raw 出力がありません: {path}")
        for r in read_jsonl(path):
            rid = r.get("request_id")
            if rid:
                rows[rid] = r
    return rows


def validate_template(template: str, plan_row: dict) -> tuple[list[str], list[str]]:
    """受け入れ検査。理由（機械名）と証拠（表層）を返す。空なら採用。"""
    reasons: list[str] = []
    evidence: list[str] = []
    if not template or not template.strip():
        return ["no_text"], []
    specs = {s["slot"]: s for s in plan_row["slots"]}
    for m in ANY_BRACE_RE.finditer(template):
        if not SLOT_RE.fullmatch(m.group(0)):
            reasons.append("malformed_slot"); evidence.append(m.group(0))
    found = list(SLOT_RE.finditer(template))
    for m in found:
        base = f"{m.group(1)}{m.group(2)}"
        if base not in specs:
            reasons.append("unknown_slot"); evidence.append(m.group(0))
    # 必須スロットの欠落・修飾だけの人物（姓だけ・あだ名だけで呼ばれる）は棄却しない。slot_usage で数える
    if not any(t in LABELED_TYPES for t in (m.group(1) for m in found)):
        reasons.append("no_labeled_slot")
    stripped = strip_slots(template)
    n = len(normalize(stripped).replace("\n", ""))
    if n < int(plan_row["length_min"] * 0.4):
        reasons.append("too_short")
    elif n > int(plan_row["length_max"] * 1.6):
        reasons.append("too_long")
    det_reasons, det_evidence = deterministic_reject_reasons(stripped)
    if det_reasons:
        reasons.append("pii_outside_slots"); evidence.extend(det_evidence or det_reasons)
    return sorted(set(reasons)), evidence


def fix_bare_slot_names(template: str) -> tuple[str, int]:
    """波括弧の無いスロット名を {…} に戻す。戻り値は (テンプレート, 直した数)。"""
    out, n = BARE_SLOT_RE.subn(lambda m: "{" + m.group(0) + "}", template)
    return out, n


def slot_usage(template: str, specs: Sequence[dict]) -> dict:
    """棄却しない統計: 使われなかった必須スロット、修飾だけで呼ばれた人物、任意スロットの利用。"""
    bases_any: set[str] = set()
    bases_plain: set[str] = set()
    for m in SLOT_RE.finditer(template):
        base = f"{m.group(1)}{m.group(2)}"
        bases_any.add(base)
        if m.group(3) is None:
            bases_plain.add(base)
    omitted = [s["slot"] for s in specs if s["required"] and s["slot"] not in bases_any]
    modifier_only = sorted(b for b in bases_any - bases_plain if b.startswith("PERSON"))
    return {"omitted_required": omitted, "modifier_only_person": modifier_only,
            "optional_used": [s["slot"] for s in specs if not s["required"] and s["slot"] in bases_any]}


def _person_scripts(plan_row: dict) -> dict[str, float]:
    g = GENRE_BY_KEY.get(plan_row.get("genre_key"))
    sub = next((x for x in (g.subgenres if g else ()) if x.key == plan_row.get("subgenre_key")), None)
    return (sub.scripts if sub and sub.scripts else PERSON_SCRIPTS)


def _new_spec(t: str, num: int, rng: random.Random, plan_row: dict, *, form: str | None = None, origin: str = "extra") -> dict:
    spec = {"slot": f"{t}{num}", "type": t, "num": num, "required": False, "role": None, "script": None, "first": None,
            "rementions": [], "form": form, "compound_suffix": None, "persona_bound": False, "origin": origin}
    if t == "PERSON":
        roles = ROLES_BY_GENRE.get(plan_row.get("genre_key"), ("関係者",))
        spec.update(role=rng.choice(roles), script=_weighted_choice(rng, _person_scripts(plan_row)), first="full")
        if spec["script"] != "historic":
            spec["first"] = _weighted_choice(rng, FIRST_MENTION)
    elif t == "LOCATION" and form is None:
        f = _weighted_choice(rng, LOCATION_FORMS)
        spec["form"] = "city" if f == "compound" else f          # 複合語は直後の語を指示できないので使わない
    elif t == "ORG" and form is None:
        spec["form"] = _weighted_choice(rng, ORG_FORMS)
    elif t == "ADDRESS" and form is None:
        spec["form"] = _weighted_choice(rng, ADDRESS_FORMS)
    return spec


def augment_specs(template: str, plan_row: dict, rng: random.Random) -> tuple[list[dict], list[str]]:
    """計画に無い番号のスロット（{DATE1} {ADDRESS1} {PERSON3} など）に spec を足して採用する。値は差し込み側が作る。"""
    specs = [dict(x) for x in plan_row["slots"]]
    known = {x["slot"] for x in specs}
    added: list[str] = []
    for m in SLOT_RE.finditer(template):
        base = f"{m.group(1)}{m.group(2)}"
        if base in known:
            continue
        known.add(base)
        specs.append(_new_spec(m.group(1), int(m.group(2)), rng, plan_row))
        added.append(base)
    return specs, added


def _place_form(surface: str) -> str:
    if surface in COUNTRIES or surface == "日本":
        return "country"
    if surface == "北海道" or surface.endswith(("県", "都", "府")):
        return "prefecture"
    if surface.endswith(("市", "区", "町", "村")):
        return "city"
    return "town"


def repair_template(template: str, specs: Sequence[dict], rng: random.Random, plan_row: dict) -> tuple[str, list[dict], list[str]]:
    """スロットの外に書かれた地名・部署名・官公庁名・連絡先の値を、新しいスロットに置き換える。人名は修復しない（棄却のまま）。"""
    specs = [dict(x) for x in specs]
    reasons, evidence = deterministic_reject_reasons(strip_slots(template))
    next_num: Counter = Counter()
    for x in specs:
        next_num[x["type"]] = max(next_num[x["type"]], int(x["num"]))
    repairs: list[str] = []
    items = []
    for ev in evidence:
        rule, _, surf = ev.partition(":")
        if rule in REPAIRABLE and surf:
            items.append((rule, surf))
    for rule, surf in sorted(set(items), key=lambda z: (-len(z[1]), z[1])):      # 長い表層から置き換える（茨城県 → 茨城 の順）
        if surf not in template:
            continue
        t, form = REPAIRABLE[rule]
        if rule == "department_name":
            pattern = re.compile(re.escape(surf) + f"(?![{_DEPT_TAIL}])")
            if not pattern.search(template):
                continue
            replacer = lambda tpl, slot, pat=pattern: pat.sub(slot, tpl)
        else:
            replacer = lambda tpl, slot, sf=surf: tpl.replace(sf, slot)
        if rule == "place_name":
            form = _place_form(surf)
        next_num[t] += 1
        slot = f"{{{t}{next_num[t]}}}"
        template = replacer(template, slot)
        spec = _new_spec(t, next_num[t], rng, plan_row, form=form, origin=f"repair:{rule}")
        specs.append(spec)
        repairs.append(f"{rule}:{surf}->{slot}")
    return template, specs, repairs


def _accepted_row(plan_row: dict, template: str, specs: Sequence[dict] | None = None, *, extra: Sequence[str] = (),
                  repairs: Sequence[str] = (), usage: dict | None = None) -> dict:
    return {"template_id": plan_row["request_id"], "template": template,
            "genre_key": plan_row["genre_key"], "subgenre_key": plan_row["subgenre_key"], "format_key": plan_row["format_key"],
            "style": plan_row["style"], "length_key": plan_row["length_key"], "slots": list(specs) if specs is not None else plan_row["slots"],
            "persona": plan_row.get("persona"), "role_ok": plan_row.get("role_ok", False), "prompt_sha256": plan_row["prompt_sha256"],
            "extra_slots": list(extra), "repairs": list(repairs), "usage": usage or {}}


def run_template_filter(raw_rows: dict[str, dict], plan_rows: Sequence[dict], *, target: int | None = None,
                        quota_by_genre: bool = True) -> dict:
    counts: Counter = Counter()
    rejected: list[dict] = []
    candidates: list[dict] = []
    for p in plan_rows:
        r = raw_rows.get(p["request_id"])
        if r is None:
            counts["missing_raw"] += 1; continue
        if r.get("prompt_sha256") and r["prompt_sha256"] != p["prompt_sha256"]:
            counts["prompt_mismatch"] += 1; continue
        t = template_from_raw(r) or ""
        rng = random.Random(f"{RULE_VERSION}:{p['request_id']}")
        t, n_bare = fix_bare_slot_names(t)
        specs, extra = augment_specs(t, p, rng)
        t, specs, repairs = repair_template(t, specs, rng, p)
        reasons, evidence = validate_template(t, {**p, "slots": specs})
        if len(repairs) > MAX_REPAIRS:
            reasons = sorted(set(reasons) | {"too_many_repairs"}); evidence = evidence + repairs
        if n_bare:
            counts["repaired_by_rule:bare_slot_name"] += 1
        if reasons:
            for x in reasons:
                counts[x] += 1
            rejected.append({"request_id": p["request_id"], "reasons": "|".join(reasons), "evidence": "|".join(evidence)[:200],
                             "head": t[:80].replace("\n", "⏎")})
            continue
        usage = slot_usage(t, specs)
        if extra:
            counts["docs_with_extra_slots"] += 1; counts["extra_slots_adopted"] += len(extra)
        if repairs:
            counts["repaired_docs"] += 1
            for x in repairs:
                counts["repaired_by_rule:" + x.split(":")[0]] += 1
        if usage["omitted_required"]:
            counts["omitted_required_docs"] += 1
            for x in usage["omitted_required"]:
                counts["omitted_required_by_type:" + re.sub(r"\d+$", "", x)] += 1
        if usage["modifier_only_person"]:
            counts["modifier_only_person_docs"] += 1
        candidates.append(_accepted_row(p, t, specs, extra=extra, repairs=repairs, usage=usage))
    # 近重複はスロットを伏せた本文で判定
    texts = [strip_slots(c["template"], "〇") for c in candidates]
    dropped, _review = near_duplicate_pass(texts)
    accepted: list[dict] = []
    openings: Counter = Counter()
    per_genre: Counter = Counter()
    weight = {g.key: g.weight for g in GENRES}
    for i, c in enumerate(candidates):
        if i in dropped:
            counts["near_duplicate"] += 1
            rejected.append({"request_id": c["template_id"], "reasons": "near_duplicate", "evidence": candidates[dropped[i][0]]["template_id"], "head": texts[i][:80]})
            continue
        sig = (c["genre_key"], opening_signature(texts[i]))
        if openings[sig] >= OPENING_CAP_PER_GENRE:
            counts["opening_repeat"] += 1
            rejected.append({"request_id": c["template_id"], "reasons": "opening_repeat", "evidence": sig[1], "head": texts[i][:80]})
            continue
        if quota_by_genre and target is not None and per_genre[c["genre_key"]] >= int(round(target * weight[c["genre_key"]] * 1.05)) + 1:
            counts["genre_quota"] += 1
            rejected.append({"request_id": c["template_id"], "reasons": "genre_quota", "evidence": c["genre_key"], "head": texts[i][:80]})
            continue
        openings[sig] += 1
        per_genre[c["genre_key"]] += 1
        accepted.append(c)
    counts["accepted"] = len(accepted)
    counts["candidates_after_rules"] = len(candidates)
    counts["planned"] = len(plan_rows)
    return {"accepted": accepted, "rejected": rejected, "counts": dict(counts), "by_genre": dict(per_genre)}


def write_template_outputs(result: dict, out_dir: str, *, plan_sha256: str, raw_sha256: dict[str, str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, TEMPLATES_FILE), result["accepted"])
    with open(os.path.join(out_dir, "rejected.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "reasons", "evidence", "head"])
        w.writeheader(); w.writerows(result["rejected"])
    slot_counts: Counter = Counter()
    for a in result["accepted"]:
        for m in SLOT_RE.finditer(a["template"]):
            slot_counts[m.group(1) + ("_" + m.group(3) if m.group(3) else "")] += 1
    manifest = {"rule_version": RULE_VERSION, "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "plan_sha256": plan_sha256, "raw_sha256": raw_sha256, "counts": result["counts"], "by_genre": result["by_genre"],
                "slot_counts": dict(slot_counts), "templates_file": TEMPLATES_FILE,
                "templates_sha256": file_sha256(os.path.join(out_dir, TEMPLATES_FILE))}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    c = result["counts"]
    lines = [f"# テンプレート受け入れ検査 {RULE_VERSION}", "",
             f"計画 {c.get('planned', 0)} / 規則通過 {c.get('candidates_after_rules', 0)} / **採用 {c.get('accepted', 0)}**", "",
             "| 理由 | 件数 |", "|---|---:|"]
    soft = ("docs_with_extra_slots", "extra_slots_adopted", "repaired_docs", "omitted_required_docs", "modifier_only_person_docs")
    is_soft = lambda k: k in soft or k.startswith(("repaired_by_rule:", "omitted_required_by_type:"))
    lines += [f"| {k} | {v} |" for k, v in sorted(c.items()) if k not in ("planned", "candidates_after_rules", "accepted") and not is_soft(k)]
    lines += ["", "採用したテンプレートの補正（棄却ではない）", "", "| 項目 | 件数 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(c.items()) if is_soft(k)]
    lines += ["", "| ジャンル | 採用 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(result["by_genre"].items())]
    lines += ["", "| スロット | 出現 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in slot_counts.most_common()]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("filter")
    p.add_argument("--raw", action="append", required=True, help="Colab の OUT_DIR かファイル（複数可。後勝ち）")
    p.add_argument("--plan", default=None); p.add_argument("--out-dir", default=None); p.add_argument("--target", type=int, default=None)
    p.add_argument("--no-quota", action="store_true")
    args = ap.parse_args(argv)
    gd = drive_dir()
    plan_path = args.plan or os.path.join(gd, PLAN_DIR_NAME, PLAN_FILE)
    out_dir = args.out_dir or os.path.join(gd, PLAN_DIR_NAME, TEMPLATES_DIR_NAME)
    plan_rows = read_jsonl(plan_path)
    raw_rows = load_raw_templates(args.raw)
    result = run_template_filter(raw_rows, plan_rows, target=args.target, quota_by_genre=not args.no_quota)
    raw_sha = {p: file_sha256(os.path.join(p, RAW_FILE) if os.path.isdir(p) else p) for p in args.raw}
    write_template_outputs(result, out_dir, plan_sha256=file_sha256(plan_path), raw_sha256=raw_sha)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2)); print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

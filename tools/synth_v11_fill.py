"""採用テンプレートに値を差し込み、位置つきの学習文書にする。値はすべてプログラム生成。位置は構築時に確定する。

usage:
  PYTHONPATH=. .venv/bin/python tools/synth_v11_fill.py fill \
      [--templates <データフォルダ>/synth_v11/templates_v1/templates.jsonl] [--out <データフォルダ>/synth_v11/filled_v1/synth_v11_docs.jsonl] [--fills 2] [--seed 42]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from negatives_v8 import file_sha256, read_jsonl, write_jsonl  # noqa: E402
from synth_v11_plan import ANY_BRACE_RE, LABELED_TYPES, PLAN_DIR_NAME, SLOT_RE, SOURCE_NAME  # noqa: E402
from synth_v11_templates import TEMPLATES_DIR_NAME, TEMPLATES_FILE  # noqa: E402
from synth_v11_values import (KEN_ALL_FILE, PERSONAS_FILE, ROLE_WORDS, Pools, build_pools, gen_address,  # noqa: E402
                              gen_facility, gen_location, gen_org, gen_unlabeled, is_excluded, render_person)

DOCS_FILE = "synth_v11_docs.jsonl"
FILLED_DIR_NAME = "filled_v1"


@dataclass(frozen=True)
class Slot:
    type: str
    num: int
    mod: str | None
    start: int
    end: int
    raw: str

    @property
    def base(self) -> str:
        return f"{self.type}{self.num}"


def parse_slots(template: str) -> list[Slot]:
    return [Slot(m.group(1), int(m.group(2)), m.group(3), m.start(), m.end(), m.group(0)) for m in SLOT_RE.finditer(template)]


def _person_surface(person: dict, first: str, mod: str | None) -> str:
    if mod == "SEI":
        return person["sei"]
    if mod == "MEI":
        return person["mei"]
    if mod == "NICK":
        return person["nick"]
    return person[{"full": "full", "sei": "sei", "mei": "mei", "nick": "nick"}[first]]


def fill_template(template: str, specs: Sequence[dict], rng: random.Random, pools: Pools, *, persona: dict | None = None,
                  role_ok: bool = False, role_rate: float = 0.10) -> dict:
    spec_by_base = {s["slot"]: s for s in specs}
    values: dict[str, object] = {}
    roles: dict[str, str] = {}
    for base, spec in spec_by_base.items():
        t = spec["type"]
        if t == "PERSON":
            if role_ok and rng.random() < role_rate:
                roles[base] = rng.choice(ROLE_WORDS)
            values[base] = render_person(rng, pools, spec["script"], persona if spec.get("persona_bound") else None)
        elif t == "ADDRESS":
            pref = persona.get("prefecture") if (persona and spec.get("persona_bound")) else None
            values[base] = gen_address(rng, pools, spec["form"] or "pref_first", prefecture=pref)
        elif t == "LOCATION":
            values[base] = gen_location(rng, pools, spec["form"] or "city")
        elif t == "ORG":
            values[base] = gen_org(rng, pools, spec["form"] or "corporate")
        elif t == "FACILITY":
            values[base] = gen_facility(rng, pools)
        else:
            values[base] = gen_unlabeled(rng, t)
    out: list[str] = []
    entities: list[list] = []
    pos = 0
    built = 0
    for s in parse_slots(template):
        out.append(template[pos:s.start]); built += s.start - pos; pos = s.end
        spec = spec_by_base.get(s.base)
        if spec is None:                       # 検査で落ちているはずだが、安全側で素通しせず空にする
            continue
        if s.type == "PERSON":
            surface = roles[s.base] if s.base in roles else _person_surface(values[s.base], spec["first"], s.mod)
            labeled = s.base not in roles
        else:
            surface = str(values[s.base]); labeled = s.type in LABELED_TYPES
        out.append(surface)
        if labeled and surface:
            entities.append([built, built + len(surface), s.type])
        built += len(surface)
    out.append(template[pos:])
    text = "".join(out)
    return {"text": text, "entities": sorted(entities, key=lambda x: (x[0], x[1])), "source": SOURCE_NAME,
            "variant": "", "template_id": "", "fill_index": 0,
            "persona_uuid": (persona or {}).get("uuid"), "role_substituted": len(roles)}


def validate_filled(doc: dict, pools: Pools) -> list[str]:
    """PERSON は excl_person、ORG / FACILITY は excl_place で検査する。
    LOCATION / ADDRESS は町域だけが除外対象なので gen_location / gen_address 側の除外に任せ、ここでは検査しない。"""
    reasons: list[str] = []
    text = doc["text"]
    if ANY_BRACE_RE.search(text):
        reasons.append("placeholder_left")
    for s, e, lab in doc["entities"]:
        seg = text[s:e]
        if not seg or seg != seg.strip():
            reasons.append("span_edge_space")
        if lab == "PERSON" and is_excluded(seg, pools.excl_person):
            reasons.append("eval_person_surface")
        if lab in ("ORG", "FACILITY") and is_excluded(seg, pools.excl_place):
            reasons.append("eval_place_surface")
    return sorted(set(reasons))


def fill_all(templates: Sequence[dict], pools: Pools, *, seed: int = 42, fills: int = 2, role_rate: float = 0.10) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    for t in templates:
        rng = random.Random(f"{seed}:{t['template_id']}")
        persona = t.get("persona")
        if persona and "sei" not in persona:       # 計画行のペルソナは名前を持たない → スナップショットから引く
            persona = next((p for p in pools.personas if p["uuid"] == persona.get("uuid")), None)
        for k in range(fills):
            d = fill_template(t["template"], t["slots"], rng, pools, persona=persona if k == 0 else None,
                              role_ok=bool(t.get("role_ok")), role_rate=role_rate)
            if d["text"] in seen or validate_filled(d, pools):
                continue
            seen.add(d["text"])
            d["variant"] = f"{t['genre_key']}/{t['subgenre_key']}/{t['format_key']}"
            d["template_id"] = t["template_id"]; d["fill_index"] = k
            docs.append(d)
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fill")
    p.add_argument("--templates", default=None); p.add_argument("--out", default=None)
    p.add_argument("--fills", type=int, default=2); p.add_argument("--seed", type=int, default=42); p.add_argument("--role-rate", type=float, default=0.10)
    args = ap.parse_args(argv)
    gd = drive_dir()
    tpath = args.templates or os.path.join(gd, PLAN_DIR_NAME, TEMPLATES_DIR_NAME, TEMPLATES_FILE)
    out = args.out or os.path.join(gd, PLAN_DIR_NAME, FILLED_DIR_NAME, DOCS_FILE)
    pools = build_pools()
    templates = read_jsonl(tpath)
    docs = fill_all(templates, pools, seed=args.seed, fills=args.fills, role_rate=args.role_rate)
    write_jsonl(out, docs)
    labels = Counter(l for d in docs for _, _, l in d["entities"])
    manifest = {"created_at": dt.datetime.now().isoformat(timespec="seconds"), "templates": tpath, "templates_sha256": file_sha256(tpath),
                "personas_sha256": file_sha256(os.path.join(gd, PLAN_DIR_NAME, PERSONAS_FILE)),
                "ken_all_sha256": file_sha256(os.path.join(gd, PLAN_DIR_NAME, KEN_ALL_FILE)),
                "seed": args.seed, "fills": args.fills, "role_rate": args.role_rate, "templates_in": len(templates), "docs": len(docs),
                "labels": dict(labels), "role_substituted_docs": sum(1 for d in docs if d["role_substituted"]),
                "persona_bound_docs": sum(1 for d in docs if d["persona_uuid"]),
                "excl_person": len(pools.excl_person), "excl_place": len(pools.excl_place), "docs_sha256": file_sha256(out)}
    with open(out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in ("templates_in", "docs", "labels", "role_substituted_docs", "persona_bound_docs")}, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

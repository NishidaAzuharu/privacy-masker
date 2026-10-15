"""llm_templates_v5.json → llm_templates_v5b.json: 平文の機関語をスロット化する。

背景（2026-09-02）: v5 テンプレート 136/1521 本に「総務課・医事課・地域包括支援センター」等の機関語が
ラベルなし平文で含まれ、手テンプレ合成の「部課名=ORG・窓口=FACILITY」と矛盾する負例として学習された
（当時の本番モデルで gov 76%・ORG 危険 6→21）。部課名は新スロット {DEPT}（ラベル ORG）に、明確な施設名は
{FACILITY} に置き換える。一般語（相談窓口・コールセンター・現場事務所）と誤マッチ（関係・放課後・課長）は残す。

実行: PYTHONPATH=. .venv/bin/python tools/fix_templates_v5b.py [--src PATH] [--dst PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
import re
from collections import Counter

GD = drive_dir()
SLOT_RE = re.compile(r"\{([A-Z]+)(\d+)(_SEI)?\}")
DEPT_RE = re.compile(r"[一-龥ァ-ヶー]{2,14}(課|係|事務局)")
_KANA_KANJI = re.compile(r"[一-龥ァ-ヶー]")
FAC_RE = re.compile(r"[一-龥ァ-ヶー]{2,12}(窓口|センター|事務所)")
GENERIC_DEPT = {"所管課", "担当課", "各課", "当課", "二重課", "放課"}
# 部課名ではない「課」「係」の語（人事考課・公租公課・賦課・係数 等）
_NOT_DEPT_SUFFIX = ("考課", "公課", "賦課", "課税", "係数", "係争", "係留")
# 明確に施設名として扱えるもののみ（一般語の窓口・センターは平文のまま）
FAC_ALLOW = {"交付窓口", "事務室窓口", "多言語窓口", "保守窓口",
             "地域包括支援センター", "各地区市民センター", "県税事務所", "漁協事務所"}
_AFTER_BAD_DEPT = "後税題目程長員数争留"   # 放課後・課税・課題・課長・課員・係数・係争 は変換しない


def convert(template: str) -> tuple[str, list[tuple[str, str]]]:
    slot_spans = [(m.start(), m.end()) for m in SLOT_RE.finditer(template)]
    max_idx = Counter()
    for m in SLOT_RE.finditer(template):
        max_idx[m.group(1)] = max(max_idx[m.group(1)], int(m.group(2)))

    def in_slot(s, e):
        return any(s < se and e > ss for ss, se in slot_spans)

    repl: list[tuple[int, int, str, str]] = []  # (start, end, surface, base)
    for m in DEPT_RE.finditer(template):
        surf, after = m.group(0), template[m.end():m.end() + 1]
        if in_slot(m.start(), m.end()) or surf in GENERIC_DEPT or "関係" in surf or "放課" in surf:
            continue
        if surf.endswith(_NOT_DEPT_SUFFIX) or after in _AFTER_BAD_DEPT:
            continue
        # 直前が漢字・カタカナなら長さ上限で名称が切れている（要保護児童対策地域協議会事務局 等）→ 変換しない
        if m.start() > 0 and _KANA_KANJI.match(template[m.start() - 1]):
            continue
        repl.append((m.start(), m.end(), surf, "DEPT"))
    for m in FAC_RE.finditer(template):
        surf = m.group(0)
        if in_slot(m.start(), m.end()) or surf not in FAC_ALLOW:
            continue
        repl.append((m.start(), m.end(), surf, "FACILITY"))
    if not repl:
        return template, []
    # 同一表層は同一スロット番号（同じ値が入る）。番号は既存スロットの続きから
    assigned: dict[tuple[str, str], str] = {}
    changes = []
    out = template
    for s, e, surf, base in sorted(repl, key=lambda x: -x[0]):
        key = (base, surf)
        if key not in assigned:
            max_idx[base] += 1
            assigned[key] = f"{{{base}{max_idx[base]}}}"
        out = out[:s] + assigned[key] + out[e:]
        changes.append((surf, assigned[key]))
    return out, changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{GD}/llm_templates_v5.json")
    ap.add_argument("--dst", default=f"{GD}/llm_templates_v5b.json")
    args = ap.parse_args()
    items = json.load(open(args.src, encoding="utf-8"))
    out, audit, n_changed = [], [], 0
    surf_counter = Counter()
    for i, it in enumerate(items):
        new, changes = convert(it["template"])
        if changes:
            n_changed += 1
            audit.append({"index": i, "genre": it["genre"], "changes": changes,
                          "before": it["template"], "after": new})
            surf_counter.update(s for s, _ in changes)
        out.append({**it, "template": new})
    json.dump(out, open(args.dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(audit, open(args.dst.replace(".json", "_changes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"templates: {len(items)}  changed: {n_changed}  conversions: {sum(surf_counter.values())}")
    print("converted surfaces:", surf_counter.most_common())
    print(f"saved: {args.dst}")
    # 変換後に残る機関語（平文）の確認
    left = Counter()
    for it in out:
        t = SLOT_RE.sub("〓", it["template"])
        left.update(m.group(0) for m in DEPT_RE.finditer(t) if "関係" not in m.group(0))
    print("残った 課/係/事務局 の平文（意図的に残したもの）:", left.most_common())


if __name__ == "__main__":
    main()

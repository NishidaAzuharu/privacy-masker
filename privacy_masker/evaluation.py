"""評価: gold 付き文書に対する検出・匿名化の到達度（scrubadub.comparison / 本プロジェクトの tools/eval_all.py の公開版）。

入力は `{"text": str, "entities": [[start, end, label], ...], "neutral": [[start, end], ...]?}` の列。
指標は文字単位:
  coverage    gold 文字のうちマスクされた割合（空白・中黒・終端ピリオドは分母から除く。2 スパン検出は漏れではない）
  danger      gold エンティティのうち coverage < 50% のもの（実質漏れ）
  miss        全く検出されなかった gold
  anyleak     1 文字以上漏れた gold
  fp_chars    gold・neutral の外でマスクされた文字数（NER 系ラベルのみ。電話・メール等の正規表現型は数えない）
使い方:
  from privacy_masker.evaluation import evaluate, load_jsonl
  report = evaluate(load_jsonl("gold.jsonl"), engine)        # engine.detect() を使う
  report = evaluate(docs, predictions=[...])                 # 既存の検出結果（list[list[Entity]]）を採点
  print(report.to_markdown())
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .types import Entity

DEFAULT_LABELS: tuple[str, ...] = ("PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION")
IGNORE_CHARS = frozenset(" 　\t\r\n・,.、。")


@dataclass
class LabelStats:
    total: int = 0
    danger: int = 0
    miss: int = 0
    anyleak: int = 0


@dataclass
class EvaluationReport:
    documents: int
    gold_entities: int
    coverage: float
    danger: int
    miss: int
    anyleak: int
    fp_chars: int
    by_label: dict[str, LabelStats] = field(default_factory=dict)
    leaks: list[dict] = field(default_factory=list)   # {"doc": i, "label", "start", "end", "coverage"}（原文は含めない）

    def to_markdown(self) -> str:
        head = "| 文書 | gold | 文字cov | 危険 | 完全見逃し | 1字以上漏れ | 過剰マスク字 |\n|---|---|---|---|---|---|---|\n"
        row = f"| {self.documents} | {self.gold_entities} | {self.coverage*100:.1f}% | {self.danger} | {self.miss} | {self.anyleak} | {self.fp_chars} |\n"
        lab = "\n| ラベル | gold | 危険 | 完全見逃し | 1字以上漏れ |\n|---|---|---|---|---|\n" + "".join(
            f"| {l} | {s.total} | {s.danger} | {s.miss} | {s.anyleak} |\n" for l, s in self.by_label.items())
        return head + row + lab


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_span(text: str, s: int, e: int) -> set[int]:
    return {i for i in range(s, e) if text[i] not in IGNORE_CHARS} or set(range(s, e))


def evaluate(
    docs: Sequence[Mapping], engine=None, *, predictions: Sequence[Sequence[Entity]] | None = None,
    labels: Iterable[str] = DEFAULT_LABELS, fp_labels: Iterable[str] | None = None, use_neutral: bool = True,
    record_leaks: bool = True, **detect_options,
) -> EvaluationReport:
    if (engine is None) == (predictions is None):
        raise ValueError("pass exactly one of engine= or predictions=")
    if predictions is None:
        predictions = [engine.detect(d["text"], **detect_options) for d in docs]
    labels = tuple(labels); fp_labels = set(fp_labels) if fp_labels is not None else set(labels)
    gold_c = cov_c = fp_c = 0
    stats: dict[str, LabelStats] = {}
    leaks: list[dict] = []
    for i, (d, ents) in enumerate(zip(docs, predictions)):
        t = d["text"]
        pc, pc_fp = set(), set()
        for e in ents:
            pc.update(range(e.start, e.end))
            if e.entity_type in fp_labels or e.entity_type.rstrip("_PUBLIC") in fp_labels:
                pc_fp.update(range(e.start, e.end))
        neutral = set()
        if use_neutral:
            for s, e_ in d.get("neutral", []):
                neutral.update(range(s, e_))
        gold_all, gold_full = set(), set()
        for s, e_, lab in d["entities"]:
            span = gold_span(t, s, e_); gold_all |= span; gold_full.update(range(s, e_))
            st = stats.setdefault(lab, LabelStats()); st.total += 1
            c = len(span & pc) / len(span)
            if c < 1:
                st.anyleak += 1
                if c < 0.5: st.danger += 1
                if c == 0: st.miss += 1
                if record_leaks:
                    leaks.append({"doc": i, "label": lab, "start": s, "end": e_, "coverage": round(c, 3)})
        gold_c += len(gold_all); cov_c += len(gold_all & pc); fp_c += len(pc_fp - gold_full - neutral)
    tot = Counter(); 
    return EvaluationReport(
        documents=len(docs), gold_entities=sum(s.total for s in stats.values()),
        coverage=cov_c / gold_c if gold_c else 1.0,
        danger=sum(s.danger for s in stats.values()), miss=sum(s.miss for s in stats.values()),
        anyleak=sum(s.anyleak for s in stats.values()), fp_chars=fp_c,
        by_label={l: stats[l] for l in list(labels) + [k for k in stats if k not in labels] if l in stats},
        leaks=leaks,
    )

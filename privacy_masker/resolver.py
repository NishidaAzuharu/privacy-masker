"""重複解決（スペック§6）。決定的・貪欲。負けたentityの残余は再評価する。"""
from __future__ import annotations

import dataclasses
import heapq
from typing import Sequence

from .types import Entity

DEFAULT_PRIORITY = [
    "MY_NUMBER",
    "CREDIT_CARD",
    "CORPORATE_NUMBER",
    "EMAIL",
    "URL",
    "PHONE_NUMBER",
    "RESIDENCE_CARD",
    "PASSPORT",
    "DRIVERS_LICENSE",
    "PENSION_NUMBER",
    "BANK_ACCOUNT",
    "PERSON",
    "ADDRESS",
    "POSTAL_CODE",
    "LOCATION",
    "ORG",
    "FACILITY",
    "USERNAME",
    "DATE",
    "AGE",
    "IP_ADDRESS",
    "MAC_ADDRESS",
]


class EntityResolver:
    def __init__(self, priority: Sequence[str] | None = None, min_score: float = 0.0) -> None:
        self.priority = list(priority) if priority is not None else list(DEFAULT_PRIORITY)
        self.min_score = min_score

    def resolve(self, entities: Sequence[Entity]) -> list[Entity]:
        cands = [e for e in entities if e.score >= self.min_score]
        prio = {t: i for i, t in enumerate(self.priority)}
        next_idx = len(self.priority)
        for e in cands:  # リスト外typeは検出結果への出現順で末尾に付ける
            if e.entity_type not in prio:
                prio[e.entity_type] = next_idx
                next_idx += 1

        heap: list[tuple[int, float, int, int, int, int, Entity]] = []
        counter = 0

        def push(e: Entity, reg: int) -> None:
            nonlocal counter
            heapq.heappush(
                heap,
                (prio[e.entity_type], -e.score, -(e.end - e.start), e.start, reg, counter, e),
            )
            counter += 1

        for reg, e in enumerate(cands):
            push(e, reg)

        adopted: list[Entity] = []
        while heap:
            _, _, _, _, reg, _, e = heapq.heappop(heap)
            overlaps = [a for a in adopted if a.start < e.end and e.start < a.end]
            if not overlaps:
                adopted.append(e)
                continue
            # 採用済みと重なる部分を切り落とし、非空の残余を再挿入する
            segments = [(e.start, e.end)]
            for a in overlaps:
                nxt: list[tuple[int, int]] = []
                for s0, s1 in segments:
                    if a.end <= s0 or s1 <= a.start:
                        nxt.append((s0, s1))
                        continue
                    if s0 < a.start:
                        nxt.append((s0, a.start))
                    if a.end < s1:
                        nxt.append((a.end, s1))
                segments = nxt
            for s0, s1 in segments:
                if s1 > s0:
                    push(dataclasses.replace(e, start=s0, end=s1), reg)
        return sorted(adopted, key=lambda a: a.start)

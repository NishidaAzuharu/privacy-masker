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

# 内側に完全に収まる他の検出を丸ごと吸収する型。URL の ?card=4111… やメールのローカル部の 12 桁を
# CREDIT_CARD / MY_NUMBER として切り出すと URL・メールが分断されるので、優先順位に関わらず外側を採る
DEFAULT_CONTAINER_TYPES = ("URL", "EMAIL")


class EntityResolver:
    def __init__(self, priority: Sequence[str] | None = None, min_score: float = 0.0,
                 container_types: Sequence[str] = DEFAULT_CONTAINER_TYPES) -> None:
        self.priority = list(priority) if priority is not None else list(DEFAULT_PRIORITY)
        self.min_score = min_score
        self.container_types = frozenset(container_types)

    def resolve(self, entities: Sequence[Entity]) -> list[Entity]:
        cands = self._absorb_into_containers([e for e in entities if e.score >= self.min_score])
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

    def _absorb_into_containers(self, cands: list[Entity]) -> list[Entity]:
        """container_types の内側に完全に収まる（かつ範囲がより短い）検出を落とす。同じ範囲どうしは優先順位に任せる。"""
        boxes = [b for b in cands if b.entity_type in self.container_types]
        if not boxes:
            return cands

        def inside(e: Entity) -> bool:
            return any(b is not e and b.start <= e.start and e.end <= b.end and (b.end - b.start) > (e.end - e.start)
                       for b in boxes)

        return [e for e in cands if not inside(e)]

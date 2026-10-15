"""Detector Protocol（スペック§5）。"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..types import Entity


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Entity]: ...


@runtime_checkable
class BatchDetector(Detector, Protocol):
    def detect_batch(self, texts: Sequence[str]) -> list[list[Entity]]: ...

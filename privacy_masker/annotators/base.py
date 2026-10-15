"""Annotator Protocol（識別性ベース匿名化、docs/design_identifiability.md §2）。

Detector → Refiner → EntityResolver の後、Policy の前に入るステージ。エンティティの
型をサブタイプ（例: PERSON → PERSON_PUBLIC）に書き換えることで、既存 Policy の
「未知の型は default operator に落とす」性質をそのまま fail-closed に使う。
Policy にサブタイプの規則が無ければ従来どおりマスクされる（保持はオプトイン）。

Annotator は原文を返り値に含めず、Entity の位置・スコア・型だけを扱う。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..types import Entity


@runtime_checkable
class Annotator(Protocol):
    name: str

    def annotate(self, text: str, entities: Sequence[Entity]) -> list[Entity]:
        """resolve 済みのエンティティ列を受け取り、型を書き換えた列を返す。

        位置（start/end）は変えないこと。位置の変更は Refiner の責務。
        """
        ...

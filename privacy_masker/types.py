"""コア型。PII非漏洩のため、原文フィールドを持たず、reprも原文由来文字列を出さない。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Entity:
    """検出結果の共通フォーマット。原文テキストは保持しない。"""

    entity_type: str
    start: int
    end: int
    score: float
    source: str = ""
    # 文字ヘッド NER の encoder 隠れ状態をスパン上で平均した文脈ベクトル（float16 little-endian のバイト列。
    # 30m は 256 次元 = 512 バイト）。検出器が付ける任意情報で、原文の文字は含まない。
    # 同一性の比較・repr には入れない（Refiner / Annotator の dataclasses.replace では引き継がれる）
    vector: bytes | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TransformedEntity:
    entity_type: str
    original_start: int
    original_end: int
    replacement: str
    score: float
    operator: str
    # transform(explain=True) のとき、検出器・Refiner・Annotator・重複解決の経緯を人が読める形で入れる
    # 。原文の文字列は含めない
    explanation: str | None = None

    def __repr__(self) -> str:  # replacementはMask等で原文文字を含み得るため出さない
        return (
            f"TransformedEntity(entity_type={self.entity_type!r}, "
            f"original_start={self.original_start}, original_end={self.original_end}, "
            f"score={self.score}, operator={self.operator!r})"
        )


@dataclass(frozen=True)
class TransformResult:
    text: str
    entities: tuple[TransformedEntity, ...]
    scope_id: str | None
    failed_detectors: tuple[str, ...] = ()

    def __repr__(self) -> str:  # textはマスク後でも部分原文を含み得るため件数のみ
        return (
            f"TransformResult(entities={len(self.entities)}, "
            f"scope_id={self.scope_id!r}, failed_detectors={self.failed_detectors!r})"
        )


@dataclass(frozen=True)
class RestoreResult:
    text: str
    unresolved: tuple[str, ...]
    restored: int = 0          # 戻した件数（v0.9。偽名の復元はパターンを持たないので unresolved には出ない）

    def __repr__(self) -> str:
        return f"RestoreResult(unresolved={len(self.unresolved)}, restored={self.restored})"


@dataclass(frozen=True)
class ExplainedEntity:
    """engine.explain() の 1 件。explanation は検出器名・スコア・位置・型だけで原文の文字を含まない。"""

    entity: Entity
    explanation: str

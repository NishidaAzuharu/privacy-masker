"""Policy（スペック§7）。未知typeはdefault operatorに落とす（素通しさせない）。"""
from __future__ import annotations

from typing import Mapping, Sequence

from .annotators.identifiability import PUBLIC_SUFFIX
from .exceptions import ConfigurationError
from .operators import Keep, Operator, Tokenize

# 識別性ベース匿名化（standard/minimal preset）で素通しするサブタイプの元の型
DEFAULT_PUBLIC_TYPES: tuple[str, ...] = ("PERSON", "ORG", "LOCATION", "FACILITY")


class Policy:
    def __init__(
        self,
        rules: Mapping[str, Operator] | None = None,
        default: Operator | None = None,
    ) -> None:
        self.rules: dict[str, Operator] = dict(rules or {})
        self.default: Operator = default if default is not None else Tokenize(strategy="sequential")

    def operator_for(self, entity_type: str) -> Operator:
        return self.rules.get(entity_type, self.default)

    def operators(self) -> list[Operator]:
        return [*self.rules.values(), self.default]

    def has_strategy(self, strategy: str | None) -> bool:
        # Tokenize / Surrogate（v0.9 で vault 対応）共通: strategy 属性を持つ operator を見る。
        # strategy=None は「どの operator にも当てはまらない」を意味させる: strategy 属性の無い
        # operator（Keep 等）を getattr(op, "strategy", None) が None として拾ってしまい、
        # has_strategy(None) が誤って True になるのを防ぐ
        return strategy is not None and any(getattr(op, "strategy", None) == strategy for op in self.operators())

    def tokenize_templates(self) -> set[str]:
        return {op.template for op in self.operators() if isinstance(op, Tokenize)}

    @classmethod
    def preset(
        cls, name: str, *, public_types: Sequence[str] = DEFAULT_PUBLIC_TYPES
    ) -> "Policy":
        if name in ("llm_input", "strict"):
            return cls(rules={}, default=Tokenize(strategy="sequential"))
        if name == "dataset":
            return cls(rules={}, default=Tokenize(strategy="hmac"))
        if name in ("standard", "minimal"):
            # 識別性ベース匿名化: IdentifiabilityAnnotator が付けた *_PUBLIC サブタイプだけを
            # 素通しする。私人・非公知・私人に帰属する準識別子はサブタイプが付かないため
            # default でトークン化される（fail-closed）。マスク方法を変えたい場合は rules に
            # PERSON / ORG / ADDRESS 等の operator を追加する
            rules = {f"{t}{PUBLIC_SUFFIX}": Keep() for t in public_types}
            return cls(rules=rules, default=Tokenize(strategy="sequential"))
        raise ConfigurationError(f"unknown policy preset: {name!r}")

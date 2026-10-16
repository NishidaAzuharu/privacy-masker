"""ContextEnhancer: 文脈語で検出スコアを上げる。

NER や正規表現の検出結果に対し、スパンの前後 window 字に型ごとの手掛かり語（氏名・担当・電話・住所 …）があれば
score を boost だけ上げる（上限 max_score）。`score_threshold` と組み合わせると「文脈のない低スコア検出だけを落とす」
運用ができる。スパンの位置・型は変えない。

使い方: PrivacyEngine(..., enhancer=ContextEnhancer())  # 既定の日本語手掛かり語
        ContextEnhancer(rules={"PERSON": ["氏名", "担当"], "PROJECT": ["案件"]}, boost=0.3, window=20)
"""
from __future__ import annotations

import dataclasses
import unicodedata
from typing import Iterable, Mapping, Sequence

from .exceptions import ConfigurationError
from .types import Entity

DEFAULT_CONTEXT_JA: dict[str, tuple[str, ...]] = {
    "PERSON": ("氏名", "名前", "担当", "様", "さん", "氏", "殿", "先生", "患者", "利用者", "申請者", "契約者", "被保険者", "宛", "御中"),
    "ADDRESS": ("住所", "所在地", "宛先", "お届け先", "送付先", "納品先", "現住所", "〒"),
    "LOCATION": ("在住", "出身", "所在", "地域"),
    "ORG": ("会社", "法人", "御中", "勤務先", "所属", "事業者", "取引先", "発注先"),
    "FACILITY": ("施設", "会場", "病院", "店", "窓口", "集合", "最寄り"),
    "PHONE_NUMBER": ("電話", "tel", "携帯", "fax", "連絡先", "℡", "直通", "内線"),
    "EMAIL": ("メール", "e-mail", "mail", "アドレス", "連絡先"),
    "POSTAL_CODE": ("〒", "郵便番号"),
    "DATE": ("生年月日", "誕生日", "生まれ", "出生"),
    "AGE": ("年齢", "満", "享年"),
    "USERNAME": ("id", "ユーザー", "アカウント", "ログイン", "担当"),
}


class ContextEnhancer:
    def __init__(
        self,
        rules: Mapping[str, Iterable[str]] | None = None,
        *,
        boost: float = 0.3,
        max_score: float = 1.0,
        window: int = 20,
        extend_default: bool = True,
    ) -> None:
        if not (0.0 <= boost <= 1.0) or not (0.0 < max_score <= 1.0) or window < 1:
            raise ConfigurationError("ContextEnhancer: boost in [0,1], max_score in (0,1], window >= 1")
        base: dict[str, tuple[str, ...]] = dict(DEFAULT_CONTEXT_JA) if extend_default or rules is None else {}
        for k, v in (rules or {}).items():
            base[k] = tuple(base.get(k, ())) + tuple(v) if extend_default else tuple(v)
        self.rules = {k: tuple(unicodedata.normalize("NFKC", w).casefold() for w in v) for k, v in base.items()}
        self.boost, self.max_score, self.window = boost, max_score, window

    def enhance(self, text: str, entities: Sequence[Entity]) -> list[Entity]:
        out: list[Entity] = []
        for e in entities:
            words = self.rules.get(e.entity_type)
            if not words or e.score >= self.max_score:
                out.append(e); continue
            lo, hi = max(0, e.start - self.window), min(len(text), e.end + self.window)
            ctx = unicodedata.normalize("NFKC", text[lo:e.start] + " " + text[e.end:hi]).casefold()
            if any(w in ctx for w in words):
                out.append(dataclasses.replace(e, score=min(self.max_score, e.score + self.boost)))
            else:
                out.append(e)
        return out

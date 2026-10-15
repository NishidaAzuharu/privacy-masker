"""検出スパンの境界リファインメント。

tokenizerの語彙結合（「日本の」「田中さん」が1トークン等）により、NERの検出スパンは
端に空白・敬称・助詞を巻き込むことがある。巻き込みは過剰マスク側（安全）だが、
(1) トークン化の同値判定が表層ゆれで壊れる（同一人物が別トークンになる）
(2) スパン境界が不正確になる
という副作用を持つ。本モジュールは非PII文字のみをスパン端から削ることで
これを補正する。削る対象が非PIIに限られるため、検出漏れ（危険側）は増えない。

既知の限界: 「田中さんの」のように敬称+助詞が連続し、かつ助詞のアンカーが
ひらがなになる場合は助詞側を削れない（安全側に倒れて残る）。
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

from .refiner_affix import AffixExpander
from .types import Entity

_DEFAULT_HONORIFICS = ("さん", "さま", "様", "氏", "くん", "ちゃん", "君", "殿")
_DEFAULT_PARTICLES = "のはがをにへとでも"
# 複数文字の機能語（スパン末尾に巻き込まれやすいもの。長い順に照合）
_DEFAULT_FUNCTION_WORDS = ("において", "における", "による", "にある", "から", "まで", "より", "など", "にて")
_MAX_PARTICLE_RUN = 2
_PUNCT_AFTER = "、。，．・「」（）()[]:：;；!！?？~〜\n\r\t 　"
_TITLES_AFTER_NAME = (
    "先生", "部長", "課長", "係長", "主任", "社長", "会長", "専務", "常務", "取締役", "教授", "医師", "看護師",
    "選手", "監督", "議員", "大臣", "首相", "知事", "市長", "院長", "所長", "店長", "局長", "室長", "班長",
    "主事", "主査", "参事", "理事", "顧問", "弁護士", "税理士", "宅", "邸", "家", "駅", "夫人", "夫妻", "親子",
)


def _is_hiragana(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ"


def _all_hiragana(s: str) -> bool:
    return bool(s) and all(_is_hiragana(ch) or ch == "ー" for ch in s)


class SpanRefiner:
    """スパン端の空白・敬称・助詞を削る。PrivacyEngineが検出直後に適用する。

    - trim_whitespace: スパン端の空白・改行を削る（全type）
    - honorifics: 末尾の敬称を削る（honorific_typesのtypeのみ。()で無効）
    - particles: 端の助詞runを削る。誤削り防止のため、削った後に残る端の文字が
      ひらがなの場合は削らない（「りの」のようなひらがな終わりの名前を守る）。
      ""で無効
    - function_words: 末尾の複数文字機能語（から・まで・にある等）を削る（全type、
      アンカー規則は助詞と同じ、累積1回まで）。Surrogateのような全体置換で
      機能語が消えて不自然になるのを防ぐ。()で無効
    """

    def __init__(
        self,
        trim_whitespace: bool = True,
        honorifics: Sequence[str] | None = None,
        honorific_types: Sequence[str] = ("PERSON",),
        particles: str | None = None,
        function_words: Sequence[str] | None = None,
        expand_affixes: bool = True,
        affix_expander: AffixExpander | None = None,
    ) -> None:
        # 種別語尾・法人格プレフィックスへの拡張（v0.8、refiner_affix.py）。
        # トリムの後に適用する。拡張のみで PII を落とさないため既定で有効
        self.expand_affixes = expand_affixes
        self._affix_expander = affix_expander if affix_expander is not None else AffixExpander()
        self.trim_whitespace = trim_whitespace
        self.honorifics = tuple(honorifics) if honorifics is not None else _DEFAULT_HONORIFICS
        self.honorific_types = frozenset(honorific_types)
        self.particles = _DEFAULT_PARTICLES if particles is None else particles
        self.function_words = tuple(sorted(
            _DEFAULT_FUNCTION_WORDS if function_words is None else function_words,
            key=len, reverse=True))

    def refine(self, text: str, entities: Iterable[Entity]) -> list[Entity]:
        out: list[Entity] = []
        for ent in entities:
            span = self._refine_span(text, ent.start, ent.end, ent.entity_type)
            if span is None:
                continue  # 空白のみ等、削り切って空になったスパンは破棄
            s, e = span
            out.append(ent if (s, e) == (ent.start, ent.end)
                       else dataclasses.replace(ent, start=s, end=e))
        if self.expand_affixes:
            out = self._affix_expander.expand(text, out)
        return out

    def _tail_blocked(self, text: str, e: int, entity_type: str) -> bool:
        """末尾の助詞を「名前の一部」と見て残すべきか。

        直後が文末・句読点・別の助詞なら、文法上そこに助詞は続かないので残す（型に関係ない判断）。
        「直後が敬称・肩書きだから残す」（佐藤まりも氏 の も）は **PERSON 固有の曖昧性**なので
        honorific_types に限る。組織・施設・地名・住所の名称が裸の助詞で終わることは無く、
        以前は「医療法人伊織会の理事長」の の が肩書き『理事』のせいで残っていた。
        なお「名前の一部か助詞か」は品詞では判別できない（佐藤まりも氏 の も と 田中は先生 の は は
        どちらも 助詞,係助詞 と解析される）ため、型で切り分けるのが唯一の一般的な手掛かりになる。
        """
        if e >= len(text):
            return True
        ch = text[e]
        if ch in _PUNCT_AFTER or ch in self.particles:
            return True
        if entity_type not in self.honorific_types:
            return False
        return any(text.startswith(h, e) for h in self.honorifics) or text.startswith(_TITLES_AFTER_NAME, e)

    def _refine_span(self, text: str, s: int, e: int, entity_type: str):
        # 助詞トリムはfixpointループを跨いだ累積でも片側 _MAX_PARTICLE_RUN 文字まで。
        # カスタムparticles設定でもスパン深部まで削れない（危険側への転落防止）。
        tail_budget = _MAX_PARTICLE_RUN
        head_budget = _MAX_PARTICLE_RUN
        fw_budget = 1
        changed = True
        while changed and s < e:
            changed = False
            if self.trim_whitespace:
                while s < e and text[s].isspace():
                    s += 1
                    changed = True
                while s < e and text[e - 1].isspace():
                    e -= 1
                    changed = True
            if entity_type in self.honorific_types:
                for h in self.honorifics:
                    if e - s > len(h) and text.startswith(h, e - len(h)):
                        e -= len(h)
                        changed = True
                        break
            if fw_budget and self.function_words:
                for fw in self.function_words:
                    if (e - s > len(fw) and text.startswith(fw, e - len(fw))
                            and not _is_hiragana(text[e - len(fw) - 1])):
                        e -= len(fw)
                        fw_budget -= 1
                        changed = True
                        break
            if self.particles:
                run = 0
                while (run < tail_budget and e - run - 1 > s
                       and text[e - run - 1] in self.particles):
                    run += 1
                # 直後が 文末・句読点・空白・別の助詞・敬称 なら、末尾の「助詞」は名前の一部
                # （篠原秋も氏、金子かの、石川まりの⏎）とみなして削らない
                if run and not _is_hiragana(text[e - run - 1]) and not self._tail_blocked(text, e, entity_type):
                    e -= run
                    tail_budget -= run
                    changed = True
                run = 0
                while (run < head_budget and s + run < e - 1
                       and text[s + run] in self.particles):
                    run += 1
                # ひらがなだけの名前（はすみ・はな・のぞみ）と、直前が助詞のとき（担当ははすみ）は先頭を削らない
                if (run and not _is_hiragana(text[s + run]) and not _all_hiragana(text[s:e])
                        and not (s > 0 and text[s - 1] in self.particles)):
                    s += run
                    head_budget -= run
                    changed = True
        if s >= e:
            return None
        return s, e

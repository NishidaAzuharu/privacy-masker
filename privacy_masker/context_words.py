"""文脈語の照合規則（RegexPattern.context と ContextEnhancer で共通）。

語と前後の窓は NFKC・casefold で正規化して比べる。英数字だけの語（T・tel・id）は前後に英数字が続かない
「単語」としてだけ一致させる（法人番号の文脈語 T が test や https の t に反応しない。tel が hotel に、
id が valid に反応しない）。日本語の語は分かち書きが無いので部分一致のまま。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_ASCII_WORD = re.compile(r"[a-z0-9]+")


def normalize_context(text: str) -> str:
    """語・窓の正規化（NFKC + casefold）。"""
    return unicodedata.normalize("NFKC", text).casefold()


def compile_context_words(words: Iterable[str]) -> "re.Pattern[str] | None":
    """文脈語を 1 本の正規表現にまとめる（normalize_context した窓に対して search する）。語が無ければ None。"""
    alts: list[str] = []
    for w in words:
        n = normalize_context(w)
        if not n:
            continue
        alts.append(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])" if _ASCII_WORD.fullmatch(n) else re.escape(n))
    return re.compile("|".join(alts)) if alts else None

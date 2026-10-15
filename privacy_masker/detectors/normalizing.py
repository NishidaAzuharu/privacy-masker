"""既存の検出器を包み、正規化したテキストで検出して結果を元位置に戻すラッパ。

字間空白（「千 葉 県 市 川 市 南 八 幡 2 - 1 - 9」）は token 検出器・文字ヘッド検出器の
**どちらでも**部分被覆になるため、検出器側ではなくこの層で直すのが安い。
``Engine`` は Detector を複数受け取れるので、既存の検出器コードを一切変えずに
``NormalizingDetector(TransformersNerDetector(...))`` と差し替えられる。

fail-closed の方針は 2 つある。

1. 正規化器が返した ``back_map`` が壊れていたら（長さ不一致・単調でない・範囲外・
   文字が対応しない）、黙って通さず ``DetectorError`` にする。ここを通してしまうと
   スパンが別の位置を指し、**PII がマスクされずに残る**。
2. 既定の ``mode="union"`` では、正規化で変化したテキストについて**元テキストでも**
   inner を呼び、両方の結果を足す。正規化すると検出が良くなる箇所がある一方で、
   稀に元テキストの方が広く取れる箇所（改行をまたぐ字間空白など）があり、実測で
   両方向の変化が出た。見逃しが最悪の失敗なので、前処理なしの結果を必ず含む側に倒す。
   追加の推論は**変化したテキストだけ**なので費用はほぼ増えない（盲検v1 は 246 件中 3 件）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from ..exceptions import ConfigurationError, DetectorError
from ..normalize import strip_letter_spacing
from ..types import Entity

# 正規化器の型: text -> (正規化テキスト, back_map)
Normalizer = Callable[[str], "tuple[str, Sequence[int]]"]

_MODES = ("union", "replace")


class NormalizingDetector:
    """``inner`` に正規化済みテキストを渡し、返ってきた Entity を元テキストの位置に戻す。

    ``name`` を省略すると ``inner`` の名前をそのまま使う。こうしないと
    ``Entity.source``（inner が入れる）と Engine 上の検出器名が食い違い、
    ``explain()`` や重複解決の見え方が変わってしまう。

    ``normalizer`` を差し替えるときの契約:
      - 文字の**削除のみ**（既定の ``verify_chars=True`` は文字の対応を検査する。
        NFKC のように文字を置換する正規化器を渡す場合は ``verify_chars=False``)
      - ``back_map[i]`` は正規化後 i 文字目に対応する元テキストの添字で、狭義単調増加

    ``min_run`` などを変えるときは ``functools.partial(strip_letter_spacing, min_run=5)``
    を ``normalizer`` に渡す。

    ``mode``:
      - ``"union"``（既定・安全側）正規化テキストの結果と元テキストの結果を足す。
        返す Entity 集合は前処理なしの結果を必ず包含する（そのあと Refiner や
        重複解決を通るので最終被覆までは保証できないが、盲検v1 + 固定OOD の
        両モデルで gold 被覆の悪化は 0 件だった）
      - ``"replace"`` 正規化テキストの結果だけを使う。推論は 1 回で済むが、
        元テキストの方が広く取れていた箇所を失い得る（実測で悪化 1〜2 件）
    """

    def __init__(
        self,
        inner,
        *,
        normalizer: Normalizer = strip_letter_spacing,
        name: str | None = None,
        verify_chars: bool = True,
        mode: str = "union",
    ) -> None:
        if inner is None or not callable(getattr(inner, "detect", None)):
            raise ConfigurationError("NormalizingDetector requires a detector with detect(text)")
        if not callable(normalizer):
            raise ConfigurationError("normalizer must be callable")
        if mode not in _MODES:
            raise ConfigurationError(f"mode must be one of {_MODES}")
        self.inner = inner
        self.normalizer = normalizer
        self.verify_chars = bool(verify_chars)
        self.mode = mode
        self.name = name or getattr(inner, "name", None) or type(inner).__name__

    # ------------------------------------------------------------------
    def _normalize(self, text: str):
        """(正規化テキスト, back_map) を返す。変化がなければ back_map は None。

        変化なしのときに None を返すのは、呼び出し側で inner を素通しさせるため
        （identity 写像のコピーとスパン変換を丸ごと省ける）。
        """
        norm, back = self.normalizer(text)
        if norm == text:
            return text, None
        self._validate_back_map(text, norm, back)
        return norm, back

    def _validate_back_map(self, text: str, norm: str, back: Sequence[int]) -> None:
        # メッセージに原文・検出値を入れない（スペック§12）
        try:
            length = len(back)
        except TypeError:
            raise DetectorError(f"normalizer of {self.name!r} returned a non-sequence back_map") from None
        if length != len(norm):
            raise DetectorError(f"normalizer of {self.name!r} returned a back_map of wrong length")
        n = len(text)
        prev = -1
        for i in range(length):
            o = back[i]
            if not isinstance(o, int) or isinstance(o, bool):
                raise DetectorError(f"normalizer of {self.name!r} returned a non-int back_map entry")
            if not (prev < o < n):  # prev=-1 始まりなので下限 0 と狭義単調増加を同時に見る
                raise DetectorError(f"normalizer of {self.name!r} returned a non-monotonic or out-of-range back_map")
            if self.verify_chars and text[o] != norm[i]:
                raise DetectorError(f"normalizer of {self.name!r} returned a back_map that does not match the text")
            prev = o

    def _remap(self, entities, norm: str, back: Sequence[int]) -> list[Entity]:
        """正規化後のスパン [a, b) を元テキストの [back[a], back[b-1]+1) に戻す。

        文字集合ではなく**連続区間**で戻すのが要点。集合として戻すと除去した
        字間空白がマスク対象から抜け、「千葉県…」が空白だらけの穴あきマスクになる。
        """
        out: list[Entity] = []
        n = len(norm)
        for e in entities:
            if not isinstance(e, Entity) or not (0 <= e.start < e.end <= n):
                raise DetectorError(f"detector {self.name!r} returned an invalid entity")
            out.append(replace(e, start=back[e.start], end=back[e.end - 1] + 1))
        return out

    @staticmethod
    def _merge(mapped: list[Entity], raw) -> list[Entity]:
        """正規化側と元テキスト側の結果を足す。完全に同じ (型, 位置) は 1 件にまとめる。

        同一スパンが両方から出たときは **スコアの高い方**を残す。正規化側を無条件に
        採ると、前処理なしより score が下がることがある（実測で最大 0.36 低下）。
        ``Engine`` の ``score_threshold`` を上げた構成では、それだけで前処理なしなら
        残っていた検出が消える＝漏れになるため、必ず max を採る。
        """
        best: dict[tuple[str, int, int], Entity] = {}
        order: list[tuple[str, int, int]] = []
        for e in list(mapped) + list(raw):
            key = (e.entity_type, e.start, e.end)
            prev = best.get(key)
            if prev is None:
                best[key] = e
                order.append(key)
            elif e.score > prev.score:
                best[key] = e
        return [best[k] for k in order]

    # ------------------------------------------------------------------
    def detect(self, text: str) -> list[Entity]:
        norm, back = self._normalize(text)
        if back is None:
            return list(self.inner.detect(text))
        mapped = self._remap(self.inner.detect(norm), norm, back)
        if self.mode == "replace":
            return mapped
        return self._merge(mapped, self.inner.detect(text))

    def _call_batch(self, inner_batch, payload: list[str]) -> list:
        results = list(inner_batch(payload))
        if len(results) != len(payload):
            raise DetectorError(
                f"detector {self.name!r} returned {len(results)} results for {len(payload)} texts"
            )
        return results

    def detect_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        texts = list(texts)
        inner_batch = getattr(self.inner, "detect_batch", None)
        if not callable(inner_batch):
            return [self.detect(t) for t in texts]  # inner が Detector だけでも動く
        changed: list[tuple[int, str, Sequence[int]]] = []
        payload: list[str] = []
        for i, t in enumerate(texts):
            norm, back = self._normalize(t)
            payload.append(t if self.mode == "union" or back is None else norm)
            if back is not None:
                changed.append((i, norm, back))
        # union のとき 1 本目は**元テキストそのまま**を渡す。バッチ構成（パディング）が
        # 変わるとバッチ仲間の推論がごく僅かに変わるため、前処理なしと bit 一致させる。
        # 正規化したテキストは変化した分だけを 2 本目のバッチで送る（費用はほぼ増えない）
        out: list[list[Entity]] = [list(f) for f in self._call_batch(inner_batch, payload)]
        if self.mode == "union" and changed:
            extra = self._call_batch(inner_batch, [norm for _, norm, _ in changed])
            for (i, norm, back), found in zip(changed, extra):
                out[i] = self._merge(self._remap(found, norm, back), out[i])
        elif self.mode == "replace":
            for i, norm, back in changed:
                out[i] = self._remap(out[i], norm, back)
        return out

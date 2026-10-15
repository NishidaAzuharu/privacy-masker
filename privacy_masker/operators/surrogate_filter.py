# privacy_masker/operators/surrogate_filter.py
"""Surrogate の候補に掛ける安全フィルタ（設計 §7）。

順に判定し、最初に当たった理由を返す。採用なら None。
- same_as_original: 正規化後に元と同一
- contains_original_part: 元の姓・名・コア語（2 字以上）を含む
- public_entity: 同梱の禁止リスト（生成器の候補全体 × Wikipedia 見出し語の衝突分）か、実行時に渡された PublicKnowledge にある
- present_in_document: 元の文書本文に既に出現している（検出漏れの実名や別の実体との同一化を防ぐ）
- denylist: 利用者の禁止語
"""
from __future__ import annotations

import os
import warnings
from array import array
from functools import lru_cache
from importlib import resources
from typing import Callable, Sequence

from ..annotators.knowledge import PublicKnowledge
from ..exceptions import ConfigurationError
from ..normalize import default_normalize

PACKAGED_DENYLIST_NAME = "surrogate_denylist.pmk"


@lru_cache(maxsize=1)
def _load_packaged_cached() -> PublicKnowledge:
    try:
        ref = resources.files("privacy_masker.data").joinpath(PACKAGED_DENYLIST_NAME)
        with resources.as_file(ref) as path:
            if not os.path.exists(path):
                raise FileNotFoundError(str(path))
            try:
                return PublicKnowledge.load(path)
            except ConfigurationError as exc:
                # ファイルは存在するが壊れている・読めない。ここは「無ければ空集合」の対象外
                # （fail-closed: 実体との衝突検査を黙って弱めるのではなく、壊れたインストールとして明示的に失敗させる）
                raise ConfigurationError(
                    f"packaged denylist {PACKAGED_DENYLIST_NAME} is corrupt or unreadable "
                    "(not a valid .pmk file); regenerate it with tools/build_surrogate_denylist.py"
                ) from exc
    except (FileNotFoundError, ModuleNotFoundError):
        warnings.warn(
            f"同梱の禁止リスト {PACKAGED_DENYLIST_NAME} が見つかりません。公知の実体との衝突検査は knowledge= を渡したときだけ働きます"
            "（tools/build_surrogate_denylist.py で生成できます）",
            UserWarning, stacklevel=3,
        )
        return PublicKnowledge(array("Q"))
    except OSError as exc:
        # missing (FileNotFoundError) 以外のOSError（例: 同名のディレクトリがあってIsADirectoryError、
        # 権限不足でPermissionError）を黙って空集合に縮退させず、明示的に失敗させる（fail-closed）
        raise ConfigurationError(
            f"packaged denylist {PACKAGED_DENYLIST_NAME} could not be read: {exc}"
        ) from exc


def load_packaged_denylist() -> PublicKnowledge:
    """パッケージ同梱の禁止リスト。

    ファイルが無ければ空の集合を返し、警告を 1 回（プロセスにつき）発する
    （``knowledge=`` を渡したときだけ公知実体の照合が働く縮退動作）。
    ファイルはあるが壊れている・``.pmk`` として読めない場合は、それを黙って空集合に
    縮退させず ``ConfigurationError`` を送出する（壊れたインストールとして明示的に失敗させる）。
    """
    return _load_packaged_cached()


class SurrogateFilter:
    def __init__(
        self,
        *,
        knowledge: PublicKnowledge | str | os.PathLike[str] | None = None,
        denylist: Sequence[str] = (),
        normalizer: Callable[[str], str] = default_normalize,
        packaged: PublicKnowledge | None = None,
    ) -> None:
        self._norm = normalizer
        self._packaged = packaged if packaged is not None else load_packaged_denylist()
        if knowledge is None or isinstance(knowledge, PublicKnowledge):
            self._knowledge = knowledge
        else:
            self._knowledge = PublicKnowledge.load(knowledge)
        self._deny = {self._norm(x) for x in denylist if x}
        self._text_cache: tuple[str, str] | None = None     # (元の文書, 正規化後)。_normalized_text 参照

    def _normalized_text(self, text: str) -> str:
        """文書全文の正規化を 1 文書につき 1 回にまとめる（設計 §7 の present_in_document 用）。

        照合は候補ごとに走るので、素直に書くと「候補の数 × 文書長」のコストが掛かる。
        キーはオブジェクト同一性（``is``）で、元の文字列への参照をキャッシュが持ち続けるため、
        解放済みオブジェクトの id を別の文字列が再利用して誤ヒットすることはない。
        等しいが別オブジェクトの文書は取りこぼすが、正規化をもう一度するだけで結果は同じ。
        スレッド安全ではない（Surrogate / Engine の他の状態と同じ前提）。
        """
        cache = self._text_cache
        if cache is not None and cache[0] is text:
            return cache[1]
        normalized = self._norm(text)
        self._text_cache = (text, normalized)
        return normalized

    def forget_document(self) -> None:
        """文書の処理が終わったら全文のメモを捨てる（長命の operator に原文を残さない）。"""
        self._text_cache = None

    def reject_reason(
        self,
        candidate: str,
        *,
        original_norm: str,
        avoid_parts: Sequence[str] = (),
        text: str | None = None,
        check_public: bool = True,
    ) -> str | None:
        c = self._norm(candidate)
        if not c or c == original_norm:
            return "same_as_original"
        for part in avoid_parts:
            # avoid_parts はローマ字を n の表記のまま返す（大小文字を保つ）。default_normalize は
            # 小文字化しないため、候補側の大小文字が n と揃うとは限らない。日本語は lower() で不変
            if len(part) >= 2 and part.lower() in c.lower():
                return "contains_original_part"
        if check_public:
            if candidate in self._packaged or c in self._packaged:
                return "public_entity"
            if self._knowledge is not None and (candidate in self._knowledge or c in self._knowledge):
                return "public_entity"
        # 候補・本文とも正規化してから照合する（空白入りの候補が空白なしの本文にも一致するように）
        if text and c in self._normalized_text(text):
            return "present_in_document"
        if c in self._deny:
            return "denylist"
        return None

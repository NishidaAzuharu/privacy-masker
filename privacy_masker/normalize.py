"""トークン化前の値正規化（スペック§7）。

NFKC + 空白の全除去。スペックの例（山田太郎 / 山田　太郎 の同一視）を満たすため、
空白は「単一化」ではなく全除去とする。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence

_WS = re.compile(r"\s+")


def default_normalize(value: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", value)).strip()


# 「和文 1 文字 + 空白 1 個」が min_run 回以上続く箇所だけに限定する。

# 区切りとして扱う空白。タブ・改行は文書構造なので**含めない**
# （字間空白は必ず同一行に現れるため、改行を消す必要がない）
_MIN_RUN_FLOOR = 3   # これ未満は姓名間の空白まで潰すので許さない
_MIN_DIGIT_RUN_FLOOR = 7  # 電話の最短（市外局番なし 10 桁）より下げると年号・金額を巻き込む
_SPACES = frozenset(" \u3000\u00a0")  # 半角空白・全角空白・NBSP

# 連鎖の途中に混ざってよい半角文字。番地「2 - 1 - 9」を住所本体と地続きで
# 拾うために必要。欧文文字を入れないのは "Kevin Marsh" を壊さないため
_FILLERS = frozenset("0123456789-−‐–—/")


def _is_ja_unit(ch: str) -> bool:
    """字間空白の「手がかり」になる和文 1 文字か。

    句読点・括弧（。「」・）を除くのは、「・ あ ・ い ・ う ・ え」のような
    箇条書きを連鎖と誤認して行をつなげてしまわないため。
    """
    if "\uff66" <= ch <= "\uff9f":  # 半角カタカナ（東アジア幅は "H" なので別扱い）
        return True
    if unicodedata.east_asian_width(ch) not in ("W", "F"):
        return False
    return unicodedata.category(ch)[0] in ("L", "N")


def _is_unit(ch: str) -> bool:
    """連鎖の要素になれる 1 文字か（和文 + 番地に使う半角）。"""
    return _is_ja_unit(ch) or ch in _FILLERS


def strip_letter_spacing(
    text: str, *, min_run: int = 4, min_digit_run: int = 8
) -> tuple[str, Sequence[int]]:
    """字間空白だけを除去し、(正規化テキスト, back_map) を返す。

    ``back_map[i]`` は正規化後 i 文字目が元テキストの何文字目かを表す。
    削除しかしないので back_map は狭義単調増加で、``text[back_map[i]] == out[i]``。

    除去するのは「1 文字 + 空白 1 個」が min_run 回以上連続し、かつその連鎖に
    和文 1 文字が min_run 個以上ある箇所の空白のみ。この 2 条件で以下を守る:

      - 空白 2 個以上は連鎖を切る → 表組みの桁揃え「　伊集院　　　薫」は触らない
      - 欧文は要素にしない → "Kevin Marsh" は触らない
      - 単発の空白は min_run に届かない → 姓名間の「金 東賢」は触らない
      - 和文が min_run 個未満なら除去しない → 「金 額  1 2 3 4 5 6 円」は触らない
        （ただし数字が min_digit_run 個以上あれば除去する。電話番号・カード番号・
          マイナンバーは字間空白のままだと正規表現も NER も引っ掛からず平文で漏れる）

    min_run を下げると誤って空白を消す危険が増えるので、既定より下げないこと。
    """
    if min_digit_run < _MIN_DIGIT_RUN_FLOOR:
        raise ValueError(
            f"min_digit_run must be >= {_MIN_DIGIT_RUN_FLOOR}"
            "（短い数値列を巻き込まないための下限）"
        )
    if min_run < _MIN_RUN_FLOOR:
        # min_run=1 だと「金 東賢」の姓名間空白まで潰れる（実測）。
        # 3 なら「1 文字 + 空白」が 3 回続く必要があり、通常の日本語文や
        # 姓名間の単発空白には掛からない。
        raise ValueError(f"min_run must be >= {_MIN_RUN_FLOOR}（既定の 4 より下げると誤除去が増える）")
    n = len(text)
    # 最短の対象は「(文字+空白)×min_run + 文字」= 2*min_run+1 文字
    if n < 2 * min(min_run, min_digit_run) + 1:
        return text, range(n)

    remove: list[int] = []
    i = 0
    while i < n:
        if not _is_unit(text[i]):
            i += 1
            continue
        # i から「1 文字 + 空白 1 個」の連鎖を最大まで伸ばす
        j = i
        spaces: list[int] = []
        ja = 1 if _is_ja_unit(text[i]) else 0
        digits = 1 if text[i].isdigit() else 0
        while j + 2 < n and text[j + 1] in _SPACES and _is_unit(text[j + 2]):
            spaces.append(j + 1)
            j += 2
            if _is_ja_unit(text[j]):
                ja += 1
            if text[j].isdigit():
                digits += 1
        # 極大連鎖が条件を満たさないなら、その部分連鎖も満たさない（各カウントは単調）
        #
        # 和文条件だけだと、数字だけの字間空白（電話番号・カード番号・マイナンバー）が
        # 素通しになる。これらは直接識別子で、字間空白のまま残ると正規表現検出も
        # NER も一切引っ掛からず**完全に平文で漏れる**（実測で確認）。そこで
        # 「数字が min_digit_run 個以上」も除去条件に加える。桁数で絞るのは、
        # 「金 額  1 2 3 4 5 6 円」のような短い数値列を巻き込まないため
        # （電話 10〜11 桁 / マイナンバー 12 桁 / カード 14〜16 桁はすべて 8 桁以上）。
        if len(spaces) >= min_run and (ja >= min_run or digits >= min_digit_run):
            remove.extend(spaces)
        i = j + 1

    if not remove:
        return text, range(n)  # 変化なしのときは identity を作らず range で返す（無駄なコピーを避ける）

    drop = set(remove)
    out: list[str] = []
    back: list[int] = []
    for idx, ch in enumerate(text):
        if idx in drop:
            continue
        out.append(ch)
        back.append(idx)
    return "".join(out), back

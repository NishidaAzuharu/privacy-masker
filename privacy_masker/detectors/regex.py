"""RegexDetector と日本語向けプリセット（スペック§5）。"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..context_words import compile_context_words, normalize_context
from ..types import Entity


@dataclass(frozen=True)
class RegexPattern:
    """正規表現 1 本。score は文脈語なしの信頼度、context の語が前後 context_window 字にあれば context_score に上げる。
    require_context=True なら文脈語が無いと出さない（7 桁の口座番号のように、数字列だけでは判別できないもの）。
    文脈語は NFKC・casefold で比べる。英数字だけの語（T・tel）は前後に英数字が続かない単語としてだけ一致し
    （法人番号の T が test や https に反応しない）、日本語の語は部分一致（privacy_masker.context_words）。"""

    entity_type: str
    pattern: "re.Pattern[str]"
    validator: Callable[[str], bool] | None = None
    score: float = 1.0
    context: tuple[str, ...] = ()
    context_score: float = 1.0
    context_window: int = 30
    require_context: bool = False
    _context_re: "re.Pattern[str] | None" = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_context_re", compile_context_words(self.context))


class RegexDetector:
    def __init__(self, patterns: Sequence[RegexPattern], name: str = "regex") -> None:
        self.patterns = list(patterns)
        self.name = name

    def detect(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        norm_cache: dict[tuple[int, int, int], str] = {}
        # 全角の英数字・記号を半角にした複製に対して照合する（長さは同じなので offset は原文と共通）。
        # 全角の電話番号・カード番号・旅券番号・日付・IP を、パターン側の対応なしで拾える
        haystack = narrow(text)
        for p in self.patterns:
            for m in p.pattern.finditer(haystack):
                if p.validator is not None and not p.validator(m.group(0)):
                    continue
                score = p.score
                if p._context_re is not None:
                    lo, hi = max(0, m.start() - p.context_window), min(len(text), m.end() + p.context_window)
                    key = (m.start(), m.end(), p.context_window)   # 窓はスパンと幅で決まる（一致部分は窓に含めない）
                    if key not in norm_cache:
                        norm_cache[key] = normalize_context(haystack[lo:m.start()] + " " + haystack[m.end():hi])
                    if p._context_re.search(norm_cache[key]):
                        score = p.context_score
                    elif p.require_context:
                        continue
                entities.append(Entity(p.entity_type, m.start(), m.end(), score, self.name))
        return entities

    @classmethod
    def default_ja(cls, *, dates: bool = False) -> "RegexDetector":
        """日本語向けの既定パターン。dates=True で DATE（年月日、和暦、生年月日文脈で 1.0）と AGE（○歳）も検出する。
        日付はほとんどの文書に大量に出る準識別子なので既定では入れない（日付・年齢も検出したい場合は True）。"""
        return cls(
            [
                RegexPattern("EMAIL", re.compile(
                    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
                # 国内電話（ハイフンは全角・半角・U+2010・U+2212 を許容。\d は全角数字にも一致する）
                RegexPattern("PHONE_NUMBER", re.compile(
                    rf"(?<![\d{_HY}])(?:[0０]\d{{1,4}}{_H}\d{{1,4}}{_H}\d{{3,4}}|[(（][0０]\d{{1,4}}[)）]\s?\d{{1,4}}{_H}?\d{{3,4}}|[0０]\d{{9,10}})(?![\d{_HY}])")),
                # 空白区切り・「の」区切り（読み上げ・書き起こし）: 桁数 10〜11 のものだけ
                RegexPattern("PHONE_NUMBER", re.compile(
                    rf"(?<![\d{_HY}])[0０]\d{{1,4}}(?:[ 　]\d{{1,4}}[ 　]|の\d{{1,4}}の)\d{{3,4}}(?![\d{_HY}])(?![ 　]\d)"), _domestic_phone_ok),
                # 国際番号 +81-3-1234-5678 / +44 7700 900123 / +886 912 345 678
                RegexPattern("PHONE_NUMBER", re.compile(
                    rf"(?<![\d{_HY}+])\+\d{{1,3}}(?:[ {_HY}]?\(0\))?(?:[ {_HY}]?\d{{1,4}}){{2,5}}(?![\d{_HY}])"), _intl_phone_ok),
                RegexPattern("POSTAL_CODE", re.compile(
                    rf"〒\s?\d{{3}}{_H}\d{{4}}|(?<![\d{_HY}])\d{{3}}{_H}\d{{4}}(?![\d{_HY}])")),
                # カード番号: 主要ブランドの先頭桁（_CARD_IIN）+ Luhn。URL のパスや型番の一部（/4111…・ID_4111…）は境界で弾く。
                # 検査数字が合うだけなら CHECKSUM_ONLY_SCORE、前後にカード・クレジット等の文脈語があれば 1.0（score_threshold で選べる）
                RegexPattern("CREDIT_CARD", re.compile(
                    rf"{_ID_LB}(?:\d{{13,19}}|\d{{4}}(?:[ {_HY}]\d{{4}}){{3}}|\d{{4}}[ {_HY}]\d{{6}}[ {_HY}]\d{{5}}){_ID_LA}"),
                    _credit_card_ok, score=CHECKSUM_ONLY_SCORE, context=_CARD_CONTEXT, context_score=1.0),
                # IPv4: 文末のピリオドは許容する（"from 198.51.100.201." で落とさない）
                RegexPattern("IP_ADDRESS", re.compile(
                    r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)(?!\.\d)"), _ipv4_ok),
                # 個人番号: 12 桁 + 検査用数字。文脈語（マイナンバー・個人番号 …）が無ければ CHECKSUM_ONLY_SCORE
                RegexPattern("MY_NUMBER", re.compile(rf"{_ID_LB}\d{{12}}{_ID_LA}"), _my_number_ok,
                             score=CHECKSUM_ONLY_SCORE, context=_MY_NUMBER_CONTEXT, context_score=1.0),
                # 4 桁区切りの個人番号 1234 5678 9018 / 1234-5678-9018（16 桁カード番号の一部は取らない）
                RegexPattern("MY_NUMBER", re.compile(
                    rf"{_ID_LB}(?<!\d[ 　])\d{{4}}[ 　{_HY}]\d{{4}}[ 　{_HY}]\d{{4}}{_ID_LA}(?![ 　]\d)"),
                    lambda v: _my_number_ok(_digits(v)), score=CHECKSUM_ONLY_SCORE, context=_MY_NUMBER_CONTEXT, context_score=1.0),
                # ---- v0.8: 日本固有の識別子（Azure PII / Google SDP の日本カバレッジに合わせた）----
                # 法人番号: 13 桁、先頭が検査用数字。登録番号の T1234567890123（T 直結）は許し、ST… のような英字直結は弾く。
                # カード番号（Luhn）と衝突したら EntityResolver の優先順位で CREDIT_CARD が勝つ
                RegexPattern("CORPORATE_NUMBER", re.compile(
                    rf"(?<![\dA-SU-Za-z_/{_HY}])(?<![\dA-Za-z]T)(?<!\d\.)\d{{13}}{_ID_LA}"), _corporate_number_ok,
                    score=CHECKSUM_ONLY_SCORE, context=("法人番号", "法人", "登記", "登録番号", "T"), context_score=1.0),
                # 運転免許証番号: 12 桁（マイナンバーと同じ桁数なので文脈必須。検査数字の公式仕様は非公開）
                RegexPattern("DRIVERS_LICENSE", re.compile(rf"(?<![\d{_HY}])\d{{12}}(?![\d{_HY}])"), None,
                             score=0.0, context=("免許", "運転免許", "免許証"), context_score=0.9, require_context=True),
                # 旅券番号: 英字 2 + 数字 7（TK1234567）
                RegexPattern("PASSPORT", re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{7}(?![A-Za-z0-9])"), None,
                             score=0.6, context=("旅券", "パスポート", "passport"), context_score=1.0),
                # 在留カード番号: 英字 2 + 数字 8 + 英字 2（AB12345678CD）
                RegexPattern("RESIDENCE_CARD", re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{8}[A-Z]{2}(?![A-Za-z0-9])"), None,
                             score=0.85, context=("在留", "在留カード", "特別永住者"), context_score=1.0),
                # 基礎年金番号: 4 桁 + 6 桁（1234-567890）。文脈必須
                RegexPattern("PENSION_NUMBER", re.compile(rf"(?<![\d{_HY}])\d{{4}}[ 　{_HY}]\d{{6}}(?![\d{_HY}])"), None,
                             score=0.0, context=("年金", "基礎年金番号", "年金手帳"), context_score=1.0, require_context=True),
                # 口座番号: 7 桁。文脈必須（口座・普通・当座・支店・振込先）
                RegexPattern("BANK_ACCOUNT", re.compile(rf"(?<![\d{_HY}])\d{{7}}(?![\d{_HY}])"), None,
                             score=0.0, context=("口座", "普通", "当座", "貯蓄", "支店", "振込先", "振込"), context_score=0.9, require_context=True),
                # ---- v0.8: 汎用型 ----
                RegexPattern("URL", re.compile(r"(?<![\w@.])(?:https?://|www\.)[^\s<>\"'（）「」【】]+")),
                # IPv6（コロン区切り。時刻 12:30:45 は ipaddress の検証で落ちる）
                RegexPattern("IP_ADDRESS", re.compile(r"(?<![\w:.])(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:.])"), _ipv6_ok),
                RegexPattern("MAC_ADDRESS", re.compile(r"(?<![\w:.-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:.-])")),
            ]
            + (_date_age_patterns() if dates else [])
        )


_HY = "\\-－‐−"          # 文字クラス内で使うハイフン類（半角・全角・U+2010・U+2212）

# 全角→半角の 1 対 1 写像。**必ず長さが変わらない**ものだけを入れる（NFKC は ㍿→株式会社 のように
# 長さが変わるため使えない）。detect() はこの写像をかけた複製に対して照合し、offset は原文にそのまま使う。
# これにより「全角対応のために各パターンへ全角の文字クラスを足す」場当たり対応が不要になる。
def _build_wide_to_narrow() -> dict[int, str]:
    """全角→半角の 1 対 1 写像を文字コードから組み立てる（手書きの対応表は数え間違えるので使わない）。"""
    pairs: list[tuple[str, str]] = []
    pairs += [(chr(0xFF10 + i), str(i)) for i in range(10)]                    # ０-９
    pairs += [(chr(0xFF21 + i), chr(ord("A") + i)) for i in range(26)]         # Ａ-Ｚ
    pairs += [(chr(0xFF41 + i), chr(ord("a") + i)) for i in range(26)]         # ａ-ｚ
    pairs += [("－", "-"), ("―", "-"), ("‐", "-"), ("−", "-"),                  # ハイフン類
              ("／", "/"), ("．", "."), ("＠", "@"), ("：", ":"), ("＋", "+"), ("　", " ")]
    assert all(len(w) == 1 and len(n) == 1 for w, n in pairs)                  # 長さ保存の担保
    return {ord(w): n for w, n in pairs}


_WIDE_TO_NARROW = _build_wide_to_narrow()


def narrow(text: str) -> str:
    """全角の英数字・記号を半角にした複製を返す。長さは原文と必ず一致する。"""
    out = text.translate(_WIDE_TO_NARROW)
    # 写像に長さの変わる要素が混入したら、offset がずれるので原文をそのまま使う（防御的）
    return out if len(out) == len(text) else text
_H = f"[{_HY}]"

# 番号類（カード番号・個人番号・法人番号）の境界。数字・英字・_・/・ハイフンが直結するもの（URL のパス、ID_…、型番）と
# 小数（0.4111…・…1234.5）は番号としない。文末のピリオド（…1234.）や No.4111… は許す
_ID_LB = rf"(?<![\dA-Za-z_/{_HY}])(?<!\d\.)"
_ID_LA = rf"(?![\dA-Za-z_/{_HY}])(?!\.[\dA-Za-z])"

CHECKSUM_ONLY_SCORE = 0.5
"""検査数字は合うが文脈語の無い数字列（カード番号・個人番号・法人番号）のスコア。既定の閾値 0 ではマスクされ、
score_threshold を 0.6 などにすると文脈語のあるもの（1.0）だけが残る。"""

_CARD_CONTEXT = ("カード", "クレジット", "card", "credit", "visa", "mastercard", "jcb", "amex", "american express",
                 "diners", "discover", "有効期限", "セキュリティコード", "cvv", "cvc")
_MY_NUMBER_CONTEXT = ("マイナンバー", "個人番号", "通知カード", "my number")


def _digits(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def _date_age_patterns() -> list[RegexPattern]:
    birth = ("生年月日", "生まれ", "誕生日", "生年", "出生", "birth")
    return [
        RegexPattern("DATE", re.compile(r"(?<!\d)(?:19|20)\d{2}\s?年\s?(?:0?[1-9]|1[0-2])\s?月\s?(?:0?[1-9]|[12]\d|3[01])\s?日"),
                     None, score=0.5, context=birth, context_score=1.0),
        RegexPattern("DATE", re.compile(r"(?<![\d.])(?:19|20)\d{2}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])(?![\d.])"),
                     None, score=0.5, context=birth, context_score=1.0),
        RegexPattern("DATE", re.compile(r"(?:令和|平成|昭和|大正|明治|[RHSTM])\s?(?:元|\d{1,2})\s?年\s?(?:0?[1-9]|1[0-2])\s?月\s?(?:0?[1-9]|[12]\d|3[01])\s?日"),
                     None, score=0.5, context=birth, context_score=1.0),
        RegexPattern("AGE", re.compile(r"(?<![\d０-９])(?:満\s?)?(?:[1-9]|[1-9]\d|1[01]\d)\s?歳(?![\d０-９])"),
                     None, score=0.5, context=("年齢", "満", "生まれ", "享年"), context_score=1.0),
    ]


def corporate_check_digit(body12: str) -> int:
    """法人番号の検査用数字。body12 は下 12 桁。P_n = 右から n 桁目、Q_n = n が奇数なら 1・偶数なら 2、9 - (ΣP_n×Q_n mod 9)。"""
    total = sum(int(ch) * (1 if n % 2 == 1 else 2) for n, ch in enumerate(reversed(body12), start=1))
    return 9 - (total % 9)


def _corporate_number_ok(value: str) -> bool:
    d = _digits(value)
    return len(d) == 13 and int(d[0]) == corporate_check_digit(d[1:])


def _ipv6_ok(value: str) -> bool:
    try:
        ipaddress.IPv6Address(value)
        return True
    except ValueError:
        return False


def _domestic_phone_ok(value: str) -> bool:
    return 10 <= len(_digits(value)) <= 11


def _intl_phone_ok(value: str) -> bool:
    """E.164 は最大 15 桁。国番号込みで 9 桁未満は電話番号として短すぎる（"+3 45 678" 等の誤検出を除く）。"""
    return 9 <= len(_digits(value)) <= 15


def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# 主要ブランドの先頭桁（IIN）。これ以外で始まる Luhn 一致の数字列（IMEI、7 や 9 で始まる伝票番号など）はカード番号としない
_CARD_IIN = re.compile(
    r"4"                                              # Visa
    r"|5\d"                                           # Mastercard 51–55、Maestro 50・56–59
    r"|6\d"                                           # Discover 6011・65・644–649、銀聯 62、Maestro 60–69
    r"|3[47]"                                         # American Express
    r"|35(?:2[89]|[3-8]\d)"                           # JCB 3528–3589
    r"|30[0-5]|3[689]"                                # Diners Club
    r"|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)"   # Mastercard 2221–2720
)


def _credit_card_ok(value: str) -> bool:
    return _CARD_IIN.match(_digits(value)) is not None and _luhn_ok(value)


def _ipv4_ok(value: str) -> bool:
    return all(0 <= int(part) <= 255 for part in value.split("."))


def _my_number_ok(value: str) -> bool:
    """個人番号のチェックデジット検証（行政手続における特定の個人を識別するための
    番号の利用等に関する法律の検査用数字算式）。"""
    if len(value) != 12 or not value.isdigit():
        return False
    digits = [int(c) for c in value]
    check = digits[11]
    p = digits[:11][::-1]  # P1..P11（検査用数字を除く11桁、最下位がP1）
    total = 0
    for n in range(1, 12):
        q = n + 1 if n <= 6 else n - 5
        total += p[n - 1] * q
    rem = total % 11
    expected = 0 if rem <= 1 else 11 - rem
    return check == expected

# privacy_masker/operators/semantic.py
"""意味を保つ Surrogate のカテゴリ判定と表層特徴（設計 §4・§5）。

置換先が元の値に依存するのは、ここで決めるカテゴリ（粗い種別）と表層特徴（文字種・構造・法人格・粒度・地方ブロック）
を通してだけ。``Features`` の文字列フィールドはすべて閉じた語彙で、元の値の文字は入らない（生成器に元の値を渡さないための型）。
"""
from __future__ import annotations

import re
import types
from dataclasses import dataclass
from typing import Callable, Mapping

from ..data.ja_pools import (
    COUNTRIES, COUNTRIES_EXTRA, COUNTRY_ALIASES, COUNTRY_CANONICAL, DEPT_FUNCTIONS, FOREIGN_CITIES_KANJI, FOREIGN_CITIES_KATA,
    GIVEN_HIRA, GIVEN_KANJI_F, GIVEN_KANJI_M, GIVEN_KATA, GIVEN_KATA_JA, GIVEN_ROMAJI, KATA_CORES, ORG_CORES,
    PREFECTURES, PREF_CITIES, PREF_CITIES_EXTRA, REGION_BLOCKS, SURNAMES_HIRA, SURNAMES_KANJI, SURNAMES_KATA_JA,
    SURNAMES_ROMAJI, TOWNS,
)

ORG_CATEGORIES = ("manufacturing", "it", "finance", "retail", "construction", "transport", "service",
                  "medical_corp", "school_corp", "government", "association", "media", "party", "department")
FACILITY_CATEGORIES = ("medical", "education", "transport", "commerce", "public", "building", "culture", "other")
LOCATION_CATEGORIES = ("country", "prefecture", "city", "town", "district", "foreign", "nature", "other")
CATEGORIES: dict[str, tuple[str, ...]] = {
    "person": ("person",), "org": ORG_CATEGORIES, "facility": FACILITY_CATEGORIES,
    "location": LOCATION_CATEGORIES, "address": ("address",),
}
OTHER: dict[str, str] = {"person": "person", "org": "service", "facility": "other", "location": "other", "address": "address"}

# 法人格（長いものから照合する）。Features.legal_form はこの語彙の値だけ
LEGAL_FORMS = ("特定非営利活動法人", "一般社団法人", "公益社団法人", "一般財団法人", "公益財団法人", "社会福祉法人",
               "医療法人社団", "医療法人", "学校法人", "株式会社", "合同会社", "有限会社", "(株)", "（株）")
PERSON_SCRIPTS = ("kanji", "kata", "hira", "romaji", "halfwidth", "latin_abbr")
STRUCTURES = ("full", "sei", "mono")
GENDER_HINTS = ("m", "f", "unknown")
LENGTH_CLASSES = ("short", "medium", "long")

_KATA_RE = re.compile(r"[ァ-ヶー・]")
_HIRA_RE = re.compile(r"[ぁ-ゖー]")
_KANJI_RE = re.compile(r"[一-龥々]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_HALFWIDTH_RE = re.compile(r"[ｦ-ﾟ]")
_DIGIT_RE = re.compile(r"[0-9０-９]")
_FULLWIDTH_DIGIT_RE = re.compile(r"[０-９]")
HAS_BUILDING_RE = re.compile(
    r"(号室|号館|階|[FＦ]\d|\d[FＦ]|ビル|マンション|ハイツ|コーポ|メゾン|レジデンス|ハウス|タワー|"
    r"パーク|ヒルズ|テラス|アパート|団地|荘|寮|棟|[A-ZＡ-Ｚ]棟)"
)
_SURNAME_SET = frozenset(SURNAMES_KANJI)
_SURNAME_KATA_JA_SET = frozenset(SURNAMES_KATA_JA)
# avoid_parts の区切り無しカタカナ外国人名の分割（最長前方一致）用。長いものから照合する
_GIVEN_KATA_LONGEST_FIRST = tuple(sorted(GIVEN_KATA, key=len, reverse=True))

# 性別: 偶奇規約（プールの偶数番目 = 男性寄り、奇数番目 = 女性寄り） + 明示の上書き。
# 読むのは import 時の _parity_sets() だけなので、後から書き換えても判定に効かない誤解を避けるため読み取り専用にする
GENDER_OVERRIDES: Mapping[str, str] = types.MappingProxyType(
    {"カイト": "m", "かいと": "m", "Kaito": "m"}                                    # 奇数番目（女性側）にある男性名
)


def _parity_sets() -> tuple[frozenset[str], frozenset[str]]:
    m, f = set(GIVEN_KANJI_M), set(GIVEN_KANJI_F)
    for pool in (GIVEN_HIRA, GIVEN_KATA_JA, GIVEN_ROMAJI):
        m.update(pool[0::2]); f.update(pool[1::2])
    for name, g in GENDER_OVERRIDES.items():
        (m if g == "m" else f).add(name); (f if g == "m" else m).discard(name)
    return frozenset(m), frozenset(f)


_GIVEN_M, _GIVEN_F = _parity_sets()
# PREF_CITIES_EXTRA（北海道の追加市。semantic 経路専用）も判定・地方ブロック引きの対象に含める。
# hmac の既存出力に影響する PREF_CITIES 自体（生成器の選択プール）は変えない
_CITY_SET = frozenset(c for _, c in PREF_CITIES + PREF_CITIES_EXTRA)
_PREF_OF_CITY = {c: p for p, c in PREF_CITIES + PREF_CITIES_EXTRA}
_BLOCK_OF_PREF = {p: b for b, ps in REGION_BLOCKS.items() for p in ps}
_CHOME_KANJI_RE = re.compile(r"[一二三四五六七八九十]丁目")
# LOCATION 判定用の追加の語彙集合（RuleCategorizer.categorize 専用。生成プールの COUNTRIES/FOREIGN_CITIES_KATA/TOWNS は不変）
_COUNTRY_SET = frozenset(COUNTRIES) | frozenset(COUNTRIES_EXTRA) | frozenset(COUNTRY_ALIASES)
_TOWN_SET = frozenset(TOWNS)
_FOREIGN_KANJI_SET = frozenset(FOREIGN_CITIES_KANJI)

_HW_PAIRS = (("ガ", "ｶﾞ"), ("ギ", "ｷﾞ"), ("グ", "ｸﾞ"), ("ゲ", "ｹﾞ"), ("ゴ", "ｺﾞ"), ("ザ", "ｻﾞ"), ("ジ", "ｼﾞ"), ("ズ", "ｽﾞ"),
             ("ゼ", "ｾﾞ"), ("ゾ", "ｿﾞ"), ("ダ", "ﾀﾞ"), ("ヂ", "ﾁﾞ"), ("ヅ", "ﾂﾞ"), ("デ", "ﾃﾞ"), ("ド", "ﾄﾞ"), ("バ", "ﾊﾞ"),
             ("ビ", "ﾋﾞ"), ("ブ", "ﾌﾞ"), ("ベ", "ﾍﾞ"), ("ボ", "ﾎﾞ"), ("パ", "ﾊﾟ"), ("ピ", "ﾋﾟ"), ("プ", "ﾌﾟ"), ("ペ", "ﾍﾟ"),
             ("ポ", "ﾎﾟ"), ("ヴ", "ｳﾞ"))
_HW_SINGLE = dict(zip("ァアィイゥウェエォオカキクケコサシスセソタチッツテトナニヌネノハヒフヘホマミムメモャヤュユョヨラリルレロワヲンー・",
                      "ｧｱｨｲｩｳｪｴｫｵｶｷｸｹｺｻｼｽｾｿﾀﾁｯﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓｬﾔｭﾕｮﾖﾗﾘﾙﾚﾛﾜｦﾝｰ･"))
_HW_PAIR_MAP = dict(_HW_PAIRS)


def to_halfwidth_kana(s: str) -> str:
    """全角カタカナを半角カナに（濁点・半濁点は 2 文字）。カタカナ以外はそのまま。"""
    return "".join(_HW_PAIR_MAP.get(ch) or _HW_SINGLE.get(ch, ch) for ch in s)


@dataclass(frozen=True)
class Features:
    """生成器に渡す表層特徴。**元の値の文字列は持たない**（閉じた語彙のみ）。"""

    kind: str                              # person / org / facility / location / address
    category: str                          # CATEGORIES[kind] のいずれか
    script: str                            # PERSON_SCRIPTS のいずれか
    length_class: str                      # short / medium / long
    # PERSON
    structure: str | None = None           # full / sei / mono（姓だけ・名だけ・あだ名など 1 要素）
    gender_hint: str | None = None         # m / f / unknown（名のプール照合。姓には使わない）
    kata_foreign: bool = False             # カタカナの外国人名か
    separator: str = ""                    # "" / " " / "・"
    # ORG
    legal_form: str | None = None          # LEGAL_FORMS のいずれか
    legal_form_pos: str | None = None      # pre / post
    # FACILITY
    org_prefix: str | None = None          # 市立 / 県立 / 私立 / 町立
    # LOCATION / ADDRESS
    granularity: str | None = None         # LOCATION: カテゴリと同じ。ADDRESS: pref / city / town / banchi / building
    region_block: str | None = None        # REGION_BLOCKS のキー
    keep: str | None = None                # ADDRESS の "city"
    has_building: bool = False
    chome_style: bool = False
    chome_kanji: bool = False              # 丁目が漢数字（一丁目）か
    fullwidth_digits: bool = False


def _script_of(norm: str, raw: str) -> str:
    if _HALFWIDTH_RE.search(raw):
        return "halfwidth"
    n_kanji = len(_KANJI_RE.findall(norm))
    n_kata = len(_KATA_RE.findall(norm))
    n_hira = len(_HIRA_RE.findall(norm))
    n_latin = len(_LATIN_RE.findall(norm))
    if n_kanji:
        return "kanji"
    if n_latin and not n_kata and not n_hira:
        return "latin_abbr" if norm.isupper() and len(norm) <= 5 else "romaji"
    if n_kata >= n_hira and n_kata:
        return "kata"
    if n_hira:
        return "hira"
    return "kanji"


def _length_class(norm: str) -> str:
    return "short" if len(norm) <= 3 else ("medium" if len(norm) <= 8 else "long")


def region_block_of(name: str) -> str | None:
    """都道府県名（前方一致）か PREF_CITIES の市区町村名から地方ブロックを引く。決まらなければ None。"""
    for pref in PREFECTURES:
        if name.startswith(pref):
            return _BLOCK_OF_PREF[pref]
    for city, pref in _PREF_OF_CITY.items():
        if name.startswith(city) or city.startswith(name):
            return _BLOCK_OF_PREF[pref]
    return None


# ADDRESS の粒度判定・avoid_parts と、surrogate.py の keep="city" の分解に使う。
# 市区町村止まりの入力にも一致させたいため rest は空も許す（.+ ではなく .*）。
# city 群は「◯◯町」という町名にも一致してしまうので、本当に市区町村かどうかは
# is_municipality() で別に判定する（keep="city" で実在の町名が残っていた欠陥の修正）
_ADDR_CITY_ONLY_RE = re.compile(
    r"^(?P<city>(?:東京都|北海道|京都府|大阪府|.{2,3}県)?(?:.{1,6}?郡)?.{1,8}?[市区町村](?:.{1,4}?区)?)"
    r"(?P<rest>.*)$"
)
# 名前の途中に 市/区/町/村 を含む市。_ADDR_CITY_ONLY_RE の非貪欲一致は途中の文字（四日市市の 2 字目の「市」）で
# 切れて末尾の「市」を rest 側に残すため、address_city_and_town がこの語彙に当たるときだけ 1 字延長する。
# 語彙を使わずに「直後が 市/区 なら延長」とすると、「市谷加賀町」「市場町」「区役所前」のように
# 市/区 で始まる町名の先頭を市区町村名に取り込んでしまう（最終レビュー C2）
MUNICIPALITIES_WITH_INNER_SHI: frozenset[str] = frozenset(
    {"四日市市", "廿日市市", "野々市市", "十日町市", "大町市", "大村市", "羽村市", "田村市"}
)


def address_granularity(n: str) -> str:
    """入力住所の粒度（pref/city/town/banchi/building）。偽住所はこの粒度までしか生成しない
    （元の情報量を超えて番地・町名を捏造すると、そのぶん SurrogateFilter の avoid_parts で
    弾けない「余計な一致」が増えるため）。"""
    if HAS_BUILDING_RE.search(n):
        return "building"
    if _DIGIT_RE.search(n):
        return "banchi"
    if n in PREFECTURES:
        return "pref"
    rest = n
    for pref in PREFECTURES:
        if n.startswith(pref) and len(n) > len(pref):
            rest = n[len(pref):]
            break
    if (n and n[-1] in "市区町村") or rest in _CITY_SET:
        return "city"
    return "town"


def address_city_and_town(n: str) -> tuple[str | None, str | None]:
    """(都道府県込みの市区町村, 番地・建物より手前の町名) を抽出する。市区町村が特定できなければ
    (None, None)。avoid_parts（生成された偽住所に元の市区町村・町名が混入するのを防ぐ）と、
    surrogate.py の keep="city" の分解で使う。city が本当に市区町村かは is_municipality() で別に判定する。"""
    m = _ADDR_CITY_ONLY_RE.match(n)
    if not m:
        return None, None
    city = m.group("city")
    rest = m.group("rest")
    # 「四日市市」のように市区町村名の途中に 市/区/町/村 を含むと非貪欲一致が 1 字手前で切れる。
    # MUNICIPALITIES_WITH_INNER_SHI の名前で終わるようになるときだけ 1 字取り込む（市谷加賀町などの町名は取り込まない）。
    # 語彙の文字列末尾一致だけでは境界を見ないため、「野田村市場」のように別の町名の内部にたまたま
    # 「田村市」が現れると誤って取り込んでいた（最終レビューの残件2）。語彙の直前が文字列の先頭、または
    # 都/道/府/県/郡 の1字のときだけ延長する
    if rest:
        cand = city + rest[0]
        for m2 in MUNICIPALITIES_WITH_INNER_SHI:
            if cand.endswith(m2):
                head = cand[: -len(m2)]
                if head == "" or head[-1] in "都道府県郡":
                    city, rest = cand, rest[1:]
                break
    d = _DIGIT_RE.search(rest)
    b = HAS_BUILDING_RE.search(rest)
    ends = [x.start() for x in (d, b) if x]
    town = rest[: min(ends)] if ends else rest
    return city, town


def is_municipality(city: str) -> bool:
    """``address_city_and_town`` が返した前置部分が、本当に（都道府県＋）市区町村かどうか。

    分解に使う正規表現の city 群 ``.{1,8}?[市区町村]`` は「本町」「栄町」のような町名にも一致する。
    次のどれかを満たすものだけ市区町村とみなし、それ以外は町名として扱う
    （keep="city" はこのとき原文を一切残さず偽住所全体に落とす）:

    - PREF_CITIES の市区町村名そのもの
    - 「市」か「区」で**終わる**（「市」「区」を含むだけでは足りない。「市場町」のような町名が
      市区町村に化けてしまうため）
    - 都道府県で始まり、そこで終わらない（郡・町・村が都道府県の下に続く形。例「北海道上川郡東川町」）
    """
    if not city:
        return False
    if city in _CITY_SET:
        return True
    if city.endswith(("市", "区")):
        return True
    return any(city.startswith(pref) and len(city) > len(pref) for pref in PREFECTURES)


def split_legal_form(norm: str) -> tuple[str | None, str | None, str]:
    """(法人格, 位置, 残り)。法人格が無ければ (None, None, norm)。"""
    for form in LEGAL_FORMS:
        if norm.startswith(form) and len(norm) > len(form):
            return form, "pre", norm[len(form):]
    for form in LEGAL_FORMS:
        if norm.endswith(form) and len(norm) > len(form):
            return form, "post", norm[: -len(form)]
    return None, None, norm


def split_person(norm: str) -> tuple[str, str] | None:
    """漢字フルネームを (姓, 名) に分割。姓プールの最長前方一致。分割できなければ None。"""
    for ln in (3, 2, 1):
        if len(norm) > ln and norm[:ln] in _SURNAME_SET:
            return norm[:ln], norm[ln:]
    return None


class RuleCategorizer:
    """種別語・法人格・粒度による決定的なカテゴリ判定。決まらなければ OTHER[kind]。"""

    _ORG_WORDS = (
        ("medical_corp", re.compile(r"^(医療法人|社会福祉法人)")),
        ("school_corp", re.compile(r"^学校法人|(学園|学院)$")),
        # 公的機関の語は業種語より先に見る（航空自衛隊・交通局・航空局 が transport にならないように）。
        # 局 は 放送局・テレビ局・ラジオ局（media）、郵便局・支局・薬局・事務局・編集局（いずれも service 相当）を除く。
        # 軍 は陸軍・海軍・空軍・国防軍・解放軍・連合軍だけを公的機関として扱い、裸の「軍」は含めない
        # （「将軍」「救世軍」「巨人軍」のような固有名まで拾ってしまうのを防ぐ。巨人軍は media の球団語で拾う）
        ("government", re.compile(
            r"(省|庁|(?<!放送)(?<!郵便)(?<!支)(?<!テレビ)(?<!ラジオ)(?<!薬)(?<!事務)(?<!編集)局|"
            r"自衛隊|陸軍|海軍|空軍|国防軍|解放軍|連合軍|警察署|警察|消防署|税務署|保健所|裁判所|議会|"
            r"県庁|市役所|区役所|町役場|村役場|公社|事業団|独立行政法人|機構)$")),
        # 公的な委員会: 国・都道府県・市区町村・制度の名を伴うものだけ。それ以外の委員会は department（表の後ろ）。
        # 市区町村側は特定の語彙（PREFECTURES/_CITY_SET）に頼らず、「…都/道/府/県/市/区/町/村…委員会」という
        # 形だけで判定する（_CITY_SET は区レベルの名前しか持たず、架空の市区町村名を拾えないため）。
        # fix round 1 で省・州 を一度足したが、fix round 2 のレビューで「社内省エネ推進委員会」「九州支社安全委員会」
        # 等を誤って拾う（しかも government の偽名生成器が ORG_CORES を使うため地名語幹が漏れる）ことが分かり、
        # 代替なしで撤回した（外国の省・州の委員会は department のまま。YAGNI）。
        # 「党中央委員会」のように政党の内部機関でも、中央（国の意）を公的扱いにする設計上の割り切り。
        # 修飾語は委員会に直接隣接するものだけ（fix round 1 で「間に最大 6 字まで許す」形にしたところ、
        # 「労働安全衛生委員会」「中央執行委員会」等の私的な委員会まで government になり、しかも一致位置が
        # 文字列先頭になるため avoid_parts/core_split がコアを切り出せず地名語幹が漏れる欠陥があり、fix round 2
        # で隣接に戻した。「連邦」だけは複合語（連邦通信・連邦準備・連邦取引・連邦選挙・連邦捜査）に限定し、
        # 裸の「連邦」がどこにでも委員会を伴って government になるのを防ぐ。最終修正波 C3: 当初は複合部分を
        # 省略可能（`?`）にしていたため裸の「連邦委員会」も government になっていた。`?` を外して複合語だけにした）
        ("government", re.compile(
            r"(国家|中央|選挙管理|公正取引|教育|公安|人事|原子力規制|国家公安|労働|農業|監査|都市計画|"
            r"欧州|国連|証券取引|運輸安全|連邦(?:通信|準備|取引|選挙|捜査))委員会$"
            r"|(^.{1,6}(都|道|府|県|市|区|町|村)[^ ]{0,10}委員会$)")),
        ("media", re.compile(r"(新聞社|新聞|放送局|放送|出版|テレビ|通信社|(?<!電気)通信|ラジオ|テレビ局|ラジオ局|巨人軍|球団|"
                             r"ホークス|ジャイアンツ|タイガース|ドラゴンズ|スワローズ|ベイスターズ|カープ|マリーンズ|ライオンズ|バファローズ|ファイターズ|イーグルス)$")),
        ("party", re.compile(r"(党|政治連盟|政治協会)$")),
        ("association", re.compile(r"(協会|連盟|組合|財団|研究会|振興会|学会|NPO法人|連合会|の会)$")),
        ("department", re.compile(r"^[^株合有]{1,8}(部|課|室|チーム|係|委員会)$")),
        ("transport", re.compile(r"(運輸|物流|交通|商船|海運|航空|鉄道|電鉄|電気鉄道|電気軌道|バス|エクスプレス|倉庫)")),
        ("manufacturing", re.compile(r"(電機|電子|製作所|工業|化学|精機|精工|金属|製薬|薬品|食品|製造|重工|鉄鋼|機械)")),
        ("it", re.compile(r"(システム|テクノロジー|ソフト|ネットワーク|デジタル|データ|コンピュータ|IT|電気通信|テレコム)")),
        ("finance", re.compile(r"(銀行|信用金庫|信金|信用組合|証券|保険|ファイナンス|リース|カード|キャピタル)")),
        ("retail", re.compile(r"(商店|ストア|マート|百貨店|フーズ|物産|販売|ドラッグ|スーパー)")),
        ("construction", re.compile(r"(建設|工務店|不動産|ハウジング|建築|土木|住宅)")),
    )
    # カテゴリには使わない、core_split 専用の語彙（商事・ホーム・電気・グループ・サービス等は
    # _ORG_WORDS からカテゴリ語として外した = categorize/type_word_span には出てこないが、
    # それらを末尾に持つ名前の avoid_parts がコアを切り出せず名前全体しか避けられなくなっていた
    # （実測: 若葉商事 18/300・青葉ホーム 22/300・東雲電気 24/300、既存の若葉サービス 16/300・
    # 北斗グループ 16/300 も同じ欠陥）。_ORG_WORDS に「末尾で一致し開始位置が 2 以上」の行が無いときだけ
    # core_split が追加で見る。categorize / type_word_span からは参照しない（カテゴリ中立）。
    # 最終修正波 A2: 委員会 も足した。自治体接頭の委員会（government の `^.{1,6}(都|…|村)…委員会$`）と
    # department の `^[^株合有]{1,8}(…|委員会)$` は文字列先頭から一致するため開始位置が 0 になり、
    # core_split がコアを切り出せず名前全体しか避けられなかった（「営業委員会」→ 基点 ('営業',)、修正前 ('営業委員会',)。
    # department の偽名生成器が「営業」を再利用 46/400）。委員会 はここでは core_split 専用で、カテゴリは変えない
    _ORG_CORE_SUFFIXES = re.compile(r"(商事|商会|ホーム|電気|グループ|サービス|興業|産業|事業|委員会)$")
    _FAC_WORDS = (
        ("medical", re.compile(r"(病院|クリニック|医院|診療所|内科|外科|歯科|眼科|皮膚科|小児科|薬局|医療センター)$")),
        ("education", re.compile(r"(大学|高校|高等学校|中学校|小学校|幼稚園|保育園|保育所|学園|学院|塾|専門学校)$")),
        ("transport", re.compile(r"(駅|空港|港|IC|インターチェンジ|バスターミナル|ターミナル)$")),
        ("commerce", re.compile(r"(ホテル|旅館|ストア|モール|マート|食堂|レストラン|カフェ|百貨店|スーパー|店)$")),
        ("public", re.compile(r"(市役所|区役所|町役場|役所|図書館|公民館|文化会館|市民会館|体育館|公園|神社|寺|署|裁判所|保健所)$")),
        ("building", re.compile(r"(ビル|マンション|タワー|ハイツ|コーポ|メゾン|レジデンス|団地|スクエア|ヒルズ|棟)")),
        ("culture", re.compile(r"(美術館|博物館|温泉|水族館|記念館|遊園地|動物園|劇場|音楽堂)$")),
    )
    _LOC_CITY = re.compile(r"^.{1,8}(市|区|町|村)$")
    _LOC_TOWN = re.compile(r"(丁目|番地|番|号)$")
    _LOC_DISTRICT = re.compile(r"(地区|駅前|商店街|繁華街|界隈)$")
    _LOC_ELECTORAL = re.compile(r"[0-9０-９]+区$")
    _LOC_FOREIGN_KATA_ADMIN = re.compile(r"^[ァ-ヶー・]+(市|町|州|省)$")
    _LOC_NATURE = re.compile(r"(山|川|湖|岬|島|峠|高原|湾|海|峡|滝|岳)$")
    # fix round 1: 共和国・帝国・合衆国・連邦 で終わる架空の国名 → country（王国・公国は検証で foreign 判定が多く足さない）
    _LOC_COUNTRY_SUFFIX = re.compile(r"(共和国|帝国|合衆国|連邦)$")

    @classmethod
    def categorize(cls, kind: str, norm: str) -> str:
        n = norm.strip()
        if kind == "org":
            for cat, rx in cls._ORG_WORDS:
                if rx.search(n):
                    return cat
            return OTHER["org"]
        if kind == "facility":
            for cat, rx in cls._FAC_WORDS:
                if rx.search(n):
                    return cat
            return OTHER["facility"]
        if kind == "location":
            if n in _COUNTRY_SET or n == "日本" or cls._LOC_COUNTRY_SUFFIX.search(n):
                return "country"
            if n in PREFECTURES:
                return "prefecture"
            if cls._LOC_DISTRICT.search(n) or cls._LOC_ELECTORAL.search(n):      # 〜地区・選挙区 は city より先
                return "district"
            if n in FOREIGN_CITIES_KATA or n in _FOREIGN_KANJI_SET or cls._LOC_FOREIGN_KATA_ADMIN.match(n):
                return "foreign"
            if n in _CITY_SET or cls._LOC_CITY.match(n):
                return "city"
            if cls._LOC_TOWN.search(n) or _DIGIT_RE.search(n):
                return "town"
            if _KATA_RE.search(n) and not _KANJI_RE.search(n):
                return "foreign"
            m = cls._LOC_NATURE.search(n)
            if m and m.start() >= 2 and n not in _CITY_SET and n not in _TOWN_SET:
                return "nature"
            return OTHER["location"]
        return OTHER[kind]

    @classmethod
    def type_word_span(cls, kind: str, norm: str) -> tuple[int, int] | None:
        """categorize() が判定に使う種別語の (start, end)。表の順で最初に一致した語（末尾かどうかは問わない）で、
        categorize() と同じ語を指す。位置は ``norm.strip()`` 上（先頭に空白があれば norm 上の位置とずれる）。
        org / facility 以外は None。"""
        n = norm.strip()
        words = cls._ORG_WORDS if kind == "org" else cls._FAC_WORDS if kind == "facility" else ()
        for _cat, rx in words:
            m = rx.search(n)
            if m:
                return m.start(), m.end()
        return None

    @classmethod
    def core_split(cls, kind: str, norm: str) -> int | None:
        """avoid_parts がコア語を切り出す位置。種別語の表を順に見て、各表の最初の一致が ``norm`` の**末尾**で終わり、
        かつ開始位置が 2 以上になった最初の語の開始位置（``norm`` そのものの上。strip しない）。org はさらに、
        末尾一致かつ開始位置 2 以上の行が無いときだけ ``_ORG_CORE_SUFFIXES``（カテゴリ中立。categorize /
        type_word_span からは参照しない）を同じ条件で試す（先頭から一致する委員会・部署の行はここで「委員会」を
        切り出す）。どちらにも無ければ None（org / facility 以外も None）。

        ``type_word_span`` とは別物: 「青葉交通データ」では type_word_span が categorize と同じ「交通」
        (2, 4、transport) を返すのに対し、core_split は末尾の「データ」の 4 を返す（コア語は「青葉交通」）。
        type_word_span をコア語の切り出しに使うと名前全体しか避けられず、コア語が偽名に混入しうる（最終レビュー C1）。"""
        words = cls._ORG_WORDS if kind == "org" else cls._FAC_WORDS if kind == "facility" else ()
        for _cat, rx in words:
            m = rx.search(norm)
            if m and m.end() == len(norm) and m.start() >= 2:
                return m.start()
        if kind == "org":
            m = cls._ORG_CORE_SUFFIXES.search(norm)
            if m and m.end() == len(norm) and m.start() >= 2:
                return m.start()
        return None


def given_gender(name: str) -> str:
    """名の性別傾向。生成器（surrogate_generators._given）と判定（gender_hint）が同じ表を使う。"""
    if name in _GIVEN_M:
        return "m"
    if name in _GIVEN_F:
        return "f"
    return "unknown"


def gender_hint(given: str | None) -> str:
    return given_gender(given) if given else "unknown"


def extract_features(kind: str, value: str, norm: str, category: str, *, keep: str | None = None) -> Features:
    raw = value.strip()
    n = norm.strip()
    script = _script_of(n, raw)
    length = _length_class(n)
    if kind == "person":
        compact = n.replace(" ", "")
        sep = " " if (" " in raw or "\u3000" in raw) else ("・" if "・" in n else "")
        # 区切り記号（半角/全角スペース・中黒）で raw を分割した要素。default_normalize は空白を
        # 全除去するため n では区切りが失われる（空白区切りのカナ氏名は raw から分割する）
        sep_parts = [p for p in re.split(r"[ \u3000・]", raw) if p] if sep else []
        # 先頭要素が既知の日本語カナ姓なら「姓→名」の日本語順、そうでなければ
        # 「名→姓」の西洋順（外国人名）とみなす
        is_known_ja_kata_name = bool(sep_parts) and sep_parts[0] in _SURNAME_KATA_JA_SET
        kata_foreign = script == "kata" and not is_known_ja_kata_name and ("・" in n or len(compact) >= 5)
        structure, gender = "mono", "unknown"
        if script == "kanji":
            if len(compact) <= 3 and not sep:
                structure = "sei"                      # 3 字以下・区切りなしは姓のみ扱い（v0.3 からの規約）
            else:
                structure = "full"
                split = split_person(compact)
                gender = gender_hint(split[1]) if split else "unknown"
        elif script in ("romaji", "latin_abbr"):
            # default_normalize は空白を全除去するため n には区切りが残らない。
            # 区切りの有無・語数は raw（元の空白）から判定する
            parts = raw.split()
            structure = "full" if len(parts) >= 2 else "mono"
            gender = gender_hint(parts[0]) if parts else "unknown"
            sep = " " if len(parts) >= 2 else ""
        elif sep:
            structure = "full"
            # kata_foreign なら西洋順（名が先頭）、日本語順なら名は最後の要素
            given = sep_parts[0] if kata_foreign else sep_parts[-1]
            gender = gender_hint(given) if sep_parts else "unknown"
        elif kata_foreign:
            structure = "full"          # 「マイケルジョンソン」: 区切り無しでも名+姓とみなす（公知照合の対象になる）
        return Features(kind, "person", script, length, structure=structure, gender_hint=gender,
                        kata_foreign=kata_foreign, separator=sep)
    if kind == "org":
        form, pos, rest = split_legal_form(n)
        return Features(kind, category, _script_of(rest, raw), length, legal_form=form, legal_form_pos=pos)
    if kind == "facility":
        org_prefix = next((p for p in ("市立", "県立", "私立", "町立") if n.startswith(p)), None)
        return Features(kind, category, script, length, org_prefix=org_prefix)
    if kind == "location":
        return Features(kind, category, script, length, granularity=category, region_block=region_block_of(n),
                        chome_style="丁目" in n, chome_kanji=bool(_CHOME_KANJI_RE.search(n)))
    if kind == "address":
        gran = address_granularity(n)
        return Features(kind, "address", script, length, granularity=gran, region_block=region_block_of(n), keep=keep,
                        has_building=bool(HAS_BUILDING_RE.search(n)), chome_style="丁目" in n,
                        chome_kanji=bool(_CHOME_KANJI_RE.search(n)),
                        fullwidth_digits=bool(_FULLWIDTH_DIGIT_RE.search(raw)))
    raise ValueError(f"unknown kind: {kind!r}")


_LOCATION_SUFFIX = frozenset("都道府県市区町村州省")      # fix round 1（レビュー I2）: 州・省 を追加


def _split_kata_person(n: str) -> tuple[str, ...]:
    """カタカナ人名の要素。「・」区切りならその要素、区切り無しなら GIVEN_KATA の最長前方一致と残り
    （「マイケルジョンソン」→「マイケル」「ジョンソン」）。分けられなければ ()。"""
    if "・" in n:
        return tuple(p for p in n.split("・") if p)
    for given in _GIVEN_KATA_LONGEST_FIRST:
        if len(n) > len(given) and n.startswith(given):
            return given, n[len(given):]
    return ()


_PERSON_POOL_WORDS: tuple[str, ...] = tuple(w for w in SURNAMES_KATA_JA + GIVEN_KATA_JA + SURNAMES_HIRA + GIVEN_HIRA if len(w) >= 2)
_PERSON_POOL_WORDS_ROMAJI: tuple[str, ...] = tuple(w.lower() for w in SURNAMES_ROMAJI + GIVEN_ROMAJI if len(w) >= 2)


def _pool_parts(n: str) -> tuple[str, ...]:
    """n の前方または後方に一致する 2 字以上のプール語（生成器が再利用しうるのはプール語だけ）。

    default_normalize は区切り（空白）を全除去するため、区切り付きの日本語順カナ氏名・平仮名氏名・
    ローマ字氏名は split_person（漢字専用）でも _split_kata_person（区切り無し外国人名専用）でも
    姓・名に分割できず、avoid_parts が名前全体しか返せなかった（最終レビューの残件1）。姓・名のプール語
    そのものが n の前方・後方に一致するときだけ、それを追加の要素として返す。"""
    found = []
    low = n.lower()
    for w in _PERSON_POOL_WORDS:
        if n.startswith(w) or n.endswith(w):
            found.append(w)
    for w in _PERSON_POOL_WORDS_ROMAJI:
        if low.startswith(w) or low.endswith(w):
            found.append(n[: len(w)] if low.startswith(w) else n[-len(w):])
    return tuple(dict.fromkeys(found))


# fix round 3: ORG/FACILITY/LOCATION の偽名生成器（surrogate_generators）が固有部分に使う語彙
# （ORG_CORES・KATA_CORES は org/facility のコア、TOWNS は location の town/district/nature）。
# core_split が抽出する「コア」は委員会・議会等の種別語の前の文字列をひとかたまりでしか扱わないため、
# その内部にこの語彙の単語（若葉・青葉・汐見 等）が部分文字列として埋もれていても個別には避けられず、
# 生成器が独立にその単語を引いた偽名が元の値の一部をそのまま再利用してしまっていた
# （例: 「中央青葉人事委員会」→ コアは「中央青葉」のみ avoid、内部の「青葉」は avoid されない）。
# 長い語から照合し（同じ長さは辞書順。順序を決定的にする）、2 文字未満の語（"曙"・"扇" 等）は対象外にする
_FAKE_CORE_VOCAB: tuple[str, ...] = tuple(sorted({w for w in ORG_CORES + KATA_CORES + TOWNS if len(w) >= 2}, key=lambda w: (-len(w), w)))
# 最終修正波: ORG だけはさらに department の偽名生成器（DEPT_FUNCTIONS × DEPT_UNITS）の機能名も語彙に入れる。
# department の規則 `^[^株合有]{1,8}(部|…|委員会)$` は先頭から一致するため、部・課・室・チーム・係で終わる名前は core_split が
# コアを切り出せず名前全体しか避けられず、生成器が元の機能名を引いた偽名が通っていた（基点から: 400 scope 中 営業部 24・
# 人事部 36。この枝で私的な委員会が department に移り、営業推進委員会 0 → 42）。DEPT_FUNCTIONS の語は FACILITY・LOCATION の
# 生成器の語彙には現れないため、facility / location の語彙は変えない
_ORG_FAKE_VOCAB: tuple[str, ...] = tuple(sorted(set(_FAKE_CORE_VOCAB) | {w for w in DEPT_FUNCTIONS if len(w) >= 2}, key=lambda w: (-len(w), w)))


def _vocab_parts(n: str, vocab: tuple[str, ...] = _FAKE_CORE_VOCAB) -> tuple[str, ...]:
    """n に部分文字列として含まれる vocab（既定 _FAKE_CORE_VOCAB）の語（長さ降順・同長は辞書順、重複なし）。"""
    return tuple(dict.fromkeys(w for w in vocab if w in n))


def avoid_parts(kind: str, norm: str) -> tuple[str, ...]:
    """偽名に含めてはいけない元の部分（2 字以上）。人名は姓と名（漢字は姓プールの最長前方一致、カタカナは「・」区切りの
    要素か GIVEN_KATA の最長前方一致と残り。分けられなければ名前全体）、組織・施設はコア語、住所は都道府県より下の
    市区町村・町名（都道府県は同一ブロックの別の市区町村なら再出現してよいため対象外）。LOCATION は地名そのものと、
    末尾が「都道府県市区町村州省」のいずれか1文字ならそれを除いた語幹（例: 「札幌市」→「札幌市」「札幌」、
    「東京都」→「東京都」「東京」、「ハルカゼ州」→「ハルカゼ州」「ハルカゼ」。「パリ」のように末尾がそれ以外ならそのまま
    1 件、全体で 2 字未満なら空）に加え、fix round 1（レビュー Important I1・I2・Minor M1）で 3 つ拡張した:
    地区・選挙区・自然地名の接尾語を外した語幹（「若葉地区」→「若葉」、「青葉1区」→「青葉」、「八雲山」→「八雲」）、
    国名の別名・略称・前身国家は同じ国の標準名 = 生成プール COUNTRIES の項目（`COUNTRY_CANONICAL`、「米国」→「アメリカ」）。
    fix round 2（N1）でさらに、n に含まれる COUNTRIES の項目（n 自身を除く）も避けるようにした
    （`_LOC_COUNTRY_SUFFIX` で country になった正式名「ロシア連邦」→「ロシア」、COUNTRIES の項目を部分文字列に
    持つ国名「北アイルランド」→「アイルランド」、「インドネシア」→「インド」）。最終修正波 A1 で、n に含まれる
    COUNTRY_CANONICAL の別名（n 自身は除く）の標準名も避けるようにした（「ソ連邦」→「ロシア」、「大韓帝国」→「韓国」）。
    いずれも避ける部分を増やす方向にだけ拡張しており、既存の組み合わせ（地名そのもの・都道府県市区町村の語幹）は変えない。
    同一ブロックの別の地名への再利用そのものは妨げない（別の市区町村名を出すのは許容する）。
    fix round 3: ORG・FACILITY・LOCATION では、生成器の固有部分の語彙（ORG_CORES・KATA_CORES・TOWNS）のうち
    元の値に含まれる語も避ける（抽出したコアの内部に埋もれていて個別には避けられなかった語を救う）。
    最終修正波: ORG ではさらに department の偽名生成器の機能名（DEPT_FUNCTIONS の 2 字以上の語。「営業部」→「営業」）も避ける。"""
    n = norm.replace(" ", "")
    if kind == "person":
        split = split_person(n)
        if split is not None:
            existing_parts = tuple(p for p in split if len(p) >= 2)
        elif _script_of(n, n) == "kata":
            # 姓プールが漢字だけなので split_person はカタカナ名を分けられない。全体だけを避けると元の名・姓
            # （「ジョン」）を再利用した偽名が通ってしまう（最終レビュー推奨 2）。要素はどれも n の部分文字列なので、
            # 要素を避ければ全体も避けられる。2 字以上の要素が残らなければ従来どおり全体を避ける
            kata_parts = tuple(dict.fromkeys(p for p in _split_kata_person(n) if len(p) >= 2))
            existing_parts = kata_parts if kata_parts else ((n,) if len(n) >= 2 else ())
        else:
            existing_parts = (n,) if len(n) >= 2 else ()
        # 残件1: 区切り付きの日本語順カナ氏名・平仮名氏名・ローマ字氏名は default_normalize が区切りを
        # 全除去するため上のどの分岐でも分割できず、名前全体しか避けられない。プール語の前方一致・後方
        # 一致を追加し、「・」の混入（漢字の「山田・太郎」等）を落として 2 字未満を捨てる
        return tuple(dict.fromkeys(
            p.strip("・") for p in (*existing_parts, *_pool_parts(n)) if len(p.strip("・")) >= 2
        ))
    if kind in ("org", "facility"):
        _form, _pos, rest = split_legal_form(n)
        core = rest
        # 末尾の種別語で切る（type_word_span は末尾かどうかを問わない最初の語なので使わない。core_split 参照）
        start = RuleCategorizer.core_split(kind, rest)
        if start is not None:
            core = rest[:start]
        parts = [core] if len(core) >= 2 and core != n else ([n] if len(n) >= 2 else [])
        # fix round 3: core_split が切り出すコアは委員会・議会等の種別語の前をひとかたまりでしか扱わないため、
        # その内部（または市区町村接尾込みのコア末尾）に生成器の固有部分の語彙（ORG_CORES・KATA_CORES）が
        # 部分文字列として埋もれていても個別には避けられなかった（「中央青葉人事委員会」→ コア「中央青葉」の
        # 内部の「青葉」、「汐見町議会」→ コア「汐見町」の内部の「汐見」）。rest に含まれる語彙語も避ける
        # 最終修正波: ORG は department の機能名（DEPT_FUNCTIONS）も避ける（_ORG_FAKE_VOCAB の注釈）
        parts.extend(w for w in _vocab_parts(rest, _ORG_FAKE_VOCAB if kind == "org" else _FAKE_CORE_VOCAB) if w not in parts)
        seen: set[str] = set()
        return tuple(p for p in parts if not (p in seen or seen.add(p)))
    if kind == "address":
        city, town = address_city_and_town(n)
        return tuple(p for p in (city, town) if p and len(p) >= 2)
    if kind == "location":
        parts = []
        if len(n) >= 2:
            parts.append(n)
        if n and n[-1] in _LOCATION_SUFFIX:
            stem = n[:-1]
            if len(stem) >= 2:
                parts.append(stem)
        # fix round 1（Minor M1）: 地区・選挙区・自然地名の接尾語を外した語幹（2 字以上）も避ける
        # （若葉地区 → 若葉、青葉1区 → 青葉、八雲山 → 八雲）
        for rx in (RuleCategorizer._LOC_ELECTORAL, RuleCategorizer._LOC_DISTRICT, RuleCategorizer._LOC_NATURE):
            m = rx.search(n)
            if m and m.end() == len(n) and m.start() >= 2:
                parts.append(n[:m.start()])
        # fix round 1（Important I1）: 国名の別名・略称・前身国家は、同じ国の標準名（偽名のプール COUNTRIES の項目）も避ける
        canon = COUNTRY_CANONICAL.get(n)
        if canon:
            parts.append(canon)
        # fix round 2（N1 再発: _LOC_COUNTRY_SUFFIX で country になった正式名「ロシア連邦」「フランス共和国」等や、
        # COUNTRIES の項目をそのまま含む国名「北アイルランド」「インドネシア」は COUNTRY_CANONICAL に無いため上の
        # canon 参照だけでは避けられず、COUNTRIES の短い国名が偽名にそのまま出ていた。n に含まれる COUNTRIES の
        # 項目（n 自身は除く）はすべて避ける）
        parts.extend(c for c in COUNTRIES if c != n and c in n)
        # 最終修正波 A1: 別名に 連邦・帝国・合衆国・共和国 が付いて country になった名前（「ソ連邦」「大韓帝国」
        # 「米合衆国」）は COUNTRY_CANONICAL にも COUNTRIES にも完全一致しないため、同じ国の標準名が偽名に出ていた
        # （400 scope 中: ソ連邦→ロシア 6、大韓帝国→韓国 7、米合衆国→アメリカ 6）。n に含まれる別名（n 自身は除く）の
        # 標準名も避ける（単漢字の略称「米」「中」等にも当たるため、避ける語は増える方向にだけ広がる）
        parts.extend(c for k, c in COUNTRY_CANONICAL.items() if k != n and k in n)
        # fix round 3: TOWNS（生成器が district/nature の固有部分にも使う語彙）のうち n に含まれる語も避ける
        parts.extend(_vocab_parts(n))
        seen: set[str] = set()
        return tuple(p for p in parts if not (p in seen or seen.add(p)))
    return ()


class CategoryResolver:
    """規則 → 分類ヘッド（entity.vector があれば。計画 B） → 退避カテゴリ。決定的。

    ``min_margin`` は ``float``（全 kind 共通）か、kind をキーにした ``Mapping``
    （``HeadCategorizer.min_margin`` のような、型ごとに学習時に選ばれた推奨値）のどちらも受け付ける。
    ``Mapping`` に kind が無ければ既定の 0.15 にフォールバックする。
    """

    def __init__(
        self,
        head: Callable[[str, bytes], tuple[str, float]] | None = None,
        min_margin: float | Mapping[str, float] = 0.15,
    ) -> None:
        self.head = head
        self.min_margin = min_margin if isinstance(min_margin, Mapping) else float(min_margin)

    def resolve(self, kind: str, norm: str, vector: bytes | None) -> str:
        cat = RuleCategorizer.categorize(kind, norm)
        if kind in ("person", "address") or cat != OTHER[kind]:
            return cat
        if self.head is not None and vector is not None:
            head_cat, margin = self.head(kind, vector)
            mm = self.min_margin.get(kind, 0.15) if isinstance(self.min_margin, Mapping) else self.min_margin
            if margin >= mm and head_cat in CATEGORIES[kind]:
                return head_cat
        return OTHER[kind]

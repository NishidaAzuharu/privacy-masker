"""IdentifiabilityAnnotator: 私人を識別する情報だけを匿名化対象に残す（design_identifiability.md）。

匿名化対象 = 私人名、私人に帰属する準識別子（最寄り駅・勤務先など）、番地以下の住所、連絡先。
公人名と公知の地名・施設・組織は ``_PUBLIC`` サブタイプに書き換え、Policy 側で Keep できるようにする。

判定は**正の証拠があるときだけ**公人・公知とする（fail-closed）:

- PERSON が公人: (1) 直後に公職肩書き（総理・首相・大臣・知事・議員…）が隣接する。裸姓でも成立。
  企業肩書き（社長・部長）や「氏」「さん」は証拠にしない。(2) 表層が 4 文字以上または区切り
  （空白・中黒）を含むフルネームで、公知集合に一致する。裸姓は集合照合の対象外
  （見出し語の約半数が姓のページと衝突するため）
- LOCATION / FACILITY / ORG が公知: 表層が公知集合に一致し、かつ私人に帰属していない。
  4 文字以下の英字略称（NDS 等）は一致しても公知としない。
  「文書内に私人がいなければ残す」という消極的な判定は行わない
  （NER の型誤りをそのまま漏れに変えるため）
- 私人への帰属: 私人 PERSON の前後 ``window`` 文字以内にあり、近傍に手掛かり語（最寄り・在住・
  勤務先…）または帳票ラベル（``住所:`` 等）があるとき。帰属していれば公知でも匿名化する

レベル: ``strict`` は何も書き換えない（現行互換）。``standard`` は上記。``minimal`` は PERSON のみ
判定し、LOCATION / FACILITY / ORG はすべて ``_PUBLIC`` にする（直接識別子のみ匿名化）。
"""
from __future__ import annotations

import dataclasses
import re
from typing import Container, Sequence

from ..exceptions import ConfigurationError
from ..refiner_affix import DEFAULT_PREFIXES
from ..types import Entity
from .knowledge import normalize_surface

# 公知データ照合時に外してみる法人格（「トヨタ自動車株式会社」→「トヨタ自動車」で見出し語に一致させる）
_LEGAL_FORMS = tuple(sorted(
    {p for p in DEFAULT_PREFIXES if p.endswith(("法人", "会社"))} | {"（株）", "(株)", "（有）", "(有)"},
    key=len, reverse=True))

LEVELS = ("strict", "standard", "minimal")
PUBLIC_SUFFIX = "_PUBLIC"

# 公職・元首・立法府の肩書き。企業肩書き（社長・部長・代表・会長）や学術肩書きは含めない
DEFAULT_PUBLIC_TITLES: tuple[str, ...] = (
    "内閣総理大臣", "総理大臣", "総理", "首相", "副総理", "官房長官", "副長官", "長官",
    "大臣", "副大臣", "政務官", "外相", "防衛相", "財務相", "法相", "厚労相", "経産相", "文科相",
    "知事", "副知事", "市長", "区長", "町長", "村長", "議員", "議長", "副議長",
    "党首", "総裁", "幹事長", "政調会長", "総務会長", "国対委員長",
    "大統領", "副大統領", "首脳", "国王", "女王", "皇帝", "天皇", "皇后", "皇太子", "王子", "王女",
    "総書記", "書記長", "大使", "教皇", "裁判長", "判事", "最高裁判事",
)
_TITLE_PREFIXES = ("元", "前", "現", "次期", "新")

# 私人への帰属を示す手掛かり語（LOCATION / FACILITY / ORG の前後 window 文字を見る）
DEFAULT_ATTRIBUTION_CUES: tuple[str, ...] = (
    "最寄り", "最寄", "在住", "勤務先", "勤務", "通学", "通勤", "通院", "かかりつけ",
    "自宅", "実家", "出身", "所属", "住所", "居住", "転居", "引越", "本籍", "配属", "赴任",
    "在籍", "入院", "入所", "連絡先",
)
# 帳票ラベル（スパン直前）
_FIELD_LABEL_RE = re.compile(
    r"(住所|所在地|勤務先|所属|最寄り?駅?|自宅|居住地|本籍|出身地?|学校|連絡先)\s*[:：]\s*$"
)
_SEPARATORS = (" ", "・", "=", "･")
_ASCII_SHORT_RE = re.compile(r"[A-Za-z0-9&.\-]{1,4}")
_FULLNAME_MIN_CHARS = 4
_FIELD_LOOKBACK = 12

# ---- 住所成分ガード（番地以下の住所は直接識別子なので全レベルで匿名化する）----
# 住所文脈を示すエンティティ型（Resolver で ADDRESS が LOCATION に勝つため、住所の断片は隣接して残る）
DEFAULT_ADDRESS_CONTEXT_TYPES: tuple[str, ...] = ("ADDRESS", "POSTAL_CODE")
# keep="city"（design_identifiability.md §3.5）と同じ粒度。ここまでは住所文脈でも公知のまま残す
DEFAULT_CITY_SUFFIX_CHARS = "都道府県市区郡"
_HY = r"\-－‐‑‒–—―−ー"
_NUM = "[0-9]"                      # 照合前に NFKC 正規化するので全角は不要
_CHAIN = rf"{_NUM}{{1,5}}(?:[{_HY}の]{_NUM}{{1,4}}){{1,3}}"      # 4-15-8 / 1の2の3
_BANCHI = rf"{_NUM}{{1,4}}番(?:{_NUM}{{1,4}}号)?"                 # 4番14号
# 表層自体が番地・丁目を含む（「宝塚市すみれガ丘1丁目」）→ 文脈を見ずに住所
_ADDR_IN_SURFACE_RE = re.compile(rf"{_CHAIN}|丁目|番地|{_BANCHI}|号室|大字")
# 建物名・組織名は「丁目」を含み得る（「西新宿五丁目駅」）ので番地・号室だけを見る
_ROOM_IN_SURFACE_RE = re.compile(rf"{_CHAIN}|番地|{_BANCHI}|号室")
# スパン直前が番地・号室（「4-15-39 東上善寺町」「1−7−3 神田駅前」）
_ADDR_BEFORE_RE = re.compile(rf"(?:{_CHAIN}(?:号室|号|番地?)?|丁目|番地|号室|{_BANCHI})[ 、,の]{{0,2}}$")
# スパン直後が番地・丁目（「壬生辻町 3-2-1」「西新宿 7丁目」）。数量・日付の単位が続くものは取らない
_NOT_COUNTER = r"(?![0-9年月日時分秒円万千百人件回名個台歳度％%])"
_JNUM = r"(?:[0-9]{1,4}|[一二三四五六七八九十〇]{1,4})"
_ADDR_AFTER_RE = re.compile(
    rf"^(?:{_JNUM}?(?:丁目|番地|大字)|{_JNUM}?番{_JNUM}|{_JNUM}?号室|{_CHAIN}{_NOT_COUNTER})")
_ISO_DATE_RE = re.compile(rf"^{_NUM}{{4}}[{_HY}/.]{_NUM}{{1,2}}[{_HY}/.]{_NUM}{{1,2}}$")
_POSTAL_RE = re.compile(rf"〒|(?<![0-9{_HY}]){_NUM}{{3}}[{_HY}]{_NUM}{{4}}(?![0-9{_HY}])")
_ADDR_LABEL_RE = re.compile(
    r"(?:住所|所在地|居住地|本籍|宛先|お届け先|送付先|納品先|郵便番号)[ :：（(\[「【]{0,4}$")
_UPPER_ADMIN_RE = re.compile(r"郡[^都道府県市区郡]{1,6}[町村]$")     # 「加茂郡川辺町」は自治体（町名ではない）
_ADDR_GAP_CHARS = " 、,の・"                      # 住所の構成要素をつなぐ文字（改行は含めない）
_ADDR_LOOKBEHIND = 16
_POSTAL_WINDOW = 40
_POSTAL_LOOKAHEAD = 20
_AFTER_LOOKAHEAD = 12


def _slice_before(text: str, pos: int, n: int, lines: int = 1) -> str:
    """pos の手前 n 字を、改行 ``lines-1`` 個までに限って NFKC 正規化して返す。"""
    seg = text[max(0, pos - n):pos]
    return normalize_surface("\n".join(seg.split("\n")[-lines:]))


def _slice_after(text: str, pos: int, n: int, lines: int = 1) -> str:
    seg = text[pos:pos + n]
    return normalize_surface("\n".join(seg.split("\n")[:lines]))


def _strip_legal_forms(surface: str) -> str:
    """先頭・末尾の法人格を外す（Refiner の拡張で「株式会社」が付いても見出し語に一致させる）。"""
    s = surface.strip()
    for form in _LEGAL_FORMS:
        if s.startswith(form):
            s = s[len(form):].strip()
            break
    for form in _LEGAL_FORMS:
        if s.endswith(form):
            s = s[:-len(form)].strip()
            break
    return s


def _title_regexes(titles: Sequence[str]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    alts = "|".join(re.escape(t) for t in sorted(set(titles), key=len, reverse=True))
    prefixes = "|".join(_TITLE_PREFIXES)
    after = re.compile(rf"^\s*(?:{prefixes})?(?:{alts})")
    inside = re.compile(rf"(?:{prefixes})?(?:{alts})$")
    return after, inside


class IdentifiabilityAnnotator:
    name = "identifiability"

    def __init__(
        self,
        *,
        level: str = "standard",
        knowledge: Container[str] | None = None,
        public_titles: Sequence[str] = DEFAULT_PUBLIC_TITLES,
        attribution_cues: Sequence[str] = DEFAULT_ATTRIBUTION_CUES,
        window: int = 40,
        person_types: Sequence[str] = ("PERSON",),
        place_types: Sequence[str] = ("LOCATION", "FACILITY", "ORG"),
        public_suffix: str = PUBLIC_SUFFIX,
        address_guard: bool = True,
        address_context_types: Sequence[str] = DEFAULT_ADDRESS_CONTEXT_TYPES,
        address_area_types: Sequence[str] = ("LOCATION",),
        city_suffix_chars: str = DEFAULT_CITY_SUFFIX_CHARS,
        address_gap: int = 2,
    ) -> None:
        if level not in LEVELS:
            raise ConfigurationError(f"level must be one of {LEVELS}")
        if window < 0:
            raise ConfigurationError("window must be >= 0")
        if not public_suffix:
            raise ConfigurationError("public_suffix must be non-empty")
        if knowledge is not None and not hasattr(knowledge, "__contains__"):
            raise ConfigurationError("knowledge must support the 'in' operator")
        self.level = level
        self.knowledge = knowledge
        self.public_titles = tuple(public_titles)
        self.attribution_cues = tuple(attribution_cues)
        self.window = window
        self.person_types = frozenset(person_types)
        self.place_types = frozenset(place_types)
        self.public_suffix = public_suffix
        if address_gap < 0:
            raise ConfigurationError("address_gap must be >= 0")
        self.address_guard = bool(address_guard)
        self.address_context_types = frozenset(address_context_types)
        self.address_area_types = frozenset(address_area_types)
        self.city_suffix_chars = city_suffix_chars
        self.address_gap = address_gap
        self._title_after, self._title_inside = _title_regexes(self.public_titles)

    # ---- 公開API ----

    def annotate(self, text: str, entities: Sequence[Entity]) -> list[Entity]:
        entities = list(entities)
        if self.level == "strict":
            return entities
        public_idx: set[int] = set()
        private_spans: list[tuple[int, int]] = []
        for i, e in enumerate(entities):
            if e.entity_type in self.person_types:
                if self._person_is_public(text, e):
                    public_idx.add(i)
                else:
                    private_spans.append((e.start, e.end))
        addr_parts = self._address_components(text, entities)
        flags: dict[int, bool] = {}
        for i, e in enumerate(entities):
            t = e.entity_type
            if t in self.person_types:
                flags[i] = i in public_idx
            elif t in self.place_types:
                if i in addr_parts:
                    flags[i] = False   # 番地以下の住所は直接識別子（minimal でも匿名化）
                elif self.level == "minimal":
                    flags[i] = True
                elif self._attributed(text, e, private_spans):
                    flags[i] = False
                else:
                    flags[i] = self._place_is_public(text, e)
        if self.level != "minimal":
            # NER が複合名を分割した断片（株式会社アオバ|情報システム、ハシバ事務機|リース株式会社）では、
            # 残った一般名詞部分が見出し語に一致して公知になる。非公知の場所エンティティに隣接（間 1 文字以内）する
            # 公知の場所エンティティは同じ名前の一部とみなして公知を取り消す（fail-closed）
            places = sorted((i for i, e in enumerate(entities) if e.entity_type in self.place_types),
                            key=lambda i: entities[i].start)
            changed = True
            while changed:
                changed = False
                for a, b in zip(places, places[1:]):
                    if a in addr_parts or b in addr_parts:
                        continue  # 住所成分の伝播は _address_components 側（市区町村の例外を尊重する）
                    ea, eb = entities[a], entities[b]
                    if eb.start - ea.end > 1 or text[ea.end:eb.start].strip():
                        continue
                    if flags[a] != flags[b]:
                        flags[a] = flags[b] = False
                        changed = True
        out: list[Entity] = []
        for i, e in enumerate(entities):
            if i not in flags:
                out.append(e)
                continue
            out.append(dataclasses.replace(e, entity_type=e.entity_type + self.public_suffix) if flags[i] else e)
        return out

    # ---- 判定 ----

    def _person_is_public(self, text: str, e: Entity) -> bool:
        surface = text[e.start:e.end]
        if self.public_titles:
            if self._title_after.match(text[e.end:e.end + 16]):
                return True  # 安倍元総理・岸田首相（裸姓でも公職肩書きで公人）
            if self._title_inside.search(surface.rstrip()):
                return True  # NER が肩書きまで巻き込んだ「岸田首相」
        if self.knowledge is None:
            return False
        norm = normalize_surface(surface)
        compact = norm.replace(" ", "")
        is_fullname = len(compact) >= _FULLNAME_MIN_CHARS or any(s in norm for s in _SEPARATORS)
        return is_fullname and surface in self.knowledge

    def _place_is_public(self, text: str, e: Entity) -> bool:
        if self.knowledge is None:
            return False
        surface = text[e.start:e.end]
        compact = normalize_surface(surface).replace(" ", "")
        if not compact or _ASCII_SHORT_RE.fullmatch(compact):
            return False  # 短い英字略称は無関係な記事名と衝突しやすい（NDS = Nintendo DS 等）
        if surface in self.knowledge:
            return True
        stripped = _strip_legal_forms(surface)
        return stripped != surface and len(stripped) >= 2 and stripped in self.knowledge

    # ---- 住所成分ガード ----

    def _address_components(self, text: str, entities: Sequence[Entity]) -> set[int]:
        """住所の一部として現れている場所エンティティの添字集合。

        NER が住所の町名・建物名を LOCATION / FACILITY として型付けすると、jawiki 見出し語に
        一致して ``_PUBLIC`` = Keep になり住所が平文で残る。住所文脈の正の証拠があり、かつ
        市区町村以上（keep="city" と同粒度）でないものを住所成分とみなして公知判定から外す。
        """
        if not self.address_guard:
            return set()
        addr = [e for e in entities if e.entity_type in self.address_context_types]
        places = sorted((i for i, e in enumerate(entities) if e.entity_type in self.place_types),
                        key=lambda i: (entities[i].start, entities[i].end))
        out: set[int] = set()
        for i in places:
            if (self._address_evidence(text, entities[i], addr)
                    and not self._address_exempt(text, entities[i])):
                out.add(i)
        # 住所成分に隣接する場所は同じ住所の一部（…西新宿4-15-8 | アオバビル | 501号室）
        changed = True
        while changed:
            changed = False
            for a, b in zip(places, places[1:]):
                if not self._connector_gap(text, entities[a].end, entities[b].start):
                    continue
                for s, d in ((a, b), (b, a)):
                    if s in out and d not in out and not self._address_exempt(text, entities[d]):
                        out.add(d)
                        changed = True
        return out

    def _address_evidence(self, text: str, e: Entity, addr: Sequence[Entity]) -> str | None:
        """住所文脈の正の証拠。無ければ None（証拠が無ければ従来どおりの公知判定に任せる）。"""
        s, t = e.start, e.end
        area = e.entity_type in self.address_area_types
        surface_re = _ADDR_IN_SURFACE_RE if area else _ROOM_IN_SURFACE_RE
        if surface_re.search(normalize_surface(text[s:t])):
            return "surface"          # 表層自体が番地・丁目・号室を含む
        for a in addr:
            if a.start < t and a.end > s:
                return "overlap"
            if a.end <= s and self._connector_gap(text, a.end, s):
                return "left"         # 「…4-15-8 神田駅前MDビル」
            if area and a.start >= t and self._connector_gap(text, t, a.start):
                return "right"        # 「壬生辻町 3-2-1」（町名 → 番地の順）
        m = _ADDR_BEFORE_RE.search(_slice_before(text, s, _ADDR_LOOKBEHIND))
        if m and not _ISO_DATE_RE.match(m.group().strip(_ADDR_GAP_CHARS)):
            return "banchi-before"    # 「2026-04-01 新宿駅」の日付は番地としない
        if not area:
            return None               # 建物名・組織名は隣接する番地だけを証拠にする（過剰マスク抑制）
        if _ADDR_AFTER_RE.match(_slice_after(text, t, _AFTER_LOOKAHEAD)):
            return "banchi-after"
        if (_POSTAL_RE.search(_slice_before(text, s, _POSTAL_WINDOW, lines=2))
                or _POSTAL_RE.search(_slice_after(text, t, _POSTAL_LOOKAHEAD, lines=2))):
            return "postal"           # 〒・郵便番号（改行 1 個まで跨ぐ。帳票は 〒 が別行）
        if _ADDR_LABEL_RE.search(_slice_before(text, s, _ADDR_LOOKBEHIND)):
            return "label"            # 「住所: 」「旧住所(」
        return None

    def _address_exempt(self, text: str, e: Entity) -> bool:
        """住所文脈でも公知のまま残す例外: 市区町村以上の行政区画と、法人格を含む組織名。"""
        s = normalize_surface(text[e.start:e.end])
        if not s:
            return True
        if e.entity_type not in self.address_area_types:
            return any(f in s for f in _LEGAL_FORMS)      # 「株式会社アオバ」は住所成分ではない
        if _ADDR_IN_SURFACE_RE.search(s):
            return False
        if s[-1] in self.city_suffix_chars:
            return True                                    # 東京都 / 新宿区 / 名古屋市 / 加茂郡
        return bool(_UPPER_ADMIN_RE.search(s))   # 加茂郡川辺町（郡が町村の直前）

    def _connector_gap(self, text: str, a: int, b: int) -> bool:
        """a..b が住所の構成要素をつなぐ文字だけか（改行を挟んだら別ブロックとみなす）。"""
        if b < a:
            return False
        raw = text[a:b]
        if "\n" in raw or len(raw) > self.address_gap:
            return False
        return all(c in _ADDR_GAP_CHARS for c in normalize_surface(raw))

    def _attributed(self, text: str, e: Entity, private_spans: Sequence[tuple[int, int]]) -> bool:
        if not private_spans:
            return False
        lo, hi = e.start - self.window, e.end + self.window
        if not any(ps < hi and pe > lo for ps, pe in private_spans):
            return False
        ctx = text[max(0, lo):hi]
        if any(c in ctx for c in self.attribution_cues):
            return True
        return bool(_FIELD_LABEL_RE.search(text[max(0, e.start - _FIELD_LOOKBACK):e.start]))

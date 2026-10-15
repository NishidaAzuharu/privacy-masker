"""種別語尾・法人格プレフィックスへのスパン拡張（v0.8）。

NER は「水戸データセンター」「浦和サービスセンター」「仙台地方裁判所」「株式会社琉伸メディカルサポート」
「鴻ノ巣興産本社ビル別館」のような複合名で、固有名部分（水戸・浦和・仙台・琉伸・鴻ノ巣興産）だけを
検出して種別語尾を落とすことがある（11 ジャンル OOD 評価で FACILITY/ORG の危険見逃しの主因）。
本モジュールは ORG / FACILITY / LOCATION のスパンに**隣接する**種別語尾（センター・病院・支店・駅・
株式会社 …）と法人格・施設種別のプレフィックス（社会福祉法人・医療法人・特別養護老人ホーム …）を
スパンに取り込む。

安全性: スパンを外側に広げるだけで PII の文字を落とすことはない（過剰マスク側）。
他のエンティティとは重ならないように、隣接エンティティの境界で拡張を止める。
PERSON は対象外（「田中クリニック」の田中が PERSON なら型が違うため触らない）。
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Sequence

from .types import Entity

# 種別語尾（右側）。長いものから照合する。複合語尾（本社ビル別館）は 1 回で、
# 「琉伸→メディカル→サポート」のような連鎖は max_steps 回まで段階的に取り込む
DEFAULT_SUFFIXES: tuple[str, ...] = (
    # 医療・介護
    "訪問看護ステーション", "訪問介護ステーション", "居宅介護支援事業所", "訪問介護事業所", "介護老人保健施設",
    "リハビリテーション病院", "リハビリテーションセンター", "地域包括支援センター", "総合病院", "記念病院",
    "温泉病院", "大学病院", "中央病院", "市民病院", "在宅クリニック", "クリニック", "病院", "医院", "診療所",
    "歯科", "眼科", "内科", "整形外科", "小児科", "調剤薬局", "薬局", "訪問看護", "デイサービス",
    # 事業所・拠点
    "データセンター", "サービスセンター", "コールセンター", "サポートセンター", "カスタマーセンター",
    "配送センター", "物流センター", "流通センター", "研修センター", "技術センター", "研究センター",
    "開発センター", "保健センター", "文化センター", "センター", "ステーション", "オフィス", "事務所",
    "営業所", "支店", "支社", "本店", "本社ビル別館", "本社ビル", "本社", "工場", "製作所", "倉庫",
    "研究所", "事業所", "事業部", "営業部", "拠点", "号店", "店舗", "ビル別館", "ビル", "別館", "本館",
    "新館", "タワー",
    # 公共・教育・文化
    "市民会館", "文化会館", "公民館", "会館", "ホール", "体育館", "図書館", "美術館", "博物館", "資料館",
    "記念館", "小学校", "中学校", "高等学校", "高校", "大学院", "大学", "学園", "学院", "幼稚園", "保育園",
    "保育所", "学校", "駅", "空港", "郵便局", "銀行", "信用金庫", "信用組合", "農協", "商工会議所", "商工会",
    "市役所", "区役所", "町役場", "村役場", "役所", "地方裁判所", "家庭裁判所", "高等裁判所", "簡易裁判所",
    "裁判所", "検察庁", "警察署", "消防署", "税務署", "保健所", "神社", "教会", "競技場", "スタジアム",
    "アリーナ", "ドーム", "庁", "省", "局", "課",
    # 組織単位・団体
    "事務局", "委員会", "協議会", "協会", "連合会", "組合", "財団", "機構", "本部", "支部",
    # 法人格・企業種別（末尾）
    "株式会社", "有限会社", "合同会社", "合資会社", "ホールディングス", "グループ", "商事", "工業",
    "電機", "建設", "運輸", "物産", "印刷", "薬品", "興産", "産業", "通信", "不動産", "食品", "化学",
    "製薬", "自動車", "電力", "鉄道", "電鉄", "航空", "海運", "証券", "保険", "出版", "新聞社", "放送",
    # カタカナ企業語（連鎖で取り込む: 琉伸→メディカル→サポート）
    "システムズ", "ソリューションズ", "テクノロジーズ", "パートナーズ", "エンジニアリング", "コンサルティング",
    "ロジスティクス", "ネットワークス", "コミュニケーションズ", "メディカル", "サポート", "サービス",
    "システム", "ソリューション", "テクノロジー",
)

# プレフィックス（左側）。名称の前に置かれる法人格・施設種別
DEFAULT_PREFIXES: tuple[str, ...] = (
    "特別養護老人ホーム", "介護老人保健施設", "介護老人福祉施設", "有料老人ホーム", "グループホーム",
    "社会福祉法人", "医療法人社団", "医療法人財団", "医療法人", "学校法人", "一般社団法人", "一般財団法人",
    "公益社団法人", "公益財団法人", "特定非営利活動法人", "NPO法人", "ＮＰＯ法人", "独立行政法人",
    "地方独立行政法人", "国立大学法人", "宗教法人", "税理士法人", "弁護士法人", "司法書士法人", "監査法人",
    "株式会社", "有限会社", "合同会社", "合資会社",
)
# 行政区画の 1 文字語尾（名古屋→名古屋市、青葉→青葉区）
DEFAULT_ADMIN_SUFFIX_CHARS = "都道府県市区町村郡"
# 数字を含む語尾（部屋・階・号館）。「サンモールビル別館302号室」の 302号室 を取り込む。
# 番地の数字（4-15-8）は ADDRESS 型で NER が拾うためここでは扱わない
DEFAULT_SUFFIX_PATTERNS: tuple[str, ...] = (
    r"[ 　]?[0-9０-９]{1,4}号室", r"[ 　]?[0-9０-９]{1,3}階", r"[ 　]?[0-9０-９]{1,3}[FＦ](?![A-Za-z])",
    r"[ 　]?[0-9０-９A-ZＡ-Ｚ]{1,3}号館", r"[ 　]?[A-ZＡ-Ｚ]棟",
)
# 数字付き語尾を付ける型。既定では付けない（2026-09-14 実測: 文字ヘッドモデルは階・号室を
# 自分で取り込むので拡張は「青柳会中央病院1階」「別館 302号室」のような gold 外の追加になるだけ。
# token モデルでは過剰マスク −8%（固定OOD）に対し盲検 v1 で 1 字漏れ +3 の取引）。
# 旧挙動に戻すには numeric_suffix_types=("FACILITY", "ORG", "LOCATION")
DEFAULT_NUMERIC_SUFFIX_TYPES: tuple[str, ...] = ()
DEFAULT_TYPES: tuple[str, ...] = ("ORG", "FACILITY", "LOCATION")
# LOCATION に付けてよい語尾。地形・交通・公共機関に限る。企業の種別語尾（支社・本社・事業所・
# センター）を地名に付けると「鳥取市支社」「北米市場」のように別の実体まで飲み込む
# （2026-09-14 実測: 文字ヘッド併用の住所書式セットで過剰 727→601 字）
DEFAULT_LOCATION_SUFFIXES: tuple[str, ...] = (
    "駅", "空港", "港", "郵便局", "市役所", "区役所", "町役場", "村役場", "役所", "役場", "県庁", "市庁", "庁",
    "警察署", "消防署", "税務署", "保健所", "地方裁判所", "家庭裁判所", "簡易裁判所", "高等裁判所", "裁判所",
    "温泉", "公園", "橋", "川", "山", "島", "湖", "湾", "海岸", "峠", "高原", "平野", "半島", "岬", "渓谷",
    "神社", "城",
)
# 住所の末尾（盲検評価 v1 で ADDRESS 漏れ 17 件の主因）。
# 直結する番地・号室（東原|5685、栄浜|3-4、栄浜1-1|-2）と、建物名の後の空白＋部屋番号（スカイコート栄浜| 908）。
# 年月日・時刻・金額・数量の単位が続くものは取らない（「2025年」「3日目」「908円」）
_NOT_COUNTER = r"(?![0-9０-９年月日時分秒円万千百人件回名個台歳度％%])"
DEFAULT_ADDRESS_TAIL_PATTERNS: tuple[str, ...] = (
    r"[0-9０-９]{1,5}(?:[-－‐の][0-9０-９]{1,4}){0,3}(?:号室|号|番地?)?" + _NOT_COUNTER,
    r"(?:[-－‐][0-9０-９]{1,4}){1,3}(?:号室|号|番地?)?" + _NOT_COUNTER,
)
# LOCATION（町名だけの検出）は区切りを含む番地だけ取る（「東京2020」「京都3」を取らない）
DEFAULT_LOCATION_TAIL_PATTERNS: tuple[str, ...] = (
    r"[0-9０-９]{1,5}(?:[-－‐の][0-9０-９]{1,4}){1,3}(?:号室|号|番地?)?" + _NOT_COUNTER,
    r"(?:[-－‐][0-9０-９]{1,4}){1,3}(?:号室|号|番地?)?" + _NOT_COUNTER,
)
DEFAULT_ROOM_TAIL_PATTERNS: tuple[str, ...] = (
    r"[ 　][0-9０-９]{3,4}(?:号室)?" + _NOT_COUNTER,
)
DEFAULT_ADDRESS_TYPES: tuple[str, ...] = ("ADDRESS", "LOCATION")   # 直結の番地はどちらにも、空白＋部屋番号は ADDRESS だけ
_PREFIX_GAP_CHARS = " 　"  # 「株式会社 琉伸」のような法人格と名称の間の空白 1 個を許容
# 住所の末尾に電話番号を吸い込まない。「栄浜2-14-9　047-000-0000」（架空）は 空白＋3〜4 桁 が部屋番号の
# パターンに合い、続く「-000-0000」を番地の連鎖として 2 段で取り込んでいた（2026-09-14 に実測した形。例は架空）。
# 取り込んだ数字列の直後が ハイフン＋数字 なら電話番号の途中なので取らない。
# 総桁数の上限は保険（住所の末尾が 9 桁を超えることは無い）
_PHONE_CONTINUATION_RE = re.compile(r"[-－‐‑–—―−ー][0-9０-９]")
_TAIL_UNIT_RE = re.compile(r"(?:号室|号|番地|番)$")
_DIGIT_RE = re.compile(r"[0-9０-９]")
DEFAULT_MAX_TAIL_DIGITS = 8


class AffixExpander:
    def __init__(
        self,
        suffixes: Sequence[str] = DEFAULT_SUFFIXES,
        prefixes: Sequence[str] = DEFAULT_PREFIXES,
        types: Sequence[str] = DEFAULT_TYPES,
        admin_suffix_chars: str = DEFAULT_ADMIN_SUFFIX_CHARS,
        suffix_patterns: Sequence[str] = DEFAULT_SUFFIX_PATTERNS,
        max_steps: int = 3,
        max_added_chars: int = 24,
        address_types: Sequence[str] = DEFAULT_ADDRESS_TYPES,
        address_tail_patterns: Sequence[str] = DEFAULT_ADDRESS_TAIL_PATTERNS,
        location_tail_patterns: Sequence[str] = DEFAULT_LOCATION_TAIL_PATTERNS,
        room_tail_patterns: Sequence[str] = DEFAULT_ROOM_TAIL_PATTERNS,
        location_suffixes: Sequence[str] = DEFAULT_LOCATION_SUFFIXES,
        numeric_suffix_types: Sequence[str] = DEFAULT_NUMERIC_SUFFIX_TYPES,
        max_tail_digits: int = DEFAULT_MAX_TAIL_DIGITS,
    ) -> None:
        self.suffixes = tuple(sorted(set(suffixes), key=len, reverse=True))
        self.location_suffixes = tuple(sorted(set(location_suffixes), key=len, reverse=True))
        self.numeric_suffix_types = frozenset(numeric_suffix_types)
        if max_tail_digits < 1:
            raise ValueError("max_tail_digits must be >= 1")
        self.max_tail_digits = max_tail_digits
        self._suffix_patterns = tuple(re.compile(p) for p in suffix_patterns)
        self.address_types = frozenset(address_types)
        self._address_tail = tuple(re.compile(p) for p in address_tail_patterns)
        self._location_tail = tuple(re.compile(p) for p in location_tail_patterns)
        self._room_tail = tuple(re.compile(p) for p in room_tail_patterns)
        self.prefixes = tuple(sorted(set(prefixes), key=len, reverse=True))
        self.types = frozenset(types)
        self.admin_suffix_chars = admin_suffix_chars
        self.max_steps = max_steps
        self.max_added_chars = max_added_chars

    def expand(self, text: str, entities: Iterable[Entity]) -> list[Entity]:
        ents = sorted(entities, key=lambda x: (x.start, x.end))
        out: list[Entity] = []
        for i, ent in enumerate(ents):
            if ent.entity_type not in self.types and ent.entity_type not in self.address_types:
                out.append(ent)
                continue
            # 他エンティティ（型を問わず）とは重ねない
            left_limit = max((x.end for x in ents if x is not ent and x.end <= ent.start), default=0)
            right_limit = min((x.start for x in ents if x is not ent and x.start >= ent.end), default=len(text))
            if ent.entity_type in self.types:
                s, e = self._expand_one(text, ent, left_limit, right_limit)
            else:
                s, e = ent.start, ent.end
            if ent.entity_type in self.address_types:
                e = self._expand_address_tail(text, ent.entity_type, s, e, right_limit)
            out.append(ent if (s, e) == (ent.start, ent.end)
                       else dataclasses.replace(ent, start=s, end=e))
        return out

    def _expand_address_tail(self, text: str, etype: str, s: int, e: int, right_limit: int) -> int:
        """住所の末尾（番地・号室・建物名の後の部屋番号）を取り込む。予算は max_added_chars。"""
        budget = self.max_added_chars
        e0 = e
        for _ in range(self.max_steps):
            grew = False
            pats = list(self._address_tail if etype == "ADDRESS" else self._location_tail)
            if etype == "ADDRESS" and e > s and not text[e - 1].isspace():
                pats += list(self._room_tail)
            for pat in pats:
                m = pat.match(text, e, right_limit)
                if not (m and m.end() > e and (m.end() - e) <= budget):
                    continue
                tail = text[e:m.end()]
                # 単位で終わらない数字列の直後が「-数字」なら電話番号の途中（「　047」+「-000-0000」）
                if not _TAIL_UNIT_RE.search(tail) and _PHONE_CONTINUATION_RE.match(text, m.end()):
                    continue
                budget -= m.end() - e
                e = m.end()
                grew = True
                break
            if not grew:
                break
        # 保険: 取り込んだ数字が多すぎれば住所の末尾ではない（電話番号・口座番号）
        if len(_DIGIT_RE.findall(text[e0:e])) > self.max_tail_digits:
            return e0
        return e

    def _expand_one(self, text: str, ent: Entity, left_limit: int, right_limit: int) -> tuple[int, int]:
        s, e = ent.start, ent.end
        budget = self.max_added_chars
        # 地名には企業の種別語尾を付けない（鳥取市|支社 は別の実体）。数字付き語尾は指定した型だけ
        suffixes = self.location_suffixes if ent.entity_type == "LOCATION" else self.suffixes
        numeric_patterns = self._suffix_patterns if ent.entity_type in self.numeric_suffix_types else ()
        for _ in range(self.max_steps):
            grew = False
            # 右: 種別語尾
            for suf in suffixes:
                n = len(suf)
                if n <= budget and e + n <= right_limit and text.startswith(suf, e):
                    e += n
                    budget -= n
                    grew = True
                    break
            else:
                if (budget >= 1 and e < right_limit and text[e] in self.admin_suffix_chars
                        and text[e - 1] not in self.admin_suffix_chars):
                    e += 1
                    budget -= 1
                    grew = True
                else:
                    for pat in numeric_patterns:   # 302号室・3階・A棟 等（numeric_suffix_types の型だけ）
                        m = pat.match(text, e, right_limit)
                        if m and m.end() > e and (m.end() - e) <= budget:
                            budget -= m.end() - e
                            e = m.end()
                            grew = True
                            break
            # 左: 法人格・施設種別プレフィックス（空白 1 個を挟んでもよい）
            for pre in self.prefixes:
                n = len(pre)
                for gap in (0, 1):
                    start = s - n - gap
                    if start < left_limit or n + gap > budget:
                        continue
                    if gap and text[s - 1] not in _PREFIX_GAP_CHARS:
                        continue
                    if text.startswith(pre, start):
                        s = start
                        budget -= n + gap
                        grew = True
                        break
                if grew and s == start:
                    break
            if not grew:
                break
        return s, e

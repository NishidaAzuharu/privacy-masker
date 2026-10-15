"""学習データ v11 の値プール。Nemotron の架空人物・KEN_ALL の住所・組織/施設の生成器・表記の変換・評価表層の除外。

スナップショット（データフォルダ、SHA 付き）:
  PYTHONPATH=. .venv/bin/python tools/synth_v11_values.py snapshot-personas [--n 20000] [--out <データフォルダ>/synth_v11/personas_v11.jsonl]
  PYTHONPATH=. .venv/bin/python tools/synth_v11_values.py snapshot-ken-all [--out <データフォルダ>/synth_v11/ken_all_v11.csv]

設計は開発記録（非公開）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from build_data_v7 import GIVEN_KATA_JA, SURNAMES_KATA_JA  # noqa: E402
from build_data_v8 import EXTRA_SURNAMES, eval_person_exclusions  # noqa: E402
from negatives_v8 import COUNTRIES_REGIONS  # noqa: E402

from privacy_masker.data.ja_pools import GIVEN_KANJI, SURNAMES_KANJI  # noqa: E402

NEMOTRON_DATASET = "nvidia/Nemotron-Personas-Japan"
NEMOTRON_REVISION = "f1f37019d8497143c507b3deb547e65646de2ab7"
KEN_ALL_URL = "https://www.post.japanpost.jp/service/search/zipcode/download/utf/zip/utf_ken_all.zip"
PERSONAS_FILE = "personas_v11.jsonl"
KEN_ALL_FILE = "ken_all_v11.csv"
SNAPSHOT_DIR_NAME = "synth_v11"

# 名前部分は非貪欲（v5c と同じ。貪欲だと直後の助詞を巻き込む）
_NAME_RE = re.compile(r"^([一-龥々ぁ-んァ-ヶー]{1,5})[ 　]([一-龥々ぁ-んァ-ヶー]{1,7}?)(?=は|が|さん|氏|、)")

PERSON_SCRIPTS: dict[str, float] = {"kanji": 0.60, "kata_foreign": 0.10, "kata_ja": 0.08, "hira": 0.07,
                                    "romaji": 0.07, "halfwidth": 0.04, "historic": 0.04}
LOCATION_FORMS: dict[str, float] = {"prefecture": 0.18, "city": 0.22, "town": 0.12, "country": 0.10,
                                    "foreign_kata": 0.20, "compound": 0.15, "station": 0.03}
ORG_FORMS: dict[str, float] = {"corporate": 0.30, "kata_brand": 0.30, "abbr_latin": 0.10, "dept": 0.10,
                               "long_compound": 0.10, "quoted_brand": 0.10}
ADDRESS_FORMS: dict[str, float] = {"pref_first": 0.40, "city_first": 0.15, "banchi_first": 0.25,
                                   "with_building": 0.15, "fullwidth": 0.05}
UNLABELED_TYPES: tuple[str, ...] = ("PHONE", "EMAIL", "URL", "POSTAL", "DATE", "ID")

ROLE_WORDS: tuple[str, ...] = ("担当者", "担当", "夜勤担当", "日勤担当", "上長", "本人", "窓口", "総合窓口", "システム",
                               "当番", "代理", "受付", "管理者", "責任者", "各自", "オペレーター", "ご家族", "保護者")

COUNTRIES: tuple[str, ...] = tuple(sorted(set(COUNTRIES_REGIONS) | {
    "日本", "アメリカ", "米国", "中国", "韓国", "イギリス", "フランス", "ドイツ", "イタリア", "スペイン", "カナダ",
    "オーストラリア", "インド", "ブラジル", "ロシア", "タイ", "ベトナム", "インドネシア", "フィリピン", "シンガポール",
    "台湾", "メキシコ", "トルコ", "エジプト", "南アフリカ", "スウェーデン", "オランダ", "スイス", "ポーランド", "ケニア"}))

FOREIGN_PLACES_KATA: tuple[str, ...] = (
    "パリ", "ロンドン", "ニューヨーク", "ベルリン", "ローマ", "マドリード", "シドニー", "トロント", "バンクーバー", "ロサンゼルス",
    "シカゴ", "ボストン", "シアトル", "サンフランシスコ", "ミュンヘン", "ウィーン", "プラハ", "ブダペスト", "ワルシャワ",
    "アムステルダム", "ブリュッセル", "ストックホルム", "オスロ", "コペンハーゲン", "ヘルシンキ", "リスボン", "バルセロナ",
    "ミラノ", "フィレンツェ", "ベネチア", "アテネ", "イスタンブール", "ドバイ", "カイロ", "ナイロビ", "ケープタウン",
    "ムンバイ", "デリー", "バンコク", "ハノイ", "ホーチミン", "ジャカルタ", "マニラ", "クアラルンプール", "ソウル",
    "メルボルン", "オークランド", "ホノルル", "モントリオール", "リヨン", "マルセイユ", "ハンブルク", "フランクフルト",
    "チューリッヒ", "ジュネーブ", "ダブリン", "エディンバラ", "マンチェスター", "リバプール", "サンパウロ", "リオデジャネイロ",
    "ブエノスアイレス", "リマ", "サンティアゴ", "ボゴタ", "モスクワ", "キエフ", "テヘラン", "リヤド", "カサブランカ",
)

# 一般的な名・姓の転写（欧米・中華圏・朝鮮半島・東南アジア・南アジア）。特定の個人を指さない組み合わせ生成用
FOREIGN_GIVEN_KATA: tuple[str, ...] = (
    "ジョン", "マイケル", "デイビッド", "ジェームズ", "ロバート", "ウィリアム", "リチャード", "トーマス", "ダニエル", "マシュー",
    "アンドリュー", "クリストファー", "ジョセフ", "ケビン", "ブライアン", "ジェイソン", "ジャスティン", "エリック", "スティーブン",
    "ポール", "マーク", "ジョージ", "アダム", "ライアン", "ジェシカ", "エミリー", "アシュリー", "アマンダ", "ステファニー",
    "ニコール", "エリザベス", "ミシェル", "キャサリン", "ハンナ", "オリビア", "ソフィア", "イザベラ", "シャーロット", "アメリア",
    "エヴァ", "ノア", "リアム", "ルーカス", "フェリックス", "マルコ", "ルカ", "パブロ", "ディエゴ", "カルロス", "ファティマ",
    "アイシャ", "オマル", "ラフマン", "フォン", "ホア", "ジウ", "ソヨン", "ジホ", "ミンジ", "ユナ", "ラージ", "プリヤ",
    "アルジュン", "ティエン", "リンダ", "バーバラ", "スーザン", "カレン", "ナンシー", "ドナ", "ヘレン", "サンドラ",
)
FOREIGN_FAMILY_KATA: tuple[str, ...] = (
    "スミス", "ジョンソン", "ウィリアムズ", "ジョーンズ", "デイビス", "ガルシア", "ロドリゲス", "ウィルソン", "マルティネス",
    "アンダーソン", "テイラー", "ヘルナンデス", "ムーア", "マーティン", "ジャクソン", "トンプソン", "ホワイト", "ロペス",
    "ゴンザレス", "ハリス", "クラーク", "ルイス", "ロビンソン", "ウォーカー", "ペレス", "ホール", "ヤング", "アレン",
    "サンチェス", "ライト", "キング", "スコット", "グリーン", "ベイカー", "アダムス", "ネルソン", "ヒル", "キャンベル",
    "ミッチェル", "ロバーツ", "カーター", "フィリップス", "エバンス", "ターナー", "パーカー", "コリンズ", "エドワーズ",
    "スチュワート", "モリス", "ミュラー", "シュミット", "シュナイダー", "フィッシャー", "ウェーバー", "マイヤー", "ワグナー",
    "ベッカー", "ホフマン", "ロッシ", "ルッソ", "フェラーリ", "エスポジト", "ビアンキ", "ロメロ", "ナバロ", "チャン", "リウ",
    "チェン", "ヤン", "ホアン", "キム", "パク", "チェ", "チョン", "カン", "グエン", "ファム", "シン", "クマール", "シャルマ", "パテル",
)
ROMAJI_GIVEN: tuple[str, ...] = ("Taro", "Hanako", "Kenichi", "Misaki", "Daisuke", "Yui", "Masashi", "Ayaka", "Ryo", "Nao",
                                 "Haruto", "Yuki", "Sora", "Aoi", "Rin", "Kaito", "Mio", "Hina", "Yuto", "Sakura")
ROMAJI_SUR: tuple[str, ...] = ("Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito", "Yamamoto", "Nakamura", "Kobayashi",
                               "Kato", "Yoshida", "Yamada", "Sasaki", "Yamaguchi", "Matsumoto", "Inoue", "Kimura", "Hayashi",
                               "Shimizu", "Yamazaki", "Mori", "Abe", "Ikeda", "Hashimoto", "Ishikawa", "Ogawa", "Goto", "Okada")
NICK_KATA: tuple[str, ...] = ("ケイ", "ユウ", "マキ", "リョウ", "タク", "ナオ", "カズ", "ミキ", "ヒロ", "アキ", "サキ", "トモ",
                              "マサ", "ユカ", "ケン", "ハル", "ミナ", "ノン", "ミッチー", "タッくん")
# 没後の公知の人物（公人。PERSON として教える。standard モードでは公知判定で残る）
HISTORIC_PERSONS: tuple[tuple[str, str], ...] = (
    ("織田", "信長"), ("豊臣", "秀吉"), ("徳川", "家康"), ("源", "頼朝"), ("平", "清盛"), ("足利", "義満"), ("武田", "信玄"),
    ("上杉", "謙信"), ("伊達", "政宗"), ("坂本", "龍馬"), ("西郷", "隆盛"), ("大久保", "利通"), ("夏目", "漱石"), ("森", "鷗外"),
    ("樋口", "一葉"), ("与謝野", "晶子"), ("松尾", "芭蕉"), ("葛飾", "北斎"), ("千", "利休"), ("藤原", "道長"), ("北条", "時宗"),
    ("明智", "光秀"), ("石田", "三成"), ("真田", "幸村"), ("福沢", "諭吉"), ("野口", "英世"), ("新渡戸", "稲造"), ("津田", "梅子"),
    ("渋沢", "栄一"), ("北里", "柴三郎"), ("宮沢", "賢治"), ("芥川", "龍之介"), ("太宰", "治"), ("正岡", "子規"), ("石川", "啄木"),
)

# 組織・施設の部品（v5c の生成器を移植）
_KANJI_CORE = ("山田", "大和", "旭", "共栄", "三和", "北斗", "丸金", "常盤", "白樺", "若葉", "東洋", "太陽", "協同", "双葉",
               "栄光", "曙", "高砂", "千代田", "扇", "早瀬")
_INDUSTRY = ("製作所", "工業", "商事", "運輸", "食品", "電機", "建設", "物産", "印刷", "薬品")
_KATA_PARTS = ("テクノ", "ソリュー", "クリエ", "ネクス", "アルフ", "メディ", "ロジ", "エコ", "サン", "フロン", "ティア",
               "システムズ", "ワークス", "ラボ", "パートナーズ", "ホールディングス", "サービス", "コム", "ブレイン", "ノーツ",
               "フィールド", "ガーデン", "キッチン", "スタジオ", "マート", "ファクトリー", "リンク", "ベース", "コア", "ウェーブ")
_LONG_ORG = ("日本{x}協会", "国際{x}委員会", "全国{x}組合連合会", "{x}振興財団", "一般社団法人{x}推進機構", "{p}{x}センター",
             "{x}研究会")
_LONG_X = ("医療情報", "環境技術", "食品安全", "地域交通", "情報通信", "再生可能エネルギー", "スポーツ", "観光", "防災", "福祉")
_LEGAL_FORMS = ("社会福祉法人{x}会", "医療法人{x}会", "医療法人社団{x}会", "学校法人{x}学園", "NPO法人{x}", "一般財団法人{x}")
_LEGAL_X = ("ひだまり", "こもれび", "つばさ", "碧水", "やすらぎ", "あおば", "みどり", "さくら", "北斗", "曙", "白樺", "若葉")
_MUNI_DEPT = ("市民課", "都市計画課", "健康推進課", "道路維持課", "上下水道部工務課", "保険年金課", "産業振興課", "税務課",
              "子育て支援課", "環境政策課", "高齢福祉課", "介護保険課", "総務課", "財政課", "人事課", "企画政策課", "地域振興課")
_CORP_DEPT = ("総務課", "経理課", "人事課", "資材課", "管理課", "医事課", "品質保証課", "情報システム課", "営業部", "開発部",
              "企画総務部", "カスタマーサポート室", "法務室", "広報室", "監査局")
_FAC_SUFFIX = ("病院", "総合病院", "クリニック", "小学校", "中学校", "高等学校", "大学", "図書館", "公民館", "体育館",
               "文化センター", "保育園", "郵便局", "市民会館", "美術館", "駐車場", "第二庁舎")
_BLD_A = ("コーポ", "メゾン", "ハイツ", "レジデンス", "グラン", "パーク")
_BLD_B = ("桜", "青葉", "ひかり", "みどり", "中央", "富士見", "本町")
_TOWER_A = ("グラン", "セント", "パーク", "クロス", "ベイ", "ノース", "アクア", "プライム")
_TOWER_B = ("クロス", "レージ", "フロント", "ゲート", "マーク", "シティ", "ポート")
_TOWER_C = ("タワー", "ビル", "スクエア", "ヒルズ", "プレイス")
_EMAIL_LOCAL = ("support", "info", "sales", "sales-team", "noreply", "no-reply", "contact", "admin", "help", "order",
                "booking", "team", "office", "recruit", "press", "cs", "billing", "service")
_DOMAIN_WORDS = ("north", "gate", "tech", "foods", "freight", "logistics", "works", "labs", "shop", "mall", "clinic",
                 "kobo", "farm", "studio", "garden", "link", "base", "core", "wave", "field")
_TLDS = ("example.com", "example.co.jp", "example.jp", "example.net", "example.org", "example.ne.jp")

_HIRA_TABLE = {chr(k): chr(k - 0x60) for k in range(0x30A1, 0x30F7)}   # カタカナ → ひらがな（ァ..ヶ）
_HW_PAIRS = (("ガ", "ｶﾞ"), ("ギ", "ｷﾞ"), ("グ", "ｸﾞ"), ("ゲ", "ｹﾞ"), ("ゴ", "ｺﾞ"), ("ザ", "ｻﾞ"), ("ジ", "ｼﾞ"), ("ズ", "ｽﾞ"),
             ("ゼ", "ｾﾞ"), ("ゾ", "ｿﾞ"), ("ダ", "ﾀﾞ"), ("ヂ", "ﾁﾞ"), ("ヅ", "ﾂﾞ"), ("デ", "ﾃﾞ"), ("ド", "ﾄﾞ"), ("バ", "ﾊﾞ"),
             ("ビ", "ﾋﾞ"), ("ブ", "ﾌﾞ"), ("ベ", "ﾍﾞ"), ("ボ", "ﾎﾞ"), ("パ", "ﾊﾟ"), ("ピ", "ﾋﾟ"), ("プ", "ﾌﾟ"), ("ペ", "ﾍﾟ"),
             ("ポ", "ﾎﾟ"), ("ヴ", "ｳﾞ"))
_HW_SINGLE = dict(zip("ァアィイゥウェエォオカキクケコサシスセソタチッツテトナニヌネノハヒフヘホマミムメモャヤュユョヨラリルレロワヲンー・",
                      "ｧｱｨｲｩｳｪｴｫｵｶｷｸｹｺｻｼｽｾｿﾀﾁｯﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓｬﾔｭﾕｮﾖﾗﾘﾙﾚﾛﾜｦﾝｰ･"))
assert len(_HW_SINGLE) == 57
_HW_PAIR_MAP = dict(_HW_PAIRS)


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def is_excluded(surface: str, excl: set[str]) -> bool:
    n = normalize(surface)
    return n in excl or n.replace(" ", "").replace("　", "") in excl


def to_halfwidth_kana(s: str) -> str:
    return "".join(_HW_PAIR_MAP.get(ch) or _HW_SINGLE.get(ch, ch) for ch in s)


def to_hiragana(s: str) -> str:
    return "".join(_HIRA_TABLE.get(c, c) for c in s)


def extract_name(persona_text: str) -> tuple[str, str] | None:
    m = _NAME_RE.match((persona_text or "").strip())
    return (m.group(1), m.group(2)) if m else None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ スナップショット

def snapshot_personas(out_path: str, n: int = 20000, revision: str = NEMOTRON_REVISION) -> dict:
    """Nemotron-Personas-Japan（リビジョン固定）を streaming で読み、名前が取れた人物を jsonl に保存する。名前以外は文脈の種に使う属性だけ。"""
    from itertools import islice

    from datasets import load_dataset

    ds = load_dataset(NEMOTRON_DATASET, split="train", streaming=True, revision=revision)
    rows: list[dict] = []
    n_skip = 0
    it = iter(ds)
    for i, row in enumerate(islice(it, n)):
        name = extract_name(row.get("persona", ""))
        if name is None:
            n_skip += 1
            continue
        hobby = re.split(r"[、,。]", str(row.get("hobbies_and_interests") or ""))[0][:20]
        occ = str(row.get("occupation") or "")
        rows.append({"uuid": str(row.get("uuid") or f"row{i}"), "sei": name[0], "mei": name[1],
                     "age": row.get("age") if row.get("age") is not None else row.get("age_band"), "sex": row.get("sex") or "",
                     "occupation": occ.split()[0] if occ else "", "prefecture": row.get("prefecture") or "", "hobby": hobby})
    if hasattr(it, "close"):
        it.close()                                  # 途中で止めた streaming の generator を明示的に閉じる
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"dataset": NEMOTRON_DATASET, "revision": revision, "requested": n, "kept": len(rows), "skipped_no_name": n_skip,
            "file": out_path, "sha256": file_sha256(out_path), "created_at": dt.datetime.now().isoformat(timespec="seconds")}


def load_personas(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def snapshot_ken_all(out_path: str, url: str = KEN_ALL_URL) -> dict:
    """KEN_ALL（UTF-8 zip）を取得し、(都道府県, 市区町村, 町域) の CSV に落とす。町域の括弧書きは除き「以下に掲載がない場合」は空にする。"""
    with urllib.request.urlopen(url, timeout=120) as r:
        blob = r.read()
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                cols = [c.strip().strip('"') for c in line.strip().split(",")]
                if len(cols) < 9:
                    continue
                pref, city, town = cols[6], cols[7], cols[8]
                if "以下に掲載がない場合" in town:
                    town = ""
                town = re.sub(r"（.*?(）|$)", "", town)
                if "（" in town or "）" in town or "〜" in town or "、" in town:
                    continue
                key = (pref, city, town)
                if key not in seen:
                    seen.add(key)
                    rows.append(key)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefecture", "city", "town"])
        w.writerows(rows)
    return {"url": url, "rows": len(rows), "file": out_path, "sha256": file_sha256(out_path),
            "zip_sha256": hashlib.sha256(blob).hexdigest(), "created_at": dt.datetime.now().isoformat(timespec="seconds")}


def load_ken_all(path: str) -> list[tuple[str, str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        return [(a, b, c) for a, b, c in r]


# ------------------------------------------------------------------ 除外とプール

def build_exclusions() -> tuple[set[str], set[str]]:
    """評価セット由来の表層。人名は表層＋先頭要素（build_data_v8 と同じ規則）。地名は表層そのもの。いずれも NFKC。"""
    from build_data_v9 import collect_eval_surfaces
    person = {normalize(x) for x in eval_person_exclusions(collect_eval_surfaces(["PERSON"]))}
    place = {normalize(x) for x in collect_eval_surfaces(["LOCATION", "ADDRESS"])}
    return person, place


@dataclass
class Pools:
    personas: list[dict]
    ken_rows: list[tuple[str, str, str]]
    excl_person: set[str] = field(default_factory=set)
    excl_place: set[str] = field(default_factory=set)

    def towns(self) -> list[str]:
        return [t for _, _, t in self.ken_rows if len(t) >= 2 and not is_excluded(t, self.excl_place)]

    def prefectures(self) -> list[str]:
        return sorted({p for p, _, _ in self.ken_rows})

    def rows_in(self, prefecture: str | None) -> list[tuple[str, str, str]]:
        rows = [r for r in self.ken_rows if prefecture is None or r[0] == prefecture]
        return rows or list(self.ken_rows)


def build_pools(personas: Sequence[dict] | None = None, ken_rows: Sequence[tuple[str, str, str]] | None = None, *,
                personas_path: str | None = None, ken_path: str | None = None,
                excl_person: set[str] | None = None, excl_place: set[str] | None = None) -> Pools:
    if personas is None:
        personas = load_personas(personas_path or os.path.join(drive_dir(), SNAPSHOT_DIR_NAME, PERSONAS_FILE))
    if ken_rows is None:
        ken_rows = load_ken_all(ken_path or os.path.join(drive_dir(), SNAPSHOT_DIR_NAME, KEN_ALL_FILE))
    if excl_person is None or excl_place is None:
        ep, pl = build_exclusions()
        excl_person = ep if excl_person is None else excl_person
        excl_place = pl if excl_place is None else excl_place
    excl_person = {normalize(x) for x in excl_person}
    excl_place = {normalize(x) for x in excl_place}
    personas = [p for p in personas if not is_excluded(p["sei"], excl_person) and not is_excluded(p["sei"] + p["mei"], excl_person)]
    return Pools(list(personas), list(ken_rows), excl_person, excl_place)


# ------------------------------------------------------------------ 人名

def _pick_kanji(rng: random.Random, pools: Pools, persona: dict | None) -> tuple[str, str]:
    if persona is not None:
        return persona["sei"], persona["mei"]
    if pools.personas and rng.random() < 0.7:
        p = rng.choice(pools.personas)
        return p["sei"], p["mei"]
    sur = list(SURNAMES_KANJI) + list(EXTRA_SURNAMES) * 3      # 珍しい姓の重み 3（v10 と同じ）
    return rng.choice(sur), rng.choice(GIVEN_KANJI)


def render_person(rng: random.Random, pools: Pools, script: str, persona: dict | None = None) -> dict:
    """同一人物の表記一式。full / sei / mei / nick。除外表層に当たれば引き直す（最大 30 回）。"""
    out: dict = {}
    for _ in range(30):
        if script == "kanji":
            sei, mei = _pick_kanji(rng, pools, persona)
            sep = rng.choice(("", "", "", "", " ", "　"))
            out = {"full": f"{sei}{sep}{mei}", "sei": sei, "mei": mei, "nick": rng.choice(NICK_KATA)}
        elif script == "historic":
            sei, mei = rng.choice(HISTORIC_PERSONS)
            out = {"full": sei + mei, "sei": sei, "mei": mei, "nick": mei}
        elif script == "kata_ja":
            sei, mei = rng.choice(SURNAMES_KATA_JA), rng.choice(GIVEN_KATA_JA)
            out = {"full": f"{sei}{rng.choice(('', ' ', '　'))}{mei}", "sei": sei, "mei": mei, "nick": mei[:2]}
        elif script == "hira":
            sei, mei = to_hiragana(rng.choice(SURNAMES_KATA_JA)), to_hiragana(rng.choice(GIVEN_KATA_JA))
            out = {"full": f"{sei}{rng.choice(('', ' '))}{mei}", "sei": sei, "mei": mei, "nick": mei[:2]}
        elif script == "halfwidth":
            sei, mei = to_halfwidth_kana(rng.choice(SURNAMES_KATA_JA)), to_halfwidth_kana(rng.choice(GIVEN_KATA_JA))
            out = {"full": f"{sei} {mei}", "sei": sei, "mei": mei, "nick": mei[:3]}
        elif script == "romaji":
            sur, giv = rng.choice(ROMAJI_SUR), rng.choice(ROMAJI_GIVEN)
            full = rng.choice((f"{giv} {sur}", f"{sur} {giv}", f"{sur.upper()} {giv}", f"{giv[0]}. {sur}"))
            out = {"full": full, "sei": sur, "mei": giv, "nick": giv}
        elif script == "kata_foreign":
            fam, giv = rng.choice(FOREIGN_FAMILY_KATA), rng.choice(FOREIGN_GIVEN_KATA)
            r = rng.random()
            full = f"{giv}・{fam}" if r < 0.50 else f"{giv} {fam}" if r < 0.65 else f"{giv}{fam}" if r < 0.85 else giv
            out = {"full": full, "sei": fam, "mei": giv, "nick": giv[:2]}
        else:
            raise ValueError(f"unknown script: {script}")
        if not any(is_excluded(v, pools.excl_person) for v in out.values()):
            return out
        persona = None                      # ペルソナ名が除外に当たったらプールから引き直す
    return out


# ------------------------------------------------------------------ 住所・地名・組織・施設

def _fullwidth_digits(s: str) -> str:
    return s.translate(str.maketrans("0123456789-", "０１２３４５６７８９－"))


def _banchi(rng: random.Random) -> str:
    c, b, g = rng.randint(1, 5), rng.randint(1, 30), rng.randint(1, 20)
    r = rng.random()
    return f"{c}-{b}-{g}" if r < 0.45 else f"{c}丁目{b}番{g}号" if r < 0.70 else f"{b}-{g}" if r < 0.88 else f"{b}番地"


def _building(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return rng.choice(_BLD_A) + rng.choice(_BLD_B) + rng.choice((f"{rng.randint(1, 12)}0{rng.randint(1, 9)}号室", f"{rng.randint(2, 9)}階"))
    return rng.choice(_TOWER_A) + rng.choice(_TOWER_B) + rng.choice(_TOWER_C) + rng.choice((f" {rng.randint(2, 30)}階", f"{rng.randint(1, 30)}0{rng.randint(1, 9)}"))


def gen_address(rng: random.Random, pools: Pools, form: str, prefecture: str | None = None) -> str:
    rows = [r for r in pools.rows_in(prefecture) if not is_excluded(r[2], pools.excl_place)] or pools.rows_in(prefecture)
    pref, city, town = rng.choice(rows)
    if is_excluded(town, pools.excl_place):
        town = ""
    num = _banchi(rng)
    if form == "banchi_first":
        if rng.random() < 0.7 or not town:
            return f"{num}{rng.choice((' ', '　', ''))}{_building(rng)}"
        return f"{town}{num}"
    if form == "city_first":
        return f"{city}{town}{num}"
    if form == "with_building":
        return f"{pref}{city}{town}{num}{rng.choice((' ', '　', ''))}{_building(rng)}"
    if form == "fullwidth":
        return f"{pref}{city}{town}{_fullwidth_digits(num)}"
    return f"{pref}{city}{town}{num}"                          # pref_first


def gen_location(rng: random.Random, pools: Pools, form: str) -> str:
    if form == "country":
        return rng.choice(COUNTRIES)
    if form == "foreign_kata":
        cands = [p for p in FOREIGN_PLACES_KATA if not is_excluded(p, pools.excl_place)] or list(FOREIGN_PLACES_KATA)
        return rng.choice(cands)
    if form == "town":
        towns = pools.towns()
        return rng.choice(towns) if towns else rng.choice(pools.ken_rows)[1]
    if form == "station":
        _, city, town = rng.choice(pools.ken_rows)
        stem = town if (len(town) >= 2 and not is_excluded(town, pools.excl_place)) else re.sub(r"(市|区|町|村)$", "", city)
        return stem + "駅"
    pref, city, _ = rng.choice(pools.ken_rows)
    if form == "prefecture":
        return pref
    return city                                               # city / compound（複合語の地名部分は市名）


def gen_org(rng: random.Random, pools: Pools, form: str) -> str:
    name = ""
    for _ in range(20):
        if form == "corporate":
            core = rng.choice(_KANJI_CORE) + rng.choice(("", rng.choice(_INDUSTRY)))
            name = rng.choice((f"株式会社{core}", f"{core}株式会社", f"（株）{core}", f"{core}合同会社",
                               rng.choice(_LEGAL_FORMS).format(x=rng.choice(_LEGAL_X))))
        elif form in ("kata_brand", "quoted_brand"):
            name = "".join(rng.choice(_KATA_PARTS) for _ in range(rng.choice((2, 2, 3))))
        elif form == "abbr_latin":
            name = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(2, 5)))
        elif form == "dept":
            name = rng.choice(_MUNI_DEPT + _CORP_DEPT)
        elif form == "long_compound":
            pref = rng.choice(pools.ken_rows)[0] if pools.ken_rows else ""
            name = rng.choice(_LONG_ORG).format(x=rng.choice(_LONG_X), p=pref)
        else:
            raise ValueError(f"unknown org form: {form}")
        if not is_excluded(name, pools.excl_place):
            return name
    return name


def gen_facility(rng: random.Random, pools: Pools) -> str:
    name = ""
    for _ in range(20):
        _, city, town = rng.choice(pools.ken_rows)
        base = rng.choice((re.sub(r"(市|区|町|村)$", "", city), town[:3] if len(town) >= 3 else city, rng.choice(_KANJI_CORE)))
        name = base + rng.choice(_FAC_SUFFIX)
        if base and not is_excluded(name, pools.excl_place):
            return name
    return name


def gen_unlabeled(rng: random.Random, type_: str) -> str:
    if type_ == "PHONE":
        if rng.random() < 0.5:
            return f"0{rng.randint(70, 90)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
        return f"0{rng.randint(3, 9)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    if type_ == "EMAIL":
        return f"{rng.choice(_EMAIL_LOCAL)}{rng.choice(('', str(rng.randint(1, 99))))}@{rng.choice(_DOMAIN_WORDS)}.{rng.choice(_TLDS)}"
    if type_ == "URL":
        return (f"https://www.{rng.choice(_DOMAIN_WORDS)}-{rng.choice(_DOMAIN_WORDS)}.{rng.choice(_TLDS)}/"
                f"{rng.choice(('info', 'news', 'contact', 'items'))}/{rng.randint(100, 9999)}")
    if type_ == "POSTAL":
        return f"{rng.choice(('〒', ''))}{rng.randint(100, 999)}-{rng.randint(1000, 9999):04d}"
    if type_ == "DATE":
        y, m, d = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
        return rng.choice((f"{y}年{m}月{d}日", f"{y}/{m:02d}/{d:02d}", f"{m}月{d}日", f"令和{y - 2018}年{m}月{d}日", f"{y}-{m:02d}-{d:02d}"))
    if type_ == "ID":
        return rng.choice((f"No.{rng.randint(1000, 99999)}", f"注文番号 {rng.choice('ABCDR')}-{rng.randint(10000, 99999)}",
                           f"受付 #{rng.randint(100, 9999)}", f"会員ID {rng.randint(100000, 999999)}"))
    raise ValueError(f"unknown unlabeled type: {type_}")


# ------------------------------------------------------------------ CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("snapshot-personas")
    a.add_argument("--n", type=int, default=20000)
    a.add_argument("--out", default=None)
    b = sub.add_parser("snapshot-ken-all")
    b.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    base = os.path.join(drive_dir(), SNAPSHOT_DIR_NAME)
    if args.cmd == "snapshot-personas":
        out = args.out or os.path.join(base, PERSONAS_FILE)
        m = snapshot_personas(out, n=args.n)
    else:
        out = args.out or os.path.join(base, KEN_ALL_FILE)
        m = snapshot_ken_all(out)
    with open(out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    if args.cmd == "snapshot-personas":
        # datasets の streaming を途中で止めると、仕事が終わってもインタプリタの終了処理で止まる（スレッド・子プロセスは
        # 残っていないのに exit しない。2026-09-22 に 5 時間放置で発覚、n=5 でも再現）。成果物は書き終えているので即終了する
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

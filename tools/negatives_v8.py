"""学習データ v8 用の負例（PII を含まない文書）を Qwen3.8-27B-FP8 で生成するための計画と選別。

役割分担（CPU はローカル、GPU は Colab）:

  1. plan   : 生成計画（ジャンル × 題材 × 形式 × 文体 × 長さ、プロンプト本文込み）をデータフォルダへ書く。
              `qwen3_8_27b_negatives_v1.ipynb` は計画を読んで vLLM で生成するだけで、プロンプトを持たない。
  2. (Colab): ノートブックが raw 出力 `qwen_negatives_raw.jsonl` をデータフォルダに書く。
  3. filter : raw 出力を機械的に選別し、学習形式 `negatives_v8.jsonl` と選別レポートを書く。
              LLM 検証器は使わない（生成と同じ LLM で検証すると誤りが相関する）。

選別の考え方（設計は開発記録、非公開）:
  - 負例の価値は「現行モデルが誤検出しがちな一般語（市民・係・政府・アミメニシキヘビ…）を、PII ではないと
    教えること」にある。だから NER の検出があるだけでは落とさない。
  - 落とすのは「本当に固有名詞・PII が入った文書」= 学習規約と矛盾する負例。決定的規則（地名辞書・人名プール・
    法人語・電話/メール/URL・他 script）で reject し、規則で決め切れない強い兆候（2 モデルが同じ PERSON/ADDRESS を
    出す、PERSON 直後に敬称、法人語尾の ORG、公知地名らしい LOCATION、ハンドル名）は review に回して既定では除外する。
    review 分は `review_excluded.csv` を見て `--include-ids` で戻せる。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/negatives_v8.py plan [--n 8000] [--seed 42] [--out <データフォルダ>/negatives_v8]
  PYTHONPATH=. .venv/bin/python tools/negatives_v8.py filter --raw <Colab OUT_DIR か raw jsonl> [--raw ...]
        [--plan <データフォルダ>/negatives_v8/plan_v1.jsonl] [--target 4000] [--models char8ep,v7] [--no-ner]
        [--include-ids review_ok.txt] [--out <データフォルダ>/negatives_v8/filtered_v1]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from build_data_v7 import EVAL_SURFACES  # noqa: E402

from privacy_masker.data.ja_pools import (  # noqa: E402
    GIVEN_KANJI, GIVEN_KANJI_F, GIVEN_KATA, ORG_KNOWN_SUFFIXES, PREF_CITIES, SURNAMES_KANJI, SURNAMES_KATA,
)

# v1 → v2（2026-09-15 pilot 後）: 200 件すべてに改行が無かった（箸条書き・手順・チャット・ログが 1 行に潰れる）ので、
# 行区切りの指示と JSON 内の \n の例を足した。計画の版はファイル名でなく manifest の plan_version / plan_tag で持ち、
# ノートブックの出力先は plan_tag を含む（旧 pilot の署名と衝突しない）。
# v2 → v3: 改行の指示を足しても 0/200 のまま（モデルは JSON 文字列内に \n を書かない）。出力を {"text": ...} から
# {"lines": [...]}（1 行 1 要素）に変え、結合はローカル側で行う。
PLAN_VERSION = "negatives_v8_plan_v3"
PLAN_TAG = "plan_v3"
PROMPT_VERSION = "negatives_v8_prompt_v3"
SOURCE_NAME = "qwen38_negatives_v8"
PLAN_FILE = "plan.jsonl"
PLAN_MANIFEST_FILE = "plan.manifest.json"
RAW_FILE = "qwen_negatives_raw.jsonl"

# ---------------------------------------------------------------------------
# 1. 生成計画
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """あなたは日本語の文章作成者です。指定された種類・題材・形式・文体・長さの日本語文書を1件だけ書き、指定の JSON だけを返します。

この文書は「個人情報や固有名詞を一切含まない文章」の見本として使います。次の規則を必ず守ってください。
1. 人名を書かない。姓・名・愛称・ハンドルネーム・ユーザー名・イニシャル（A氏、Bさん、K）も禁止。人物は役職・関係・役割で呼ぶ（担当者、店長、上司、母、先生、客、隣の人）。
2. 地名を書かない。国名（日本を含む）、都道府県、市区町村、町名、駅名、山・川・湖・海、外国の地名、地方名（関東、欧州、中東）も禁止。「近くの駅」「隣の県」「海沿いの町」「山あいの村」のように書く。
3. 会社名・団体名・学校名・病院名・店名・ブランド名・商品名・サービス名・部署名（総務課、営業部など）を書かない。「取引先」「メーカー」「近所の病院」「担当部署」「この製品」と書く。
4. 住所、郵便番号、電話番号、メールアドレス、URL、口座番号、会員番号、車のナンバーなどの番号列を書かない。日付・時刻・金額・数量・温度・割合・型番風の短いコード（ERR-2041 など）は書いてよい。
5. 英字キーの値（status=OK など）には状態語・数値・日時・役割名だけを使い、人の ID やハンドル名（taro_88、@user のような形）を使わない。
6. 動物・植物・料理・道具・病名・法律の一般名、一般的な英語の略語（PDF、API、OK、NG）は自由に使ってよい。
7. 実在・架空を問わず固有名詞を使わない。それ以外は自然で具体的に書く。数値・日時・手順・状況を具体的に描く。
8. 日本語で書く。ASCII 英数字と通常の記号は使ってよいが、簡体字・繁体字・ハングル・キリル文字など他の文字体系は使わない。
9. 出力は {"lines": ["1行目", "2行目", ...]} の形の JSON だけ。本文を行ごとに分け、配列の 1 要素を 1 行にする。箇条書きの 1 項目・手順の 1 番・項目名と値の 1 組・表の 1 行・ログの 1 行・チャットの 1 発言は、それぞれ別の要素にする。段落の区切りは空文字列 "" の要素で表す。通常の文章も段落ごとに要素を分ける。
10. 説明・前置き・Markdown の囲みや見出し記号（#）は書かない。例: {"lines": ["・一項目目", "・二項目目", "", "次の段落の本文。"]}
"""

LENGTH_KEYS: dict[str, tuple[int, int]] = {"short": (40, 120), "medium": (120, 300), "long": (300, 700)}
LENGTH_WEIGHTS: dict[str, float] = {"short": 0.30, "medium": 0.50, "long": 0.20}
MAX_TOKENS: dict[str, int] = {"short": 260, "medium": 520, "long": 1000}

FORMATS: dict[str, str] = {
    "prose": "段落分けした通常の文章（段落の間は改行で分ける）",
    "bullets": "箇条書き（「・」または「-」で始まる行）を中心に構成する。1 項目 1 行、行は改行で区切る",
    "numbered_steps": "番号付きの手順（1. 2. 3. …）で構成する。1 手順 1 行、行は改行で区切る",
    "key_value": "status=OK のように英字キーと値を = でつないだ行を3行以上含める（1 行に 1 組、改行で区切る）。値は状態語・数値・日時・役割名だけ",
    "label_colon": "「品名: 」「数量: 」「担当: 」のような項目名と値の行を4行以上含める（1 項目 1 行、改行で区切る）。担当などの値には人名ではなく役割名（担当者未定、夜勤担当、当番）を入れる",
    "table_pipe": "| で区切った簡易表を含める（見出し行1行とデータ行3行以上、1 行ごとに改行）。人名の列は作らない",
    "log_lines": "「時刻 レベル メッセージ」の形式のログ行を5行以上、1 行ずつ改行して含める（例: 09:12:03 WARN 再試行 2回目）",
    "chat_roles": "行頭に「役割名: 」を付けた発言を6行以上、1 発言 1 行で改行して並べる。役割名は店員・客・質問者・回答者・先輩・後輩・上司・部下・担当・依頼者・オペレーター・お客様などの一般語だけ",
    "letter": "冒頭に宛名（役職名や「各位」）の行、本文の段落、末尾に差出人（役職名）の行を置いた連絡文。行は改行で区切る",
}

STYLES: dict[str, str] = {
    "polite": "敬体（です・ます調）",
    "plain": "常体（だ・である調）",
    "spoken": "話し言葉に近い口調",
    "formal": "硬い事務文体",
    "classic": "明治から昭和初期の小説のような文語寄りの文体",
}

OPENINGS: tuple[str, ...] = (
    "日付や数値から始める",
    "結論や要点から始める",
    "状況や場面の描写から始める",
    "見出し行から始める",
    "呼びかけや挨拶から始める",
    "疑問文から始める",
    "項目名から始める",
    "短い一文から始める",
)

# 役割語（人名の代わりに使う）。文学ジャンルの追加指示に使う。
LITERARY_ROLES = "母、父、ねえや、先生、主人、おばあさま、女中、番頭、若旦那、隣家の娘、車夫、下宿の主、書生"


@dataclass(frozen=True)
class Subgenre:
    key: str
    label: str
    topics: tuple[str, ...]
    formats: dict[str, float]
    styles: dict[str, float]
    extra: str = ""


@dataclass(frozen=True)
class Genre:
    key: str
    label: str
    weight: float
    subgenres: tuple[Subgenre, ...]
    extra: str = ""


_BUSINESS_FORMATS = {"key_value": 0.20, "label_colon": 0.25, "log_lines": 0.10, "table_pipe": 0.10,
                     "bullets": 0.10, "numbered_steps": 0.10, "prose": 0.10, "letter": 0.05}
_BUSINESS_STYLES = {"polite": 0.5, "formal": 0.4, "plain": 0.1}
_PROSE_STYLES = {"polite": 0.5, "plain": 0.4, "spoken": 0.1}

GENRES: tuple[Genre, ...] = (
    Genre("business", "業務文書", 0.30, (
        Subgenre("meeting_minutes", "議事録", (
            "月次の進捗確認", "予算の見直し", "新しい勤怠ルールの説明", "在庫の棚卸し方法", "納期遅れの原因整理",
            "研修計画", "備品購入の優先順位", "繁忙期のシフト調整", "問い合わせ対応の改善", "清掃当番の見直し",
            "安全点検の結果共有", "経費精算の締め切り変更"),
            {"bullets": 0.4, "label_colon": 0.3, "prose": 0.3}, _BUSINESS_STYLES),
        Subgenre("work_report", "業務報告・日報", (
            "設備の定期点検", "来客対応の記録", "検品作業の結果", "配送遅延の報告", "電話対応の件数", "売上の速報",
            "資料作成の進み具合", "研修受講の報告", "出張の報告", "現場の安全確認", "棚卸しの差異", "クレーム対応の経過"),
            {"label_colon": 0.4, "prose": 0.3, "bullets": 0.3}, _BUSINESS_STYLES),
        Subgenre("procedure", "作業手順書", (
            "複合機の紙詰まり対処", "経費申請の手順", "会議室予約の方法", "新入社員の初日案内", "パスワード再設定",
            "在庫登録", "冷蔵庫の温度記録", "消火器の点検", "データのバックアップ", "電話の取り次ぎ", "鍵の受け渡し",
            "レジ締めの手順"),
            {"numbered_steps": 0.7, "bullets": 0.2, "prose": 0.1}, _BUSINESS_STYLES),
        Subgenre("incident_log", "障害報告・システムログ", (
            "バッチ処理の失敗", "ディスク容量不足", "ログイン失敗の増加", "バックアップの遅延", "API 応答の遅延",
            "証明書の期限切れ", "メール送信キューの滞留", "プリンターの通信エラー", "監視アラートの誤検知",
            "再起動後の復旧確認", "在庫同期のずれ", "決済処理のタイムアウト"),
            {"key_value": 0.4, "log_lines": 0.4, "label_colon": 0.2}, {"formal": 0.6, "plain": 0.4},
            extra="ログ行のメッセージや値は状態・件数・時刻・処理名だけで構成し、ユーザー名やホスト名の代わりに「操作者」「対象サーバー」のような一般語を使う。"),
        Subgenre("form_sheet", "帳票・伝票", (
            "納品書の内訳", "発注一覧", "備品在庫表", "検査記録", "作業チェックリスト", "勤務シフト表", "会議室利用記録",
            "車両運行記録", "温度管理表", "修理受付票", "清掃点検表", "鍵の貸出記録"),
            {"label_colon": 0.5, "table_pipe": 0.4, "key_value": 0.1}, {"formal": 0.7, "polite": 0.3},
            extra="担当・記録者・承認などの欄には人名を入れず「早番」「夜勤担当」「当番」「担当者未定」「承認済」のような値を入れる。"),
        Subgenre("internal_notice", "社内連絡", (
            "年末年始の休業案内", "健康診断の受付", "駐車場の利用ルール", "節電のお願い", "社内システムの停止予告",
            "避難訓練の実施", "懇親会の出欠確認", "服装規定の緩和", "ゴミ出しルール", "来客時の受付方法",
            "共有プリンターの入れ替え", "残業申請の運用変更"),
            {"letter": 0.4, "prose": 0.3, "bullets": 0.3}, {"polite": 0.8, "formal": 0.2}),
        Subgenre("customer_record", "顧客対応記録", (
            "返品の申し出", "配送日の変更", "料金の問い合わせ", "故障の申告", "解約の相談", "予約の変更",
            "領収書の再発行", "苦情の受付", "使い方の質問", "キャンペーンの問い合わせ", "サイズ交換", "納期の確認"),
            {"label_colon": 0.5, "prose": 0.3, "bullets": 0.2}, _BUSINESS_STYLES,
            extra="相手は「お客様」「先方」、対応者は「担当者」「オペレーター」と書き、名前・会員番号・連絡先は書かない。"),
        Subgenre("schedule", "工程表・計画", (
            "改装工事の工程", "新製品の立ち上げ計画", "年間の研修計画", "引っ越し作業の段取り", "イベント準備の日程",
            "監査対応の日程", "システム移行の段階", "棚卸しの計画", "採用選考の日程", "展示会の準備", "決算作業の分担",
            "季節商品の入れ替え"),
            {"table_pipe": 0.4, "bullets": 0.3, "numbered_steps": 0.3}, _BUSINESS_STYLES),
    )),
    Genre("news", "報道・解説", 0.20, (
        Subgenre("economy", "経済の解説", (
            "物価上昇と家計", "為替の変動が輸入価格に与える影響", "最低賃金の引き上げ", "人手不足と省人化", "中古市場の拡大",
            "定額制サービスの広がり", "住宅ローン金利の動き", "電気料金の見直し", "観光需要の回復", "キャッシュレス決済の普及",
            "副業の広がり", "食品値上げの背景"),
            {"prose": 0.8, "bullets": 0.2}, {"plain": 0.6, "polite": 0.4}),
        Subgenre("science", "科学・技術ニュース風", (
            "新しい電池技術", "気候変動と農作物", "睡眠と記憶の研究", "深海生物の発見", "再生可能エネルギーの発電量",
            "人工知能の医療応用", "小惑星の観測", "感染症の予防研究", "食品ロス削減技術", "3Dプリンターの建築利用",
            "昆虫食の栄養価", "都市の緑化と気温"),
            {"prose": 0.8, "bullets": 0.2}, {"plain": 0.6, "polite": 0.4}),
        Subgenre("policy", "政策・制度の解説", (
            "子育て支援の拡充", "高齢者の運転免許制度", "ごみの分別ルール変更", "公共交通の運賃改定", "防災備蓄の推奨",
            "空き家対策", "学校給食費の扱い", "働き方の見直し", "消費税の軽減対象", "自転車のヘルメット着用",
            "電子申請の拡大", "投票所の設置基準"),
            {"prose": 0.7, "bullets": 0.3}, {"plain": 0.5, "polite": 0.5},
            extra="特定の国・自治体・政党・団体・人物には触れず、「自治体」「担当省庁」「与党」「有識者」のような一般語で書く。"),
        Subgenre("weather", "気象・防災", (
            "大雨への備え", "熱中症の予防", "台風接近時の行動", "積雪時の運転", "花粉の飛散予測", "地震後の安全確認",
            "停電時の過ごし方", "断水時の生活用水", "雷の注意点", "濃霧での運転", "河川増水時の避難", "猛暑日の外出"),
            {"prose": 0.6, "bullets": 0.4}, {"polite": 0.7, "plain": 0.3}),
        Subgenre("health", "健康コラム", (
            "睡眠の質を上げる方法", "塩分の摂り方", "運動習慣のつくり方", "目の疲れ対策", "冷え性の改善", "花粉症のセルフケア",
            "食物繊維の重要性", "腰痛の予防", "禁煙の効果", "水分補給の目安", "朝食を抜く影響", "座りすぎのリスク"),
            {"prose": 0.7, "bullets": 0.3}, {"polite": 0.7, "plain": 0.3}),
        Subgenre("tech_explainer", "技術解説", (
            "パスワード管理の考え方", "二段階認証とは", "クラウド保存のメリット", "写真の整理方法", "スマートフォンの電池寿命",
            "無線LANが遅い原因", "迷惑メールの判定", "表計算の関数の使い分け", "オンライン会議のマナー", "偽サイトの見分け方",
            "バックアップの三原則", "文字コードの違い"),
            {"prose": 0.6, "numbered_steps": 0.2, "bullets": 0.2}, {"polite": 0.7, "plain": 0.3}),
        Subgenre("history_general", "歴史・文化の一般論", (
            "稲作の広がり", "印刷技術の発展", "貨幣の始まり", "暦の成り立ち", "茶の湯の作法", "木造建築の技術", "和紙の作り方",
            "祭りの起源", "交易路と香辛料", "時計の歴史", "染め物の技法", "橋の構造の変化"),
            {"prose": 1.0}, {"plain": 0.6, "polite": 0.4},
            extra="王朝名・国名・人物名・都市名・年号（西暦の世紀表現は可）を出さず、一般的な流れとして書く。"),
    ), extra="特定の国・地域・団体・人物には触れず、一般的な傾向として書く。"),
    Genre("life", "生活・趣味・自然・食", 0.20, (
        Subgenre("recipe", "レシピ", (
            "肉じゃが", "卵焼き", "味噌汁の出汁", "鶏の唐揚げ", "炊き込みご飯", "ぬか漬け", "カレーの作り置き", "焼き魚のコツ",
            "おにぎりの握り方", "ホットケーキ", "冷やし中華", "豚汁", "きんぴらごぼう", "茶碗蒸し", "ジャム作り"),
            {"numbered_steps": 0.5, "prose": 0.3, "bullets": 0.2}, _PROSE_STYLES),
        Subgenre("nature_observation", "動物・植物の観察記", (
            "カラスの行動", "ヘビの脱皮", "カエルの鳴き声", "セミの羽化", "渡り鳥の群れ", "猫の毛づくろい", "カブトムシの飼育",
            "メダカの産卵", "タヌキの足跡", "ミツバチの巣", "スズメの子育て", "アオダイショウの観察", "ニシキヘビの飼育上の注意",
            "犬の散歩中のしぐさ", "トンボの見分け方", "アマガエルの色の変化"),
            {"prose": 0.8, "bullets": 0.2}, _PROSE_STYLES,
            extra="動物・植物の種名（アミメニシキヘビ、アオダイショウ、カラスなど）は自由に使ってよい。"),
        Subgenre("gardening", "園芸", (
            "トマトの支柱立て", "バジルの摘心", "紫陽花の剪定", "芝生の手入れ", "多肉植物の水やり", "球根の植え付け",
            "落葉の堆肥化", "害虫対策", "プランターの土の再利用", "朝顔の種取り", "苗の植え替え", "冬越しの準備"),
            {"prose": 0.5, "numbered_steps": 0.3, "bullets": 0.2}, _PROSE_STYLES),
        Subgenre("travel_diary", "旅行の感想", (
            "温泉宿の朝食", "山道の霧", "海沿いの町の市場", "夜行バスの過ごし方", "城下町の散策", "島へのフェリー",
            "紅葉の渓谷", "雪国の宿", "田園の自転車旅", "港の朝市", "湖畔のキャンプ", "終着駅の売店"),
            {"prose": 1.0}, _PROSE_STYLES,
            extra="地名・宿名・店名・路線名を書かず、「海沿いの町」「山あいの温泉地」「隣の県」のように書く。"),
        Subgenre("household", "家事のコツ", (
            "洗濯物の生乾き臭対策", "換気扇の掃除", "靴の消臭", "冷蔵庫の整理", "布団の干し方", "窓の結露対策", "食器の水垢",
            "フライパンの手入れ", "アイロンのかけ方", "収納の見直し", "排水口のぬめり", "カビの予防"),
            {"prose": 0.4, "numbered_steps": 0.3, "bullets": 0.3}, _PROSE_STYLES),
        Subgenre("parenting", "育児・家族", (
            "寝かしつけ", "離乳食の進め方", "公園遊びの安全", "絵本の選び方", "兄弟げんかの対応", "入園準備", "おやつのルール",
            "トイレトレーニング", "発熱時の観察", "習い事の選び方", "宿題の習慣づけ", "祖父母との過ごし方"),
            {"prose": 0.7, "bullets": 0.3}, _PROSE_STYLES,
            extra="家族は「長女」「息子」「夫」「祖母」のような関係語で呼び、名前・愛称を書かない。"),
        Subgenre("pet", "ペット", (
            "犬のしつけ", "猫の爪とぎ対策", "金魚の水替え", "ハムスターの温度管理", "老犬の介護", "猫の多頭飼い", "インコの放鳥",
            "犬の夏場の散歩", "動物病院への慣らし方", "迷子対策", "留守番のさせ方", "食欲不振の観察"),
            {"prose": 0.7, "bullets": 0.3}, _PROSE_STYLES,
            extra="ペットの名前を書かず「うちの犬」「先住猫」のように呼ぶ。"),
        Subgenre("sports", "スポーツ・運動", (
            "マラソン大会の完走記", "草野球の練習", "卓球の基礎", "登山の準備", "水泳のフォーム", "サッカー観戦の楽しみ方",
            "ジョギングの継続", "筋トレの記録", "自転車の整備", "ボウリングのコツ", "ラジオ体操の効果", "ストレッチの順番"),
            {"prose": 0.6, "numbered_steps": 0.2, "bullets": 0.2}, _PROSE_STYLES,
            extra="チーム名・選手名・大会名・競技場名を書かない。"),
        Subgenre("review", "商品レビュー", (
            "電気ケトル", "掃除機", "折りたたみ傘", "ワイヤレスイヤホン", "保温弁当箱", "ランニングシューズ", "加湿器",
            "デスクチェア", "フライパン", "モバイルバッテリー", "電動歯ブラシ", "ノートパソコンのスタンド"),
            {"prose": 0.6, "bullets": 0.4}, _PROSE_STYLES,
            extra="商品名・ブランド名・販売店名を書かず「この製品」「前に使っていた機種」のように書く。"),
    )),
    Genre("literary", "物語・随筆", 0.15, (
        Subgenre("classic_novel", "明治から昭和初期風の小説", (
            "雪の朝の縁側", "汽車を待つ停車場", "蔵の中の古い手紙", "病床の見舞い", "縄暖簾の店先", "夏の夕立", "卒業の日の校門",
            "提灯の灯る夜道", "初めての洋服", "帰郷の船", "夜学の帰り", "正月の餅つき"),
            {"prose": 1.0}, {"classic": 0.8, "plain": 0.2},
            extra=f"登場人物は「{LITERARY_ROLES}」のような呼び方だけで表し、名前を付けない。旧仮名遣いや古風な語（ねえや、おなご、到頭、恰度）を使ってよい。"),
        Subgenre("modern_novel", "現代小説の一節", (
            "深夜のコンビニ", "雨の日のバス停", "引っ越しの朝", "病院の待合室", "高架下の練習", "古い喫茶店", "終電の車内",
            "屋上の風", "河川敷の花火", "別れの改札", "図書館の閉館時間", "夜勤明けの朝食"),
            {"prose": 1.0}, {"plain": 0.7, "spoken": 0.3},
            extra="登場人物は「彼」「彼女」「少年」「店主」「老人」「運転手」のように呼び、名前を付けない。"),
        Subgenre("essay", "随筆", (
            "雨の匂い", "古い辞書", "早起きの効用", "手紙を書くこと", "台所の音", "冬の日差し", "読み終えた本", "歩くことについて",
            "祖母の味", "静けさの価値", "季節の変わり目", "使い込んだ道具"),
            {"prose": 1.0}, {"plain": 0.6, "polite": 0.4}),
        Subgenre("fairy_tale", "童話風", (
            "森の小さな郵便屋", "月を欲しがった猫", "石の下の宝", "雨雲を売る商人", "迷子の風", "眠らない街灯", "小人の靴屋",
            "川を渡るカメ", "歌をなくした鳥", "雪だるまの願い", "時計のない村", "おしゃべりなやかん"),
            {"prose": 1.0}, {"polite": 0.6, "plain": 0.4},
            extra="登場するのは「きつね」「おじいさん」「王さま」「粉屋」のような呼び方の者だけで、名前を付けない。"),
        Subgenre("diary", "日記", (
            "初雪の日", "試験前夜", "引っ越しの片付け", "体調を崩した週", "夕焼けがきれいだった日", "財布を落とした日",
            "久しぶりの映画", "新しい靴", "停電の夜", "給料日の買い物", "遠回りして帰った日", "早起きに失敗した朝"),
            {"prose": 1.0}, {"plain": 0.6, "spoken": 0.4},
            extra="登場する人は「同僚」「友人」「母」「店の人」のように呼び、名前を書かない。"),
    )),
    Genre("rules", "規約・FAQ・注意書き", 0.10, (
        Subgenre("terms", "利用規約", (
            "会員登録", "退会と返金", "禁止行為", "免責事項", "利用停止の条件", "料金改定", "著作権の帰属", "通知の方法",
            "未成年の利用", "アカウントの共有禁止", "サービス内容の変更", "問い合わせ窓口の案内"),
            {"numbered_steps": 0.5, "prose": 0.3, "bullets": 0.2}, {"formal": 0.8, "polite": 0.2},
            extra="サービス名・会社名は「本サービス」「当社」と書き、管轄裁判所や準拠法に国名・都市名を書かない。"),
        Subgenre("internal_rule", "社内規程", (
            "休暇の申請", "経費の精算", "情報機器の持ち出し", "服装規定", "残業の事前申請", "慶弔休暇", "在宅勤務の条件",
            "備品の貸出", "安全衛生", "入退室管理", "私用スマートフォンの扱い", "出張旅費の精算"),
            {"numbered_steps": 0.5, "bullets": 0.3, "prose": 0.2}, {"formal": 0.8, "polite": 0.2},
            extra="会社名・部署名は「当社」「所属部署」「管理担当」と書く。"),
        Subgenre("faq", "よくある質問", (
            "配送日数", "返品の条件", "支払い方法", "会員特典", "パスワードを忘れた場合", "領収書の発行", "キャンセル料",
            "営業時間", "予約の変更", "ポイントの有効期限", "ギフト包装", "サイズの選び方"),
            {"label_colon": 0.5, "prose": 0.3, "bullets": 0.2}, {"polite": 0.9, "formal": 0.1},
            extra="「Q: 」「A: 」の形で質問と回答を並べてよい。店名・商品名は「当店」「本商品」と書く。"),
        Subgenre("caution", "注意書き・取扱説明", (
            "電化製品の安全上の注意", "薬の服用方法", "洗濯表示の読み方", "食品の保存方法", "家具の組み立ての注意", "自転車の点検",
            "ガスコンロの使い方", "乾電池の廃棄", "充電器の使用条件", "浴室乾燥機の手入れ", "脚立の使い方", "電子レンジの禁止事項"),
            {"bullets": 0.5, "numbered_steps": 0.3, "prose": 0.2}, {"polite": 0.6, "formal": 0.4}),
        Subgenre("contract_clause", "契約条項の一般文", (
            "納期と検収", "支払い条件", "秘密保持", "損害賠償の範囲", "契約期間と更新", "解除条件", "再委託の禁止", "権利の帰属",
            "反社会的勢力の排除", "協議事項", "不可抗力", "通知義務"),
            {"numbered_steps": 0.7, "prose": 0.3}, {"formal": 1.0},
            extra="当事者は「甲」「乙」と書き、会社名・所在地・裁判所名を書かない。"),
    )),
    Genre("chat", "チャット・掲示板", 0.05, (
        Subgenre("role_chat", "役割ラベル付きチャット", (
            "注文の変更", "修理の依頼", "会議の時間調整", "資料の場所", "休みの相談", "使い方の質問", "待ち合わせの調整",
            "締め切りの確認", "引き継ぎの確認", "在庫の確認", "出欠の連絡", "遅刻の連絡"),
            {"chat_roles": 1.0}, {"spoken": 0.7, "polite": 0.3},
            extra="待ち合わせ場所は「駅前」「いつもの店」のように書き、地名・店名を出さない。"),
        Subgenre("board_post", "掲示板の投稿と返信", (
            "家電の選び方の相談", "資格勉強の進め方", "引っ越し先の選び方", "おすすめの読書法", "節約術", "睡眠の改善",
            "ペットの悩み", "職場の人間関係", "料理の失敗談", "運動の続け方", "衣替えのタイミング", "通勤時間の使い方"),
            {"chat_roles": 1.0}, {"spoken": 0.8, "plain": 0.2},
            extra="発言者は「投稿者」「返信1」「返信2」のように書き、ハンドル名やユーザー名を出さない。"),
        Subgenre("support_chat", "サポートチャット", (
            "ログインできない", "請求額の確認", "配送状況の確認", "解約手続き", "機器の初期化", "プラン変更", "領収書の発行",
            "アプリの不具合", "パスワード再設定", "返品の受付", "二重請求の疑い", "配送先の変更"),
            {"chat_roles": 1.0}, {"polite": 1.0},
            extra="発言者は「オペレーター」「お客様」と書き、名前・会員番号・注文番号・連絡先は「ご登録の情報」「対象のご注文」のように置き換える。"),
    )),
)

GENRE_BY_KEY: dict[str, Genre] = {g.key: g for g in GENRES}


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def build_user_prompt(row: dict) -> str:
    lo, hi = row["length_min"], row["length_max"]
    lines = [
        "次の条件で日本語の文書を1件書いてください。",
        f"- 種類: {row['genre_label']} / {row['subgenre_label']}",
        f"- 題材: {row['topic']}",
        f"- 形式: {FORMATS[row['format_key']]}",
        f"- 文体: {STYLES[row['style']]}",
        f"- 長さ: 全体で{lo}〜{hi}文字",
        f"- 書き出し: {row['opening']}",
    ]
    extras = [e for e in (row.get("genre_extra", ""), row.get("subgenre_extra", "")) if e]
    if extras:
        lines.append("- 追加の指示: " + " ".join(extras))
    lines.append(
        "禁止事項の再確認: 人名・地名（日本を含む国名も）・組織名・施設名・部署名・商品名などの固有名詞、"
        "住所・電話番号・メールアドレス・URL・番号列は一切書かない。人物や場所は役職・関係・一般名詞で表す。")
    lines.append('出力は {"lines": ["1行目", "2行目", ...]} の JSON だけ。1 行を配列の 1 要素にし、段落の区切りは "" の要素にする。')
    return "\n".join(lines)


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_plan(n: int, seed: int = 42, max_repeat: int = 3) -> list[dict]:
    """ジャンルの重み付きラウンドロビンで並べる。先頭 k 件を取っても比率が保たれる（pilot は先頭 200 件）。"""
    rng = random.Random(seed)
    credit = {g.key: 0.0 for g in GENRES}
    sub_cursor = {g.key: 0 for g in GENRES}
    combo_count: Counter = Counter()
    rows: list[dict] = []
    for i in range(n):
        for g in GENRES:
            credit[g.key] += g.weight
        gkey = max(credit, key=lambda k: (credit[k], k))
        credit[gkey] -= 1.0
        g = GENRE_BY_KEY[gkey]
        sub = g.subgenres[sub_cursor[gkey] % len(g.subgenres)]
        sub_cursor[gkey] += 1
        for _ in range(20):
            topic = rng.choice(sub.topics)
            fmt = _weighted_choice(rng, sub.formats)
            style = _weighted_choice(rng, sub.styles)
            length_key = _weighted_choice(rng, LENGTH_WEIGHTS)
            combo = (sub.key, topic, fmt, style)
            if combo_count[combo] < max_repeat:
                break
        combo_count[combo] += 1
        lo, hi = LENGTH_KEYS[length_key]
        row = {
            "request_id": f"neg_v8_{i:05d}", "index": i,
            "genre_key": g.key, "genre_label": g.label, "genre_extra": g.extra,
            "subgenre_key": sub.key, "subgenre_label": sub.label, "subgenre_extra": sub.extra,
            "topic": topic, "format_key": fmt, "style": style,
            "length_key": length_key, "length_min": lo, "length_max": hi, "max_tokens": MAX_TOKENS[length_key],
            "opening": rng.choice(OPENINGS),
            "plan_version": PLAN_VERSION, "prompt_version": PROMPT_VERSION,
        }
        row["prompt"] = build_user_prompt(row)
        row["prompt_sha256"] = sha256_short(SYSTEM_PROMPT + "\n" + row["prompt"])
        rows.append(row)
    return rows


def plan_manifest(rows: Sequence[dict], seed: int, plan_sha256: str) -> dict:
    return {
        "plan_version": PLAN_VERSION, "plan_tag": PLAN_TAG, "prompt_version": PROMPT_VERSION, "seed": seed, "n": len(rows),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/negatives_v8.py plan",
        "plan_file": PLAN_FILE, "plan_sha256": plan_sha256,
        "system_prompt": SYSTEM_PROMPT,
        "genre_weights": {g.key: g.weight for g in GENRES},
        "genre_counts": dict(Counter(r["genre_key"] for r in rows)),
        "subgenre_counts": dict(Counter(r["subgenre_key"] for r in rows)),
        "format_counts": dict(Counter(r["format_key"] for r in rows)),
        "style_counts": dict(Counter(r["style"] for r in rows)),
        "length_counts": dict(Counter(r["length_key"] for r in rows)),
        "length_ranges": LENGTH_KEYS, "max_tokens": MAX_TOKENS,
        "sampling_recommended": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                 "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "output_schema": OUTPUT_SCHEMA,
    }


# 1 行 1 要素。段落の区切りは "" 要素。ローカルで "\n".join して text にする（parsed_text）
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {"lines": {"type": "array", "minItems": 1, "maxItems": 80,
                             "items": {"type": "string", "maxLength": 600}}},
    "required": ["lines"], "additionalProperties": False,
}


def parsed_text(parsed) -> str | None:
    """raw 行の parsed から本文を取る。v3 の {"lines": [...]} と旧 {"text": ...} の両方を受ける。"""
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("lines"), list):
        text = "\n".join(str(x).rstrip() for x in parsed["lines"]).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text or None
    text = parsed.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. 選別（決定的規則）
# ---------------------------------------------------------------------------

_PREF_FULL: tuple[str, ...] = tuple(sorted({p for p, _ in PREF_CITIES}))
# 短縮形。「三重」（三重の意）「大分」（だいぶ）は一般語と衝突するので正式名だけ
_PREF_SHORT: tuple[str, ...] = tuple(sorted({
    (p if p == "北海道" else re.sub(r"[都府県]$", "", p)) for p in _PREF_FULL
} - {"三重", "大分"}))
_CITY_FULL: tuple[str, ...] = tuple(sorted({re.sub(r"(市|区).*$", r"\1", c) for _, c in PREF_CITIES}))
# 市名の語幹（区名と一般語の「柏」「港」「中央」は除く）
_CITY_STEM_EXCLUDE = {"柏", "港", "中央", "大分", "新宿", "世田谷", "品川", "杉並", "練馬"}
_CITY_STEM: tuple[str, ...] = tuple(sorted({
    re.sub(r"市.*$", "", c) for _, c in PREF_CITIES if c.endswith("市") or "市" in c
} - _CITY_STEM_EXCLUDE | {
    "横浜", "名古屋", "神戸", "北九州", "川崎", "さいたま", "千葉", "堺", "浜松", "静岡", "岡山", "熊本", "鹿児島", "仙台", "札幌",
    "福岡", "広島", "京都", "大阪", "東京", "那覇", "金沢", "新潟", "長崎",
}))
COUNTRIES_REGIONS: tuple[str, ...] = (
    "アメリカ", "米国", "イギリス", "英国", "フランス", "ドイツ", "イタリア", "スペイン", "ロシア", "中国", "韓国", "北朝鮮", "台湾",
    "インド", "ブラジル", "カナダ", "オーストラリア", "メキシコ", "インドネシア", "ベトナム", "フィリピン", "マレーシア", "トルコ",
    "イラン", "イラク", "イスラエル", "エジプト", "ウクライナ", "ポーランド", "オランダ", "ベルギー", "スイス", "スウェーデン",
    "ノルウェー", "デンマーク", "フィンランド", "ギリシャ", "ポルトガル", "アルゼンチン", "チリ", "サウジアラビア", "ニュージーランド",
    "シンガポール", "香港", "アフリカ", "ヨーロッパ", "欧州", "アジア", "東アジア", "東南アジア", "南アジア", "中東", "北米", "南米",
    "中南米", "オセアニア", "欧米", "西欧", "東欧", "北欧", "南欧",
    "関東", "関西", "近畿", "東北地方", "九州", "四国", "北陸", "中部地方", "東海地方", "甲信越", "首都圏", "京阪神", "中国地方",
    "ワシントン", "ニューヨーク", "ロンドン", "パリ", "ベルリン", "ローマ", "モスクワ", "北京", "上海", "ソウル", "台北", "バンコク",
    "シドニー", "ロサンゼルス", "サンフランシスコ", "シカゴ", "トロント", "ドバイ", "ハワイ",
    "富士山", "琵琶湖", "太平洋", "大西洋", "インド洋", "日本海", "瀬戸内海", "利根川", "信濃川", "淀川", "阿蘇山", "桜島", "屋久島",
    "知床", "尾瀬", "立山", "白神山地",
)
PLACE_SURFACES: tuple[str, ...] = tuple(sorted(set(_PREF_FULL) | set(_PREF_SHORT) | set(_CITY_FULL) | set(_CITY_STEM)
                                              | set(COUNTRIES_REGIONS), key=lambda s: (-len(s), s)))
_PLACE_RE = re.compile("|".join(map(re.escape, PLACE_SURFACES)))
_PLACE_DOMESTIC = frozenset(_PREF_FULL) | frozenset(_PREF_SHORT) | frozenset(_CITY_FULL) | frozenset(_CITY_STEM)
# 国名・地方名の派生語（中国語・フランス料理・アメリカ人・イタリア風・ローマ字・ギリシャ文字）は地名として数えない。
# 国内地名には適用しない（北海道産は地名）
_DERIVATION_CHARS = "語料式風産製人字数文"
_KATA_CHARS_RE = re.compile(r"[ァ-ヶー]")
# 「日本」は日本語・日本人・日本酒などの複合語だけ許す
_NIHON_RE = re.compile(r"日本(?![語人酒茶式製風画舞髪刀])")


def place_name_hits(t: str) -> list[str]:
    """NFKC 済みテキスト中の地名。カタカナ地名はカタカナ連続の全体と一致するときだけ（パリパリ・ローマ字・アメリカザリガニは除く）。"""
    hits: list[str] = []
    for m in _PLACE_RE.finditer(t):
        s = m.group(0)
        if _KATA_CHARS_RE.match(s):
            left = m.start() > 0 and _KATA_CHARS_RE.match(t[m.start() - 1])
            right = m.end() < len(t) and _KATA_CHARS_RE.match(t[m.end()])
            if left or right:
                continue
        if s not in _PLACE_DOMESTIC and m.end() < len(t) and t[m.end()] in _DERIVATION_CHARS:
            continue
        hits.append(s)
    return hits

KNOWN_ORGS: tuple[str, ...] = (
    "国連", "国際連合", "世界保健機関", "世界銀行", "日本銀行", "日銀", "東証", "国会", "衆議院", "参議院", "下院", "上院", "内閣府",
    "財務省", "経済産業省", "経産省", "総務省", "外務省", "文部科学省", "文科省", "厚生労働省", "厚労省", "国土交通省", "国交省",
    "環境省", "防衛省", "法務省", "農林水産省", "農水省", "警察庁", "消防庁", "気象庁", "金融庁", "国税庁", "特許庁", "宮内庁",
    "最高裁", "最高裁判所", "自民党", "自由民主党", "民主党", "公明党", "共産党",
    "フジテレビ", "日本テレビ", "テレビ朝日", "テレビ東京", "読売", "朝日新聞", "毎日新聞", "産経", "日経", "日本経済新聞",
    "トヨタ", "ソニー", "パナソニック", "ホンダ", "日産", "三菱", "三井", "住友", "楽天", "ソフトバンク", "ドコモ", "東京電力", "関西電力",
    "グーグル", "アップル", "マイクロソフト", "アマゾン", "フェイスブック", "ツイッター", "ユーチューブ", "インスタグラム", "ネットフリックス",
    "ユニクロ", "無印良品", "イオンモール", "セブンイレブン", "ローソン", "ファミリーマート", "スターバックス", "マクドナルド", "ヤマト運輸",
    "佐川急便", "日本郵便", "ゆうちょ", "みずほ", "三菱UFJ", "りそな", "ジャニーズ", "吉本興業", "任天堂", "スクウェア", "カプコン",
    "Google", "Apple", "Microsoft", "Amazon", "Facebook", "Meta", "Twitter", "YouTube", "Instagram", "TikTok", "Netflix", "LINE",
    "NHK", "TBS", "JR", "ANA", "JAL", "NTT", "KDDI", "WHO", "NATO", "IMF", "OECD", "FIFA", "IOC", "NASA", "JAXA",
)
# ASCII の組織名は英字の途中（ANALYSIS の ANA、Metadata の Meta）に一致させない
_KNOWN_ORG_RE = re.compile("|".join(
    (rf"(?<![A-Za-z]){re.escape(s)}(?![A-Za-z])" if s.isascii() else re.escape(s))
    for s in sorted(KNOWN_ORGS, key=lambda s: (-len(s), s))))
# 法人格の明示語。「株式会社の設立手続き」のような一般文も落ちるが、負例の損失としては小さい
CORPORATE_MARKERS: tuple[str, ...] = (
    "株式会社", "(株)", "（株）", "有限会社", "合同会社", "合資会社", "一般社団法人", "一般財団法人", "公益財団法人", "公益社団法人",
    "特定非営利活動法人", "NPO法人", "学校法人", "医療法人", "社会福祉法人", "独立行政法人", "国立大学法人", "宗教法人", "税理士法人",
    "弁護士法人", "監査法人", "ホールディングス",
)
_CORPORATE_RE = re.compile("|".join(map(re.escape, CORPORATE_MARKERS)))
DEPARTMENTS: tuple[str, ...] = (
    "総務課", "総務部", "人事部", "人事課", "経理部", "経理課", "営業部", "営業課", "企画部", "企画課", "広報部", "広報課", "法務部",
    "開発部", "技術部", "製造部", "品質保証部", "品質管理部", "購買部", "資材部", "情報システム部", "システム部", "管理部", "管理課",
    "業務部", "業務課", "庶務課", "会計課", "財務部", "秘書室", "監査室", "社長室", "市民課", "戸籍課", "税務課", "福祉課",
    "保険年金課", "介護保険課", "子育て支援課", "環境課", "建設課", "都市計画課", "教育委員会", "学務課", "生活安全課",
    "地域包括支援センター", "経営企画部", "海外事業部", "カスタマーサポート部", "商品開発部", "生産管理部", "物流部", "調達部",
)
_DEPT_RE = re.compile("|".join(map(re.escape, sorted(DEPARTMENTS, key=lambda s: (-len(s), s)))))

_SURNAMES = tuple(sorted(set(SURNAMES_KANJI), key=lambda s: (-len(s), s)))
_GIVEN_KANJI = tuple(sorted(set(GIVEN_KANJI) | set(GIVEN_KANJI_F), key=lambda s: (-len(s), s)))
_GIVEN_LONG = tuple(g for g in _GIVEN_KANJI if len(g) >= 3)
# 一般名詞と同形のカタカナ人名（ミラー = 鏡、ブラウン = 色、リン = 元素、アリ = 虫、ハンセン = 病名…）は人名判定から外す
_KATA_NAME_EXCLUDE = frozenset({"ミラー", "ブラウン", "マックス", "メイ", "リン", "ワン", "アリ", "ハンセン", "レオ", "サラ"})
_KATA_NAMES = (frozenset(GIVEN_KATA) | frozenset(SURNAMES_KATA)) - _KATA_NAME_EXCLUDE
_HONORIFIC = r"(?:さん|様|氏|君|くん|ちゃん|殿|先生|夫人|部長|課長|係長|主任|社長|店長|医師|教授|選手|監督|議員|大臣|先輩|後輩)"
_SURNAME_CTX_RE = re.compile(rf"(?:{'|'.join(map(re.escape, _SURNAMES))}){_HONORIFIC}")
_FULLNAME_RE = re.compile(rf"(?:{'|'.join(map(re.escape, _SURNAMES))})[ 　]?(?:{'|'.join(map(re.escape, _GIVEN_KANJI))})(?![一-龥])")
_GIVEN_CTX_RE = re.compile(rf"(?:{'|'.join(map(re.escape, _GIVEN_KANJI))})(?:さん|ちゃん|君|くん|様|氏)")
_GIVEN_LONG_RE = re.compile("|".join(map(re.escape, _GIVEN_LONG))) if _GIVEN_LONG else None
_NAME_LABEL_RE = re.compile(
    rf"(?:担当|氏名|名前|記録者|担当者|作成者|承認者|申請者|報告者|責任者|宛名|差出人|operator|user|author|owner|assignee|reporter)"
    rf"\s*[:：=]\s*(?:{'|'.join(map(re.escape, _SURNAMES + _GIVEN_KANJI))})")
_KATA_RUN_RE = re.compile(r"[ァ-ヶー]+")
_HONORIFIC_AFTER_RE = re.compile(_HONORIFIC)

_MENTION_RE = re.compile(r"(?<![A-Za-z0-9._%+\-])@[A-Za-z][A-Za-z0-9_.]{1,30}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\d-])0\d{1,4}-\d{1,4}-\d{3,4}(?![\d-])|(?<!\d)0\d{9,10}(?!\d)|\+\d{1,3}[- ]\d{1,4}[- ]\d{3,4}[- ]\d{3,4}")
# 〒なしの 3-4 桁は「100-2000円」のような範囲表現と衝突するので、直後が単位語なら郵便番号と見ない
_POSTAL_RE = re.compile(r"〒\s?\d{3}-?\d{4}|(?<![\d-])\d{3}-\d{4}(?![\d-])(?![ 　]?[円個件人本台枚回名年月日時分秒%kgmc])")
_BANCHI_RE = re.compile(r"[一-龥ぁ-んァ-ヶ]{1,6}\d{1,4}丁目|\d{1,4}丁目\d{1,4}番|\d{1,4}番地|\d{1,4}-\d{1,4}-\d{1,4}号")
_FOREIGN_SCRIPT_RE = re.compile(r"[Ѐ-ӿ؀-ۿ֐-׿฀-๿가-힯㄰-㆏]")
# 日本語では使わない簡体字（这个们说时为对还现样关从发电让过种经）
_SIMPLIFIED_RE = re.compile(r"[这个们说时为对还现样关从发电让过种经]")
_TAG_RE = re.compile(r"</?(?:PERSON|ORG|FACILITY|ADDRESS|LOCATION)>|\[\[|\]\]")

HANDLE_GENERIC_VALUES: frozenset[str] = frozenset(
    "admin administrator system root test guest null none unknown auto batch cron api bot service support ops dev prod "
    "staging default anonymous operator staff user manager owner night_shift day_shift shift_a shift_b team_a team_b "
    "n/a na tbd pending done ok ng error warn info debug".split())


@dataclass
class Verdict:
    decision: str                      # keep / reject / review
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    ner_hits: list[dict] = field(default_factory=list)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def deterministic_reject_reasons(text: str) -> tuple[list[str], list[str]]:
    """規則で確実に落とす理由と、その証拠となる表層を返す。"""
    t = normalize(text)
    reasons: list[str] = []
    evidence: list[str] = []

    def hit(reason: str, surfaces: Iterable[str]) -> None:
        surfaces = [s for s in surfaces if s]
        if surfaces:
            reasons.append(reason)
            evidence.extend(f"{reason}:{s}" for s in sorted(set(surfaces))[:5])

    if _FOREIGN_SCRIPT_RE.search(t):
        hit("foreign_script", [_FOREIGN_SCRIPT_RE.search(t).group(0)])
    if _SIMPLIFIED_RE.search(t):
        hit("simplified_chinese", _SIMPLIFIED_RE.findall(t))
    if _TAG_RE.search(t):
        hit("tag_residue", _TAG_RE.findall(t))
    hit("url", _URL_RE.findall(t))
    hit("email", _EMAIL_RE.findall(t))
    hit("phone", _PHONE_RE.findall(t))
    hit("postal_code", _POSTAL_RE.findall(t))
    hit("address_banchi", _BANCHI_RE.findall(t))
    hit("mention", _MENTION_RE.findall(t))
    hit("place_name", place_name_hits(t))
    hit("place_name", ["日本"] if _NIHON_RE.search(t) else [])
    hit("known_org", _KNOWN_ORG_RE.findall(t))
    hit("corporate_marker", _CORPORATE_RE.findall(t))
    hit("department_name", _DEPT_RE.findall(t))
    hit("surname_with_title", _SURNAME_CTX_RE.findall(t))
    hit("full_name", _FULLNAME_RE.findall(t))
    hit("given_name_with_title", _GIVEN_CTX_RE.findall(t))
    if _GIVEN_LONG_RE is not None:
        hit("given_name", _GIVEN_LONG_RE.findall(t))
    hit("name_after_label", _NAME_LABEL_RE.findall(t))
    kata_hits = [run for run in _KATA_RUN_RE.findall(t) if run in _KATA_NAMES]
    hit("katakana_name", kata_hits)
    hit("eval_surface", [s for s in EVAL_SURFACES if s in text or s in t])
    return sorted(set(reasons)), evidence


def length_reason(text: str, hard_min: int = 20, hard_max: int = 900) -> str | None:
    n = len(text)
    if n < hard_min:
        return "too_short"
    if n > hard_max:
        return "too_long"
    return None


# ---------------------------------------------------------------------------
# 3. 選別（NER の兆候 → review）
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIRS: dict[str, str] = {
    "char8ep": "outputs_ruri-v3-30m-charhead-v45e-frompre/best",
    "v7": "outputs_ruri-v3-30m-charhead-v7-frompre/best",
    "token": "outputs_ruri-v3-30m-v45e/best",
}


def build_detector(path: str, device: str = "cpu"):
    """eval_all.build_detector と同じ出し分け。データフォルダを import 時に要求しないためここに写す。"""
    from privacy_masker.detectors import CharHeadNerDetector, TransformersNerDetector
    if os.path.exists(os.path.join(path, "char_head_config.json")):
        return CharHeadNerDetector(model=path, device=device)
    return TransformersNerDetector(model=path, device=device)


def load_detectors(names: Sequence[str], drive: str | None = None) -> dict[str, object]:
    drive = drive or drive_dir()
    out = {}
    for name in names:
        rel = DEFAULT_MODEL_DIRS.get(name, name)
        path = rel if os.path.isabs(rel) else os.path.join(drive, rel)
        if not os.path.isdir(path):
            raise SystemExit(f"モデルディレクトリがありません: {path}")
        out[name] = build_detector(path)
    return out


def load_public_knowledge(drive: str | None = None):
    drive = drive or drive_dir(require=False)
    path = os.path.join(drive, "jawiki_titles_ns0.pmk")
    if not os.path.exists(path):
        return None
    from privacy_masker.annotators.knowledge import PublicKnowledge
    return PublicKnowledge.load(path)


def ner_hits(text: str, detectors: dict[str, object], handle_detector=None) -> list[dict]:
    hits = []
    for name, det in detectors.items():
        for e in det.detect(text):
            hits.append({"detector": name, "type": e.entity_type, "start": e.start, "end": e.end,
                         "surface": text[e.start:e.end], "score": round(float(e.score), 3)})
    if handle_detector is not None:
        for e in handle_detector.detect(text):
            hits.append({"detector": "handle", "type": e.entity_type, "start": e.start, "end": e.end,
                         "surface": text[e.start:e.end], "score": round(float(e.score), 3)})
    return hits


_ORG_SUFFIXES = tuple(sorted(set(ORG_KNOWN_SUFFIXES) | {
    "株式会社", "学院", "高校", "高等学校", "中学校", "小学校", "クリニック", "新聞", "テレビ", "製作所", "工業", "商事", "運輸",
    "電機", "グループ", "法律事務所", "事務所", "センター", "商店", "書店", "薬局", "自治会",
}, key=lambda s: (-len(s), s)))
_GENERIC_ORG_PREFIXES = frozenset({
    "国立", "私立", "公立", "市立", "県立", "都立", "府立", "町立", "村立", "総合", "大学", "地方", "都市", "中央", "地域", "近所",
    "近隣", "生命", "損害", "医療", "自動車", "火災", "健康", "介護", "年金", "労働", "生活", "農業", "漁業", "森林", "信用",
    "地元", "最寄", "最寄り", "かかりつけ", "町の", "村の", "市の", "県の", "当", "同", "本", "各", "全", "新", "旧", "元", "某", "小", "大",
})


_GENERIC_ORG_LEAD_RE = re.compile(r"^(?:近所の|町の|村の|市の|県の|隣の|地元の|最寄りの|最寄の|かかりつけの|うちの|この|その|あの|当|同|本|各|某|元|旧|新)+")


def _org_like(surface: str) -> bool:
    """法人・施設の語尾を持ち、語尾の前が一般語でないもの（曙製作所・青葉総合病院・みなと書店）。
    「市立病院」「近所の薬局」「総合病院」「当センター」は一般語として False。"""
    for suf in _ORG_SUFFIXES:
        if surface.endswith(suf) and len(surface) > len(suf):
            prefix = _GENERIC_ORG_LEAD_RE.sub("", surface[: -len(suf)])
            return bool(prefix) and prefix not in _GENERIC_ORG_PREFIXES
    return False


ROLE_WORDS: tuple[str, ...] = tuple(LITERARY_ROLES.split("、")) + (
    "きつね", "おじいさん", "おばあさん", "王さま", "王様", "粉屋", "小人", "商人", "郵便屋", "靴屋", "店員", "客", "質問者",
    "回答者", "先輩", "後輩", "上司", "部下", "担当", "依頼者", "オペレーター", "お客様", "投稿者", "返信", "彼", "彼女", "少年",
    "少女", "店主", "老人", "運転手", "同僚", "友人", "長女", "長男", "息子", "娘", "夫", "妻", "祖母", "祖父", "甲", "乙",
    "担当者", "責任者", "当番", "早番", "夜勤", "講師", "受講者", "店長", "課長", "部長", "主任", "係", "受付", "看護師", "医師",
)


def _prompt_vocab() -> frozenset[str]:
    """プロンプトで使うよう指示した語（題材・役割語）。これらは人名ではないので NER の一致があっても負例に残す。"""
    vocab: set[str] = set(ROLE_WORDS)
    for g in GENRES:
        for sub in g.subgenres:
            for topic in sub.topics:
                vocab.add(topic)
                vocab.update(p for p in re.split(r"[のとや・、（）]", topic) if len(p) >= 2)
            vocab.update(re.findall(r"「([^」]+)」", sub.extra))
    vocab.update(re.findall(r"「([^」]+)」", SYSTEM_PROMPT))
    return frozenset(v for v in vocab if len(v) >= 2)


PROMPT_VOCAB: frozenset[str] = _prompt_vocab()


def in_prompt_vocab(surface: str) -> bool:
    if len(surface) < 2:
        return False
    return any(surface in v or v in surface for v in PROMPT_VOCAB)


def review_reasons(text: str, hits: list[dict], knowledge=None) -> tuple[list[str], list[str]]:
    """規則で決め切れない強い兆候。既定では除外し、review_excluded.csv で人が戻す。"""
    reasons: list[str] = []
    evidence: list[str] = []
    ner = [h for h in hits if h["detector"] != "handle"]
    by_detector = defaultdict(list)
    for h in ner:
        by_detector[h["detector"]].append(h)
    names = sorted(by_detector)
    # (a) 2 つ以上の NER が同じ型の PERSON / ADDRESS を重ねて出す。
    #     対象外: 1 文字、平仮名だけ（ねえや・おなご）、プロンプトで使うよう指示した題材・役割語（粉屋・書生・縄暖簾・カメ）。
    #     これらは役割語・一般語の誤検出で、まさに負例で教えたい語。
    if len(names) >= 2:
        for typ in ("PERSON", "ADDRESS"):
            spans = {n: [(h["start"], h["end"]) for h in by_detector[n] if h["type"] == typ] for n in names}
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    for s1, e1 in spans[a]:
                        surface = text[s1:e1]
                        if len(surface) < 2 or not _NAME_SCRIPT_RE.search(surface) or in_prompt_vocab(surface):
                            continue
                        if any(s1 < e2 and s2 < e1 for s2, e2 in spans[b]):
                            reasons.append(f"agreed_{typ.lower()}")
                            evidence.append(f"agreed_{typ.lower()}:{surface}")
    # (b) PERSON の直後に敬称・肩書き
    for h in ner:
        if h["type"] == "PERSON" and len(h["surface"]) >= 2 and _HONORIFIC_AFTER_RE.match(text, h["end"]):
            reasons.append("person_with_title")
            evidence.append(f"person_with_title:{h['surface']}")
    # (c) 法人・施設の語尾を持つ ORG / FACILITY（一般語の接頭は除く）
    for h in ner:
        if h["type"] in ("ORG", "FACILITY") and _org_like(normalize(h["surface"])):
            reasons.append("org_like_surface")
            evidence.append(f"org_like_surface:{h['surface']}")
    # (d) 公知の見出し語に一致する 2 文字以上の LOCATION（1 文字の「区」「係」は誤検出として残す）。
    #     平仮名だけ（山あい の「あい」）と、語の途中を切ったもの（大通り の「大通」= 前後 1〜2 字を足しても見出し語）は対象外
    if knowledge is not None:
        for h in ner:
            s, e, surface = h["start"], h["end"], h["surface"]
            if h["type"] != "LOCATION" or len(surface) < 2 or surface not in knowledge:
                continue
            if not _NAME_SCRIPT_RE.search(surface) or in_prompt_vocab(surface):
                continue
            if any(text[max(0, s - k):e] in knowledge for k in (1, 2) if s - k >= 0) or \
                    any(text[s:e + k] in knowledge for k in (1, 2) if e + k <= len(text)):
                continue
            reasons.append("public_location")
            evidence.append(f"public_location:{surface}")
    # (e) ハンドル名らしい値（一般的な状態語は除く）
    for h in hits:
        if h["detector"] == "handle" and h["surface"].lower() not in HANDLE_GENERIC_VALUES:
            reasons.append("handle_like")
            evidence.append(f"handle_like:{h['surface']}")
    return sorted(set(reasons)), sorted(set(evidence))


_NAME_SCRIPT_RE = re.compile(r"[一-龥々ァ-ヶA-Za-z]")


# ---------------------------------------------------------------------------
# 4. 近重複と書き出しの反復
# ---------------------------------------------------------------------------

def _grams(text: str, n: int = 3) -> set[str]:
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def symmetric_ratio(a: str, b: str) -> float:
    f = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    g = difflib.SequenceMatcher(None, b, a, autojunk=False).ratio()
    return max(f, g)


def near_duplicate_pass(texts: Sequence[str], hard: float = 0.88, review: float = 0.84, min_chars: int = 30,
                        jaccard_floor: float = 0.30, max_candidates: int = 40) -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    """順に受け入れ、既採用と近すぎるものを落とす。戻り値: (落とした index → (相手, 類似度), 要注意 index → (相手, 類似度))。"""
    norm = [normalize(t).casefold() for t in texts]
    grams = [_grams(t) for t in norm]
    index: dict[str, list[int]] = defaultdict(list)
    dropped: dict[int, tuple[int, float]] = {}
    flagged: dict[int, tuple[int, float]] = {}
    for i, g in enumerate(grams):
        cand: Counter = Counter()
        for x in g:
            for j in index.get(x, ()):
                cand[j] += 1
        best: tuple[int, float] | None = None
        for j, shared in cand.most_common(max_candidates):
            jacc = shared / max(1, len(g) + len(grams[j]) - shared)
            if jacc < jaccard_floor:
                continue
            if min(len(norm[i]), len(norm[j])) < min_chars:
                continue
            score = symmetric_ratio(norm[i], norm[j])
            if best is None or score > best[1]:
                best = (j, score)
        if best is not None and best[1] >= hard:
            dropped[i] = best
            continue
        if best is not None and best[1] >= review:
            flagged[i] = best
        for x in g:
            index[x].append(i)
    return dropped, flagged


def opening_signature(text: str, chars: int = 20) -> str:
    return normalize(text).casefold().replace("\n", " ")[:chars]


# ---------------------------------------------------------------------------
# 5. filter 本体
# ---------------------------------------------------------------------------

def load_raw(paths: Sequence[str]) -> dict[str, dict]:
    """Colab の OUT_DIR かファイルを複数受け、request_id ごとに後勝ちで束ねる。"""
    rows: dict[str, dict] = {}
    for p in paths:
        path = os.path.join(p, RAW_FILE) if os.path.isdir(p) else p
        if not os.path.exists(path):
            raise SystemExit(f"raw 出力がありません: {path}")
        for r in read_jsonl(path):
            rid = r.get("request_id")
            if rid:
                rows[rid] = r
    return rows


def to_training_record(text: str, plan: dict) -> dict:
    return {"text": text, "entities": [], "source": SOURCE_NAME,
            "variant": f"{plan['genre_key']}/{plan['subgenre_key']}/{plan['format_key']}"}


def run_filter(raw_rows: dict[str, dict], plan_rows: Sequence[dict], *, target: int, detectors: dict[str, object] | None,
               handle_detector=None, knowledge=None, include_ids: Iterable[str] = (), quota_by_genre: bool = True,
               progress: Callable[[str], None] | None = None) -> dict:
    include = set(include_ids)
    plan_by_id = {p["request_id"]: p for p in plan_rows}
    verdicts: dict[str, Verdict] = {}
    texts: dict[str, str] = {}          # 規則を通った文書（近重複・採用の対象）
    texts_all: dict[str, str] = {}      # JSON が取れた全文書（CSV の本文表示用）
    counts: Counter = Counter()

    for plan in plan_rows:
        rid = plan["request_id"]
        raw = raw_rows.get(rid)
        if raw is None:
            counts["not_generated"] += 1
            continue
        counts["generated"] += 1
        text = parsed_text(raw.get("parsed"))
        if text is None:
            verdicts[rid] = Verdict("reject", ["structured_output_failed"], [str(raw.get("structured_error", ""))[:80]])
            continue
        texts_all[rid] = text
        reason = length_reason(text)
        if reason:
            verdicts[rid] = Verdict("reject", [reason], [f"{reason}:{len(text)}"])
            continue
        reasons, evidence = deterministic_reject_reasons(text)
        if reasons:
            verdicts[rid] = Verdict("reject", reasons, evidence)
            continue
        hits = ner_hits(text, detectors, handle_detector) if detectors or handle_detector else []
        r_reasons, r_evidence = review_reasons(text, hits, knowledge)
        if r_reasons and rid not in include:
            verdicts[rid] = Verdict("review", r_reasons, r_evidence, hits)
            continue
        verdicts[rid] = Verdict("keep", ["included_from_review"] if r_reasons else [], r_evidence, hits)
        texts[rid] = text

    if progress:
        progress(f"規則の選別: keep {len(texts)} / reject {sum(v.decision == 'reject' for v in verdicts.values())} / "
                 f"review {sum(v.decision == 'review' for v in verdicts.values())}")

    # 近重複・書き出し反復（計画順 = ジャンル均衡順）
    order = [p["request_id"] for p in plan_rows if p["request_id"] in texts]
    dropped, flagged = near_duplicate_pass([texts[r] for r in order])
    for i, (j, score) in dropped.items():
        v = verdicts[order[i]]
        v.decision, v.reasons = "reject", ["near_duplicate"]
        v.evidence = [f"near_duplicate:{order[j]}:{score:.3f}"]
    for i, (j, score) in flagged.items():
        verdicts[order[i]].evidence.append(f"near_duplicate_review:{order[j]}:{score:.3f}")
    opening_seen: Counter = Counter()
    for rid in order:
        if verdicts[rid].decision != "keep":
            continue
        sig = (plan_by_id[rid]["genre_key"], opening_signature(texts[rid]))
        opening_seen[sig] += 1
        if opening_seen[sig] > 5:
            v = verdicts[rid]
            v.decision, v.reasons, v.evidence = "reject", ["opening_repeat"], [f"opening_repeat:{sig[1]}"]

    # 採用（ジャンル割当 → 余りは計画順）
    keep_ids = [rid for rid in order if verdicts[rid].decision == "keep"]
    selected: list[str] = []
    if quota_by_genre:
        quota = {g.key: int(round(target * g.weight)) for g in GENRES}
        taken: Counter = Counter()
        rest = []
        for rid in keep_ids:
            gk = plan_by_id[rid]["genre_key"]
            if taken[gk] < quota[gk]:
                selected.append(rid)
                taken[gk] += 1
            else:
                rest.append(rid)
        for rid in rest:
            if len(selected) >= target:
                break
            selected.append(rid)
        shortfall = {k: quota[k] - taken[k] for k in quota if taken[k] < quota[k]}
    else:
        selected = keep_ids[:target]
        shortfall = {}
    selected_set = set(selected)
    for rid in keep_ids:
        if rid not in selected_set:
            verdicts[rid].decision = "surplus"

    records = [to_training_record(texts[rid], plan_by_id[rid]) for rid in selected]
    return {"verdicts": verdicts, "texts": texts, "texts_all": texts_all, "records": records, "selected": selected,
            "counts": counts, "shortfall": shortfall, "near_dup_dropped": len(dropped), "near_dup_flagged": len(flagged)}


def write_filter_outputs(result: dict, plan_rows: Sequence[dict], out_dir: str, *, target: int, model_names: Sequence[str],
                         raw_paths: Sequence[str]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    plan_by_id = {p["request_id"]: p for p in plan_rows}
    verdicts: dict[str, Verdict] = result["verdicts"]
    texts = result["texts"]
    texts_all = result.get("texts_all", texts)
    write_jsonl(os.path.join(out_dir, "negatives_v8.jsonl"), result["records"])

    def row_of(rid: str, v: Verdict) -> dict:
        p = plan_by_id[rid]
        body = texts_all.get(rid) or ""
        return {"request_id": rid, "genre": p["genre_key"], "subgenre": p["subgenre_key"], "format": p["format_key"],
                "decision": v.decision, "reasons": ";".join(v.reasons), "evidence": " | ".join(v.evidence[:8]),
                "chars": len(body), "text": body.replace("\n", "⏎")[:200]}

    def write_csv(name: str, rows: list[dict]) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["request_id", "genre", "subgenre", "format", "decision", "reasons",
                                              "evidence", "chars", "text"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv("rejected.csv", [row_of(rid, v) for rid, v in verdicts.items() if v.decision == "reject"])
    write_csv("review_excluded.csv", [row_of(rid, v) for rid, v in verdicts.items() if v.decision == "review"])
    write_csv("surplus.csv", [row_of(rid, v) for rid, v in verdicts.items() if v.decision == "surplus"])

    hits_rows = []
    for rid in result["selected"]:
        for h in verdicts[rid].ner_hits:
            hits_rows.append({"request_id": rid, "genre": plan_by_id[rid]["genre_key"], "detector": h["detector"],
                              "type": h["type"], "surface": h["surface"], "score": h["score"]})
    with open(os.path.join(out_dir, "ner_hits.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "genre", "detector", "type", "surface", "score"])
        w.writeheader()
        for r in hits_rows:
            w.writerow(r)

    # 集計
    reason_counts: Counter = Counter()
    for v in verdicts.values():
        if v.decision in ("reject", "review"):
            for r in v.reasons:
                reason_counts[f"{v.decision}:{r}"] += 1
    decision_counts = Counter(v.decision for v in verdicts.values())
    selected = result["selected"]
    hard_neg = sum(1 for rid in selected if verdicts[rid].ner_hits)
    hit_types = Counter(h["type"] for rid in selected for h in verdicts[rid].ner_hits)
    hit_surfaces = Counter(h["surface"] for rid in selected for h in verdicts[rid].ner_hits if h["detector"] != "handle")
    lens = sorted(len(texts[rid]) for rid in selected)
    q = (lambda p: lens[int(p * (len(lens) - 1))]) if lens else (lambda p: 0)
    by_genre = Counter(plan_by_id[rid]["genre_key"] for rid in selected)
    by_format = Counter(plan_by_id[rid]["format_key"] for rid in selected)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"), "target": target, "selected": len(selected),
        "plan_rows": len(plan_rows), **{k: int(v) for k, v in result["counts"].items()},
        "decisions": dict(decision_counts), "reasons": dict(reason_counts.most_common()),
        "near_duplicate_dropped": result["near_dup_dropped"], "near_duplicate_flagged": result["near_dup_flagged"],
        "shortfall_by_genre": result["shortfall"], "selected_by_genre": dict(by_genre), "selected_by_format": dict(by_format),
        "selected_chars": {"p10": q(.1), "p50": q(.5), "p90": q(.9), "mean": (sum(lens) // len(lens)) if lens else 0},
        "hard_negatives_with_ner_hit": hard_neg, "ner_hit_types": dict(hit_types),
        "ner_hit_top_surfaces": hit_surfaces.most_common(40), "models": list(model_names), "raw_inputs": list(raw_paths),
        "source_name": SOURCE_NAME,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [f"# 負例 v8 の選別レポート（{summary['created_at']}）", "",
             f"- raw 入力: {', '.join(raw_paths)}", f"- 計画 {len(plan_rows)} 件、生成済み {summary.get('generated', 0)} 件、"
             f"未生成 {summary.get('not_generated', 0)} 件", f"- 採用 **{len(selected)}** / 目標 {target}（余剰 {decision_counts.get('surplus', 0)}）",
             f"- reject {decision_counts.get('reject', 0)}、review 除外 {decision_counts.get('review', 0)}"
             f"（`review_excluded.csv` を見て `--include-ids` で戻せる）",
             f"- 近重複で落とした {result['near_dup_dropped']}、要注意 {result['near_dup_flagged']}",
             f"- NER の検出を含む採用文書（ハード負例）: {hard_neg} / {len(selected)}", ""]
    lines += ["## 理由別", "", "| 判定:理由 | 件数 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in reason_counts.most_common()]
    lines += ["", "## 採用のジャンル別", "", "| ジャンル | 採用 | 目標割当 | 不足 |", "|---|---:|---:|---:|"]
    for g in GENRES:
        quota = int(round(target * g.weight))
        lines.append(f"| {g.key} | {by_genre.get(g.key, 0)} | {quota} | {result['shortfall'].get(g.key, 0)} |")
    lines += ["", "## 採用の形式別", "", "| 形式 | 採用 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in by_format.most_common()]
    lines += ["", f"## 文字数（採用） p10 / p50 / p90 = {q(.1)} / {q(.5)} / {q(.9)}", "",
              "## 採用文書に対する NER 検出の型別（この語が PII でないと教える）", "", "| 型 | 件数 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in hit_types.most_common()]
    lines += ["", "上位の検出表層: " + ", ".join(f"{s}×{c}" for s, c in hit_surfaces.most_common(30)), ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    out_dir = args.out or os.path.join(drive_dir(), "negatives_v8")
    rows = build_plan(args.n, args.seed)
    plan_path = os.path.join(out_dir, PLAN_FILE)
    write_jsonl(plan_path, rows)
    manifest = plan_manifest(rows, args.seed, file_sha256(plan_path))
    with open(os.path.join(out_dir, PLAN_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    for stale in ("plan_v1.jsonl", "plan_v1.manifest.json"):
        p = os.path.join(out_dir, stale)
        if os.path.exists(p):
            os.remove(p)
            print("removed stale:", p)
    print(f"plan: {plan_path} ({len(rows)} 件, {PLAN_VERSION}, sha256 {manifest['plan_sha256'][:16]}…)")
    for k in ("genre_counts", "format_counts", "length_counts", "style_counts"):
        print(f"  {k}: {json.dumps(manifest[k], ensure_ascii=False)}")
    print(f"  subgenres: {len(manifest['subgenre_counts'])}, prompt 例:\n")
    print(rows[0]["prompt"])


def cmd_filter(args: argparse.Namespace) -> None:
    drive = drive_dir()
    plan_path = args.plan or os.path.join(drive, "negatives_v8", PLAN_FILE)
    plan_rows = read_jsonl(plan_path)
    raw_rows = load_raw(args.raw)
    model_names = [] if args.no_ner else [m for m in args.models.split(",") if m]
    detectors = load_detectors(model_names, drive) if model_names else None
    handle = None
    if not args.no_ner:
        from privacy_masker.detectors import HandleDetector
        handle = HandleDetector()
    knowledge = None if args.no_ner else load_public_knowledge(drive)
    include_ids = []
    if args.include_ids:
        with open(args.include_ids, encoding="utf-8") as f:
            include_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    out_dir = args.out or os.path.join(drive, "negatives_v8", "filtered_v1")
    result = run_filter(raw_rows, plan_rows, target=args.target, detectors=detectors, handle_detector=handle,
                        knowledge=knowledge, include_ids=include_ids, progress=print)
    summary = write_filter_outputs(result, plan_rows, out_dir, target=args.target, model_names=model_names,
                                   raw_paths=list(args.raw))
    print(json.dumps({k: summary[k] for k in ("selected", "target", "decisions", "near_duplicate_dropped",
                                               "hard_negatives_with_ner_hit", "shortfall_by_genre")}, ensure_ascii=False, indent=2))
    print("出力:", out_dir)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="生成計画をデータフォルダに書く")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="既定: <データフォルダ>/negatives_v8")
    p.set_defaults(func=cmd_plan)
    f = sub.add_parser("filter", help="Colab の raw 出力を選別して学習形式にする")
    f.add_argument("--raw", action="append", required=True, help="Colab の OUT_DIR か raw jsonl（複数可、後勝ち）")
    f.add_argument("--plan", default=None)
    f.add_argument("--target", type=int, default=4000)
    f.add_argument("--models", default="char8ep,v7", help=f"NER の名前かパス。既定 {list(DEFAULT_MODEL_DIRS)}")
    f.add_argument("--no-ner", action="store_true", help="NER・Handle・公知知識を使わず規則だけで選別")
    f.add_argument("--include-ids", default=None, help="review から戻す request_id の一覧ファイル")
    f.add_argument("--out", default=None, help="既定: <データフォルダ>/negatives_v8/filtered_v1")
    f.set_defaults(func=cmd_filter)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

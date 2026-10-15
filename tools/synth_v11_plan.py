"""学習データ v11 の生成計画。文書種 × 書式 × 文体 × 長さ × スロット処方（型・表記・再言及）× ペルソナの網羅行列を
seed 固定で並べ、Qwen への依頼文と JSON スキーマを付けて plan.jsonl にする。値は書かせない（テンプレートだけ）。

usage:
  PYTHONPATH=. .venv/bin/python tools/synth_v11_plan.py plan --n 8000 [--seed 42] [--personas <データフォルダ>/synth_v11/personas_v11.jsonl] [--out-dir <データフォルダ>/synth_v11]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from negatives_v8 import _weighted_choice, file_sha256, read_jsonl, sha256_short, write_jsonl  # noqa: E402,F401
from synth_v11_values import (ADDRESS_FORMS, LOCATION_FORMS, ORG_FORMS, PERSON_SCRIPTS, PERSONAS_FILE,  # noqa: E402
                              SNAPSHOT_DIR_NAME, UNLABELED_TYPES, load_personas)

PLAN_VERSION = "synth_v11_plan_v2"
PLAN_TAG = "plan_v2"
PROMPT_VERSION = "synth_v11_prompt_v2"
SOURCE_NAME = "qwen38_synth_v11"
PLAN_DIR_NAME = SNAPSHOT_DIR_NAME
PLAN_FILE = "plan.jsonl"
PLAN_MANIFEST_FILE = "plan.manifest.json"
RAW_FILE = "qwen_templates_raw.jsonl"

LABELED_TYPES: tuple[str, ...] = ("PERSON", "ADDRESS", "LOCATION", "ORG", "FACILITY")
# _SEI/_MEI/_NICK は PERSON だけ（{ORG1_SEI} は不正）。群は 1=型 2=番号 3=修飾 のまま
SLOT_RE = re.compile(r"\{(?!(?:ADDRESS|LOCATION|ORG|FACILITY|PHONE|EMAIL|URL|POSTAL|DATE|ID)\d+_)"
                     r"(PERSON|ADDRESS|LOCATION|ORG|FACILITY|PHONE|EMAIL|URL|POSTAL|DATE|ID)(\d+)(?:_(SEI|MEI|NICK))?\}")
ANY_BRACE_RE = re.compile(r"\{[^{}]*\}")
COMPOUND_SUFFIXES: tuple[str, ...] = ("大会", "戦", "地震", "支店", "本社", "線", "産", "人", "出身", "代表", "大学", "駅前")
FIRST_MENTION: dict[str, float] = {"full": 0.55, "sei": 0.30, "mei": 0.10, "nick": 0.05}
REMENTION_COUNT: dict[str, float] = {"0": 0.35, "1": 0.40, "2": 0.25}
UNLABELED_WEIGHTS: dict[str, float] = {"PHONE": 0.30, "EMAIL": 0.25, "DATE": 0.20, "URL": 0.10, "POSTAL": 0.10, "ID": 0.05}
ROLE_GENRES: frozenset[str] = frozenset({"business", "forms_logs", "messaging", "medical_care", "legal_admin", "hr_edu"})

LENGTH_KEYS: dict[str, tuple[int, int]] = {"short": (60, 150), "medium": (150, 400), "long": (400, 900)}
LENGTH_WEIGHTS: dict[str, float] = {"short": 0.35, "medium": 0.45, "long": 0.20}
MAX_TOKENS: dict[str, int] = {"short": 320, "medium": 800, "long": 1600}

FORMATS: dict[str, str] = {
    "prose": "段落分けした通常の文章",
    "bullets": "箇条書き（「・」または「-」で始まる行）を中心に構成する。1 項目 1 行",
    "numbered_steps": "番号付きの手順（1. 2. 3. …）。1 手順 1 行",
    "label_colon": "「氏名: {PERSON1}」「住所: {ADDRESS1}」のように項目名の後にスロットを置く行を 4 行以上",
    "table_pipe": "| で区切った簡易表（見出し行 1 行とデータ行 3 行以上）。人名の列にはスロットを置く",
    "chat_lines": "「名前: 発言」の形の発言行を 6 行以上。話者名の位置に {PERSON1_SEI} などのスロットを置いてよい",
    "email_headers": "冒頭に To / From / 件名 の行、本文の段落、末尾に署名の行。宛名や差出人にスロットを置く",
    "log_lines": "「時刻 レベル メッセージ」形式のログ行 5 行以上。operator={PERSON1_SEI} のようにキーの値にスロットを置く行を含める",
    "letter": "宛名の行、本文の段落、差出人の行がある連絡文",
    "mixed": "見出し行、箇条書き、通常の段落を混ぜた構成",
}
STYLES: dict[str, str] = {
    "polite": "敬体（です・ます調）", "plain": "常体（だ・である調）", "spoken": "話し言葉に近い口調", "formal": "硬い事務文体",
    "classic": "明治から昭和初期の小説のような文語寄りの文体",
    "ocr_noise": "OCR で読み取ったような文章。文字間に不規則な空白、誤変換が数か所、行の途切れを少量含める",
    "halfwidth_mix": "半角カナと半角英数字が混ざった帳票の文体",
}
OPENINGS: tuple[str, ...] = ("日付や数値から始める", "結論や要点から始める", "状況や場面の描写から始める", "見出し行から始める",
                             "呼びかけや挨拶から始める", "疑問文から始める", "項目名から始める", "短い一文から始める",
                             "人物のスロットから始める")


@dataclass(frozen=True)
class Subgenre:
    key: str
    label: str
    topics: tuple[str, ...]
    formats: dict[str, float]
    styles: dict[str, float]
    profile: str                 # SLOT_PROFILES のキー
    persona: bool = False        # ペルソナを文脈の種として渡すか
    scripts: dict[str, float] | None = None   # PERSON の表記分布の上書き（None なら PERSON_SCRIPTS）
    extra: str = ""


@dataclass(frozen=True)
class Genre:
    key: str
    label: str
    weight: float
    subgenres: tuple[Subgenre, ...]
    extra: str = ""


# 型ごとの (最小, 最大) 個数。UNLABELED はラベルなしスロットの個数
SLOT_PROFILES: dict[str, dict[str, tuple[int, int]]] = {
    "business": {"PERSON": (1, 3), "ORG": (0, 2), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (0, 2)},
    "form": {"PERSON": (1, 2), "ORG": (0, 1), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (1, 3)},
    "log": {"PERSON": (1, 2), "ORG": (0, 1), "FACILITY": (0, 1), "ADDRESS": (0, 0), "LOCATION": (0, 1), "UNLABELED": (1, 2)},
    "message": {"PERSON": (1, 3), "ORG": (0, 1), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (0, 2)},
    "medical": {"PERSON": (1, 2), "ORG": (0, 1), "FACILITY": (0, 2), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (0, 1)},
    "legal": {"PERSON": (1, 3), "ORG": (0, 2), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 2), "UNLABELED": (0, 1)},
    "news": {"PERSON": (1, 3), "ORG": (0, 2), "FACILITY": (0, 1), "ADDRESS": (0, 0), "LOCATION": (1, 2), "UNLABELED": (0, 1)},
    "personal": {"PERSON": (1, 3), "ORG": (0, 1), "FACILITY": (0, 1), "ADDRESS": (0, 0), "LOCATION": (0, 2), "UNLABELED": (0, 1)},
    "literary": {"PERSON": (1, 3), "ORG": (0, 0), "FACILITY": (0, 1), "ADDRESS": (0, 0), "LOCATION": (0, 2), "UNLABELED": (0, 0)},
    "hr": {"PERSON": (1, 2), "ORG": (0, 2), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (0, 2)},
    "noisy": {"PERSON": (1, 2), "ORG": (0, 1), "FACILITY": (0, 1), "ADDRESS": (0, 1), "LOCATION": (0, 1), "UNLABELED": (0, 2)},
}
ROLES_BY_GENRE: dict[str, tuple[str, ...]] = {
    "business": ("担当者", "顧客", "取引先の担当", "上司", "部下", "出席者", "報告者", "承認者"),
    "forms_logs": ("申込者", "受取人", "記録者", "利用者", "操作者", "確認者", "代表者", "保護者"),
    "messaging": ("送信者", "宛先の人", "同僚", "友人", "家族", "店の担当", "投稿者", "返信した人"),
    "medical_care": ("患者", "担当医", "看護師", "介護職員", "家族", "ケアマネジャー", "薬剤師"),
    "legal_admin": ("原告", "被告", "申立人", "代理人", "参考人", "証人", "担当職員", "通報者"),
    "news_media": ("取材対象", "記者", "関係者", "選手", "監督", "代表", "発表者", "受賞者"),
    "personal_web": ("筆者", "友人", "家族", "店主", "同行者", "講師", "子ども", "隣人"),
    "literary": ("主人公", "語り手の知人", "老人", "娘", "若者", "旅人", "師", "客"),
    "hr_edu": ("応募者", "採用担当", "生徒", "保護者", "教員", "卒業生", "研修講師", "部員"),
    "misc_noisy": ("記載者", "名義人", "担当", "利用者", "受付", "連絡先の人"),
}
_LIT_SCRIPTS = {"kanji": 0.70, "hira": 0.15, "historic": 0.10, "kata_ja": 0.05}
_NOISY_SCRIPTS = {"kanji": 0.45, "halfwidth": 0.30, "kata_ja": 0.15, "romaji": 0.10}
_INTL_SCRIPTS = {"kanji": 0.35, "kata_foreign": 0.35, "romaji": 0.30}
_B_FMT = {"prose": 0.25, "bullets": 0.20, "label_colon": 0.25, "table_pipe": 0.10, "numbered_steps": 0.10, "letter": 0.10}
_B_STY = {"polite": 0.5, "formal": 0.4, "plain": 0.1}
_P_STY = {"polite": 0.45, "plain": 0.35, "spoken": 0.20}


def _sub(key, label, topics, formats, styles, profile, persona=False, scripts=None, extra=""):
    return Subgenre(key, label, tuple(topics), formats, styles, profile, persona, scripts, extra)


GENRES: tuple[Genre, ...] = (
    Genre("business", "業務文書", 0.16, (
        _sub("meeting_minutes", "議事録", ["月次の進捗確認", "予算見直し", "新規取引の可否", "納期遅れの対応", "人事異動の共有", "安全点検の結果", "システム更改", "顧客訪問の報告"], _B_FMT, _B_STY, "business"),
        _sub("daily_report", "日報・業務報告", ["設備点検", "来客対応", "検品結果", "配送遅延", "売上速報", "出張報告", "クレーム対応の経過", "研修受講"], _B_FMT, _B_STY, "business"),
        _sub("procedure", "手順書・引き継ぎ", ["承認の流れ", "問い合わせのエスカレーション", "備品貸出", "鍵の管理", "在庫棚卸し", "新人の初日案内"], {"numbered_steps": 0.5, "bullets": 0.3, "prose": 0.2}, _B_STY, "business"),
        _sub("approval", "稟議・申請", ["備品購入", "出張申請", "外注契約", "研修参加", "システム導入", "採用計画"], {"label_colon": 0.5, "prose": 0.3, "letter": 0.2}, {"formal": 0.7, "polite": 0.3}, "business"),
        _sub("internal_notice", "社内通知", ["座席変更", "健康診断", "システム停止", "訪問者の受け入れ", "防災訓練", "年末調整"], {"letter": 0.4, "bullets": 0.3, "prose": 0.3}, _B_STY, "business"),
        _sub("cs_record", "顧客対応記録", ["返品希望", "配送先変更", "料金の問い合わせ", "不具合の申告", "解約手続き", "予約変更"], {"label_colon": 0.4, "prose": 0.3, "chat_lines": 0.3}, _B_STY, "business", persona=True),
        _sub("quote_invoice", "見積・請求", ["工事見積", "翻訳料の請求", "保守契約の更新", "納品書", "領収書の再発行", "支払い催促"], {"label_colon": 0.5, "table_pipe": 0.3, "letter": 0.2}, {"formal": 0.6, "polite": 0.4}, "business"),
        _sub("intl_business", "英日混在の業務連絡", ["海外拠点との調整", "外国人講師の招聘", "海外顧客からの問い合わせ", "留学生の受け入れ", "輸入手続き"], {"email_headers": 0.4, "prose": 0.4, "bullets": 0.2}, _B_STY, "business", scripts=_INTL_SCRIPTS),
    )),
    Genre("forms_logs", "帳票・ログ", 0.14, (
        _sub("application_form", "申込書・登録", ["会員登録", "講座申込", "口座開設", "保険の加入", "サービス契約", "駐車場利用"], {"label_colon": 0.7, "table_pipe": 0.2, "mixed": 0.1}, {"formal": 0.6, "polite": 0.4}, "form", persona=True),
        _sub("delivery_slip", "配送伝票・予約確認", ["宅配の伝票", "宿泊予約", "レンタカー予約", "レストラン予約", "修理受付票", "クリーニング預かり票"], {"label_colon": 0.7, "table_pipe": 0.3}, {"formal": 0.7, "polite": 0.3}, "form", persona=True),
        _sub("roster", "名簿・出席表・当番表", ["会議の出席者", "部活の名簿", "夜勤の当番表", "研修の受講者", "同窓会の参加者", "町内会の役員"], {"table_pipe": 0.5, "bullets": 0.3, "label_colon": 0.2}, {"formal": 0.7, "plain": 0.3}, "form"),
        _sub("system_log", "システムログ", ["ログイン履歴", "ジョブの実行", "承認フロー", "ファイル出力", "設定変更", "アラート"], {"log_lines": 0.8, "table_pipe": 0.2}, {"plain": 0.8, "formal": 0.2}, "log"),
        _sub("audit_log", "監査・受付記録", ["入退室記録", "鍵の貸出", "来訪者の受付", "電話の受付簿", "苦情の受付", "遺失物の届け"], {"label_colon": 0.5, "log_lines": 0.3, "table_pipe": 0.2}, {"formal": 0.7, "plain": 0.3}, "form"),
        _sub("care_form", "介護・保育の連絡帳", ["日中の様子", "食事と排泄の記録", "服薬の確認", "送迎の連絡", "体調の申し送り"], {"label_colon": 0.5, "prose": 0.3, "bullets": 0.2}, {"polite": 0.7, "plain": 0.3}, "form", persona=True),
    )),
    Genre("messaging", "通信", 0.14, (
        _sub("business_email", "業務メール", ["日程調整", "資料送付", "見積の依頼", "不具合の報告", "面談のお礼", "採用連絡"], {"email_headers": 0.7, "letter": 0.3}, {"polite": 0.8, "formal": 0.2}, "message"),
        _sub("private_email", "私用メール", ["旅行の相談", "同窓会の案内", "引っ越しの報告", "結婚の報告", "お礼", "近況"], {"email_headers": 0.5, "letter": 0.5}, {"polite": 0.5, "spoken": 0.5}, "message", persona=True),
        _sub("work_chat", "業務チャット", ["障害対応", "レビュー依頼", "当番の交代", "会議のリマインド", "資料の場所", "客先の連絡"], {"chat_lines": 0.9, "prose": 0.1}, {"spoken": 0.6, "plain": 0.4}, "message"),
        _sub("private_chat", "私用チャット", ["飲み会の調整", "誕生日の相談", "推しの話", "子どもの送迎", "旅行の写真", "ゲームの約束"], {"chat_lines": 0.9, "prose": 0.1}, {"spoken": 0.9, "plain": 0.1}, "message", persona=True),
        _sub("sns_post", "SNS 投稿・掲示板", ["イベントの感想", "お店の紹介", "落とし物の呼びかけ", "地域の告知", "質問への回答", "推薦"], {"prose": 0.7, "bullets": 0.3}, {"spoken": 0.7, "plain": 0.3}, "message"),
        _sub("transcript", "音声書き起こし", ["電話の応対", "会議の一部", "インタビュー", "窓口のやりとり", "講演の質疑"], {"chat_lines": 0.8, "prose": 0.2}, {"spoken": 0.9, "polite": 0.1}, "message"),
    )),
    Genre("medical_care", "医療・介護", 0.08, (
        _sub("clinical_note", "診療記録", ["初診の主訴", "術後の経過", "検査結果の説明", "紹介状", "退院時のまとめ"], {"label_colon": 0.4, "prose": 0.4, "bullets": 0.2}, {"formal": 0.6, "plain": 0.4}, "medical", persona=True),
        _sub("nursing_note", "看護・介護記録", ["夜間の巡回", "転倒の報告", "食事量", "リハビリの進捗", "家族への説明"], {"label_colon": 0.5, "prose": 0.3, "log_lines": 0.2}, {"polite": 0.5, "plain": 0.5}, "medical", persona=True),
        _sub("prescription_contact", "処方・検査の連絡", ["薬局からの疑義照会", "検査予約", "健診結果の通知", "予防接種の案内"], {"letter": 0.5, "label_colon": 0.5}, {"polite": 0.8, "formal": 0.2}, "medical", persona=True),
        _sub("facility_news", "施設だより", ["季節の行事", "面会のお願い", "感染対策", "職員の紹介", "献立の紹介"], {"prose": 0.6, "bullets": 0.4}, {"polite": 0.9, "spoken": 0.1}, "medical"),
    )),
    Genre("legal_admin", "法務・行政", 0.08, (
        _sub("contract", "契約書", ["業務委託", "賃貸", "売買", "秘密保持", "雇用"], {"numbered_steps": 0.5, "prose": 0.3, "label_colon": 0.2}, {"formal": 1.0}, "legal"),
        _sub("judgment", "判決・審決", ["損害賠償", "建物明渡し", "労働審判", "行政処分の取消", "遺産分割"], {"prose": 0.8, "numbered_steps": 0.2}, {"formal": 1.0}, "legal", extra="当事者は氏名のスロットで書き、肩書き（原告・被告・代表取締役）はスロットの外に置く"),
        _sub("gov_notice", "行政通知・条例", ["給付金の案内", "ごみ収集の変更", "選挙の公示", "工事のお知らせ", "補助金の募集"], {"letter": 0.5, "bullets": 0.3, "prose": 0.2}, {"formal": 0.7, "polite": 0.3}, "legal"),
        _sub("complaint_report", "苦情処理・事故報告", ["騒音の苦情", "接触事故", "落下物", "誤配送", "設備の故障"], {"label_colon": 0.5, "prose": 0.5}, {"formal": 0.6, "plain": 0.4}, "legal", persona=True),
    )),
    Genre("news_media", "報道", 0.10, (
        _sub("news_article", "ニュース", ["地域の催し", "企業の発表", "事故", "選挙", "受賞", "気象", "新店開業"], {"prose": 1.0}, {"plain": 0.8, "formal": 0.2}, "news"),
        _sub("press_release", "プレスリリース", ["新製品", "提携", "人事", "拠点開設", "受賞", "イベント開催"], {"prose": 0.6, "bullets": 0.4}, {"formal": 0.7, "polite": 0.3}, "news"),
        _sub("interview", "インタビュー", ["起業の経緯", "研究の成果", "職人の技", "選手の抱負", "移住の暮らし"], {"chat_lines": 0.5, "prose": 0.5}, {"polite": 0.6, "spoken": 0.4}, "news", persona=True),
        _sub("sports", "スポーツ記事", ["試合結果", "移籍", "大会展望", "記録達成", "監督交代"], {"prose": 0.9, "bullets": 0.1}, {"plain": 0.9, "formal": 0.1}, "news", extra="大会名は「{LOCATION1}大会」のように地名スロットの直後に語を続ける形を使ってよい"),
        _sub("local_info", "地域情報", ["祭りの案内", "商店街", "交通規制", "新しい公園", "郷土料理"], {"prose": 0.6, "bullets": 0.4}, {"polite": 0.6, "plain": 0.4}, "news"),
    )),
    Genre("personal_web", "個人の Web", 0.12, (
        _sub("blog", "ブログ", ["休日の過ごし方", "資格の勉強", "引っ越し", "在宅勤務", "家庭菜園", "推しのライブ"], {"prose": 0.8, "bullets": 0.2}, _P_STY, "personal", persona=True),
        _sub("review", "EC・店のレビュー", ["家電", "宿", "飲食店", "美容院", "習い事", "書籍"], {"prose": 0.7, "bullets": 0.3}, {"spoken": 0.5, "polite": 0.3, "plain": 0.2}, "personal", persona=True),
        _sub("travel_diary", "旅行記", ["温泉", "海外旅行", "登山", "ドライブ", "食べ歩き"], {"prose": 0.8, "bullets": 0.2}, _P_STY, "personal", persona=True, scripts={"kanji": 0.55, "kata_foreign": 0.25, "romaji": 0.10, "kata_ja": 0.10}),
        _sub("hobby_cooking", "料理・趣味", ["レシピ", "手芸", "釣り", "カメラ", "ランニング"], {"prose": 0.5, "numbered_steps": 0.3, "bullets": 0.2}, _P_STY, "personal", persona=True),
        _sub("parenting", "子育て", ["入園準備", "習い事", "発熱", "運動会", "反抗期"], {"prose": 0.8, "bullets": 0.2}, {"spoken": 0.5, "polite": 0.5}, "personal", persona=True),
        _sub("qa_forum", "Q&A 掲示板", ["引っ越し先の相談", "保険の選び方", "資格の勉強法", "近所のトラブル", "転職"], {"chat_lines": 0.5, "prose": 0.5}, {"spoken": 0.6, "polite": 0.4}, "personal"),
    )),
    Genre("literary", "文語・創作", 0.08, (
        _sub("modern_novel", "近代文体の小説・随筆", ["下宿の日々", "故郷への手紙", "師との別れ", "祭りの夜", "病床", "汽車の旅"], {"prose": 1.0}, {"classic": 0.8, "plain": 0.2}, "literary", scripts=_LIT_SCRIPTS, extra="登場人物は名だけ・あだ名・「お〜」の付く呼び名で呼ばれることが多い。そうした短い呼び名もスロットで書く（{PERSON1_MEI} や {PERSON1_NICK}）"),
        _sub("folktale_biography", "昔話・伝記", ["武将の逸話", "文豪の少年時代", "村の言い伝え", "商人の成功譚", "名医の物語"], {"prose": 1.0}, {"classic": 0.5, "plain": 0.5}, "literary", scripts={"historic": 0.45, "kanji": 0.40, "hira": 0.15}),
        _sub("script", "脚本・戯曲", ["家族の食卓", "職場の口論", "駅での再会", "面接", "祝いの席"], {"chat_lines": 0.9, "prose": 0.1}, {"spoken": 0.7, "classic": 0.3}, "literary", scripts=_LIT_SCRIPTS),
        _sub("verse", "詩・歌詞風", ["別れ", "故郷", "春", "友", "旅"], {"prose": 1.0}, {"classic": 0.6, "plain": 0.4}, "literary", scripts=_LIT_SCRIPTS),
    )),
    Genre("hr_edu", "人事・教育", 0.06, (
        _sub("job_posting", "求人", ["事務職", "介護職", "エンジニア", "店舗スタッフ", "講師"], {"label_colon": 0.5, "bullets": 0.3, "prose": 0.2}, {"polite": 0.6, "formal": 0.4}, "hr"),
        _sub("resume", "履歴書・職務経歴", ["転職", "新卒", "再就職", "アルバイト", "海外経験"], {"label_colon": 0.6, "table_pipe": 0.2, "prose": 0.2}, {"formal": 0.8, "polite": 0.2}, "hr", persona=True, scripts={"kanji": 0.60, "kata_ja": 0.20, "romaji": 0.20}),
        _sub("school_news", "学校だより", ["運動会", "進路説明会", "PTA", "部活の成績", "転入生の紹介"], {"prose": 0.6, "bullets": 0.4}, {"polite": 0.9, "plain": 0.1}, "hr"),
        _sub("grades_attendance", "成績・出席", ["期末の所見", "欠席連絡", "遅刻の記録", "補習の案内"], {"label_colon": 0.5, "table_pipe": 0.3, "prose": 0.2}, {"formal": 0.6, "polite": 0.4}, "hr"),
        _sub("training_alumni", "研修案内・同窓会", ["新任研修", "安全講習", "同窓会の案内", "OB 訪問", "部の合宿"], {"letter": 0.5, "bullets": 0.3, "prose": 0.2}, {"polite": 0.8, "plain": 0.2}, "hr"),
    )),
    Genre("misc_noisy", "雑多・ノイズ", 0.04, (
        _sub("ocr_document", "OCR 風の文書", ["申請書のスキャン", "領収書", "名刺", "古い議事録", "回覧板"], {"label_colon": 0.5, "prose": 0.3, "mixed": 0.2}, {"ocr_noise": 1.0}, "noisy", scripts=_NOISY_SCRIPTS),
        _sub("csv_table", "表・CSV 風", ["顧客一覧", "配送先一覧", "参加者一覧", "在庫と担当", "施設と管理者"], {"table_pipe": 1.0}, {"plain": 1.0}, "noisy", scripts={"kanji": 0.6, "kata_ja": 0.2, "halfwidth": 0.2}),
        _sub("halfwidth_form", "半角カナ帳票", ["振込明細", "会員情報", "配送ラベル", "レシート"], {"label_colon": 0.7, "table_pipe": 0.3}, {"halfwidth_mix": 1.0}, "noisy", scripts={"halfwidth": 0.6, "kanji": 0.3, "kata_ja": 0.1}),
        _sub("vertical_transcript", "縦書き・掲示の転写", ["張り紙", "掲示板の告知", "表彰状", "看板"], {"prose": 0.6, "bullets": 0.4}, {"formal": 0.5, "plain": 0.5}, "noisy"),
    )),
)
GENRE_BY_KEY: dict[str, Genre] = {g.key: g for g in GENRES}
assert abs(sum(g.weight for g in GENRES) - 1.0) < 1e-9

SYSTEM_PROMPT = """あなたは日本語の文書テンプレートを書く専門家です。指定された種類・題材・書式・文体・長さで、実務や生活の場面にありそうな文書を 1 件書きます。
ただし人名・住所・地名・組織名・施設名・電話番号・メールアドレス・URL・郵便番号・日付・番号などの値は書かず、指示されたスロットをそのまま置きます。

規則:
1. スロットは {PERSON1} {PERSON1_SEI} {PERSON1_MEI} {PERSON1_NICK} {ADDRESS1} {LOCATION1} {ORG1} {FACILITY1} {PHONE1} {EMAIL1} {URL1} {POSTAL1} {DATE1} {ID1} のように、波括弧・大文字の型・番号だけで書く。波括弧の中に助詞や敬称を入れない（「{PERSON1さん}」は誤り。「{PERSON1}さん」が正しい）。
2. 指示された「必須」のスロットは必ず使う。「任意」のスロットは自然でなければ省いてよい。指示にない値（日付・住所・電話・メール・URL・郵便番号・番号、別の人物・組織・地名・施設）が必要になったら、同じ型で番号を増やした新しいスロットを作る（例: {DATE1} {ADDRESS1} {PERSON3}）。上に無い型のスロットは作らない。
3. 同じ番号は同じ人物・同じ場所・同じ組織。{PERSON1_SEI} は {PERSON1} と同じ人物の姓だけ、{PERSON1_MEI} は名だけ、{PERSON1_NICK} はあだ名。人物の初出は修飾なしの {PERSON1} で書き、{PERSON1_SEI} などはその後に使う。
4. 敬称・肩書き・役職（さん、様、氏、先生、部長、代表取締役）はスロットの外に書く。
5. スロットの外には、実在・架空を問わず人名・地名・組織名・施設名・商品名を書かない。都道府県名（愛知県・北海道 など）・市区町村名・国名（日本 を含む）・部署名（総務部・営業部・人事課 など）・官公庁名（気象庁 など）も同じ。住所・電話番号・メールアドレス・URL・郵便番号・口座や会員の番号は、例示用の値（03-1234-5678、example.com、〒150-0001 など）を含めて書かない。それらが必要な場所には必ずスロットを置く。日付は書いてもスロット（{DATE1}）にしてもよい。金額は書いてよい。
6. スロット以外の文章は自然で具体的に書く。人物の呼び方（姓だけ、名だけ、あだ名、敬称の有無）は指示に従い、文書の種類に合った書き方にする。
7. 日本語で書く。ASCII 英数字と通常の記号は使ってよいが、簡体字・繁体字・ハングル・キリル文字など他の文字体系は使わない。
8. 出力は {"lines": ["1行目", "2行目", ...]} の形の JSON だけ。本文を行ごとに分け、配列の 1 要素を 1 行にする。段落の区切りは "" の要素で表す。説明・前置き・Markdown の囲みや見出し記号（#）は書かない。
例: {"lines": ["{PERSON1}様", "", "お世話になっております。{ORG1}の{PERSON2}です。", "先日の件、{PERSON2_SEI}までご連絡ください。"]}
"""

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {"lines": {"type": "array", "minItems": 1, "maxItems": 120,
                             "items": {"type": "string", "maxLength": 600}}},
    "required": ["lines"], "additionalProperties": False,
}

_SCRIPT_LABEL = {"kanji": "漢字の日本人名", "kata_ja": "カタカナ表記の日本人名", "kata_foreign": "カタカナ表記の外国人名",
                 "hira": "ひらがな表記の名前", "romaji": "ローマ字表記の名前", "halfwidth": "半角カナ表記の名前",
                 "historic": "歴史上の人物の名前"}
_FIRST_LABEL = {"full": "フルネーム", "sei": "姓だけ", "mei": "名だけ", "nick": "あだ名"}
_MOD_LABEL = {"SEI": "姓だけ", "MEI": "名だけ", "NICK": "あだ名"}
_LOC_LABEL = {"prefecture": "都道府県名", "city": "市区町村名", "town": "町名・地区名", "country": "国名",
              "foreign_kata": "カタカナの外国の都市名", "compound": "地名", "station": "駅名"}
_ORG_LABEL = {"corporate": "法人格付きの会社名", "kata_brand": "カタカナの屋号・ブランド名・チーム名（法人格なし）",
              "abbr_latin": "英字略称の組織名", "dept": "部署名・課名", "long_compound": "長い正式名称の団体名",
              "quoted_brand": "店名・ブランド名。「{slot}」のように「」で囲んで書く"}
_ADDR_LABEL = {"pref_first": "都道府県から始まる住所", "city_first": "市区町村から始まる住所",
               "banchi_first": "番地から始まる短い住所（建物名・部屋番号付き）", "with_building": "建物名と部屋番号まで含む住所",
               "fullwidth": "全角数字の住所"}
_UNLABELED_LABEL = {"PHONE": "電話番号", "EMAIL": "メールアドレス", "URL": "URL", "POSTAL": "郵便番号", "DATE": "日付", "ID": "受付番号・注文番号など"}


def describe_slot(spec: dict) -> str:
    t, slot = spec["type"], spec["slot"]
    req = "必須" if spec["required"] else "任意"
    if t == "PERSON":
        s = f"- {slot}（{req}）: {spec['role']}。{_SCRIPT_LABEL[spec['script']]}。初出の {slot} には{_FIRST_LABEL[spec['first']]}が入る"
        if spec["rementions"]:
            parts = [f"{{{t}{spec['num']}_{m}}}（{_MOD_LABEL[m]}）" for m in spec["rementions"]]
            s += f"。その後、{'、'.join(parts)} で再び言及する"
        return s
    if t == "LOCATION":
        s = f"- {slot}（{req}）: {_LOC_LABEL[spec['form']]}"
        if spec.get("compound_suffix"):
            s += f"。「{slot}{spec['compound_suffix']}」のように直後に「{spec['compound_suffix']}」を続けて書く"
        return s
    if t == "ORG":
        return f"- {slot}（{req}）: {_ORG_LABEL[spec['form']].replace('{slot}', slot)}"
    if t == "ADDRESS":
        return f"- {slot}（{req}）: {_ADDR_LABEL[spec['form']]}"
    if t == "FACILITY":
        return f"- {slot}（{req}）: 施設名（病院・学校・店舗・ビルなど）"
    return f"- {slot}（{req}）: {_UNLABELED_LABEL[t]}。値は書かず {slot} のまま置く"


def build_slot_specs(rng: random.Random, sub: Subgenre, genre_key: str, persona: dict | None) -> list[dict]:
    prof = SLOT_PROFILES[sub.profile]
    scripts = sub.scripts or PERSON_SCRIPTS
    roles = list(ROLES_BY_GENRE[genre_key])
    rng.shuffle(roles)
    specs: list[dict] = []
    n_person = rng.randint(*prof["PERSON"])
    for i in range(1, n_person + 1):
        bound = persona is not None and i == 1
        script = "kanji" if bound else _weighted_choice(rng, scripts)
        first = _weighted_choice(rng, FIRST_MENTION)
        if script == "historic":
            first = "full"
        k = int(_weighted_choice(rng, REMENTION_COUNT))
        cand = ["SEI", "SEI", "MEI", "NICK"] if first != "sei" else ["MEI", "NICK", "SEI"]
        if script in ("historic", "romaji", "halfwidth"):
            cand = ["SEI", "MEI"]
        rementions = [rng.choice(cand) for _ in range(k)]
        specs.append({"slot": f"PERSON{i}", "type": "PERSON", "num": i, "required": i == 1 or rng.random() < 0.7,
                      "role": roles[(i - 1) % len(roles)], "script": script, "first": first, "rementions": rementions,
                      "form": None, "compound_suffix": None, "persona_bound": bound})
    for t, forms in (("ORG", ORG_FORMS), ("FACILITY", None), ("ADDRESS", ADDRESS_FORMS), ("LOCATION", LOCATION_FORMS)):
        lo, hi = prof[t]
        for i in range(1, rng.randint(lo, hi) + 1):
            form = _weighted_choice(rng, forms) if forms else None
            suffix = rng.choice(COMPOUND_SUFFIXES) if (t == "LOCATION" and form == "compound") else None
            specs.append({"slot": f"{t}{i}", "type": t, "num": i, "required": rng.random() < 0.7, "role": None, "script": None,
                          "first": None, "rementions": [], "form": form, "compound_suffix": suffix,
                          "persona_bound": persona is not None and t == "ADDRESS" and i == 1})
    lo, hi = prof["UNLABELED"]
    counts: Counter = Counter()
    for _ in range(rng.randint(lo, hi)):
        t = _weighted_choice(rng, UNLABELED_WEIGHTS)
        counts[t] += 1
        specs.append({"slot": f"{t}{counts[t]}", "type": t, "num": counts[t], "required": rng.random() < 0.5, "role": None,
                      "script": None, "first": None, "rementions": [], "form": None, "compound_suffix": None, "persona_bound": False})
    return specs


def build_user_prompt(row: dict) -> str:
    lo, hi = row["length_min"], row["length_max"]
    lines = ["次の条件で日本語の文書テンプレートを 1 件書いてください。",
             f"- 種類: {row['genre_label']} / {row['subgenre_label']}", f"- 題材: {row['topic']}",
             f"- 書式: {FORMATS[row['format_key']]}", f"- 文体: {STYLES[row['style']]}",
             f"- 長さ: スロットを除いて {lo}〜{hi} 文字", f"- 書き出し: {row['opening']}"]
    extras = [e for e in (row.get("genre_extra", ""), row.get("subgenre_extra", "")) if e]
    if extras:
        lines.append("- 追加の指示: " + " ".join(extras))
    if row.get("persona"):
        p = row["persona"]
        lines.append(f"- 中心となる人物 {{PERSON1}} の設定（名前は書かない）: {p.get('age', '')}歳前後、{p.get('sex', '')}性、"
                     f"職業は{p.get('occupation', '')}、{p.get('prefecture', '')}に住み、趣味は{p.get('hobby', '')}。この設定に合う内容にする")
    lines.append("- 使うスロット:")
    lines.extend(describe_slot(s) for s in row["slots"])
    lines.append("禁止事項の再確認: スロットの外に人名・地名（都道府県・市区町村・国名）・組織名・部署名・施設名・住所・電話・メール・URL・番号を書かない。"
                 "例示用の値も書かない。必要なら番号を増やした新しいスロットを作る。波括弧の中は型と番号（と _SEI/_MEI/_NICK）だけ。敬称・肩書きはスロットの外。")
    lines.append('出力は {"lines": ["1行目", "2行目", ...]} の JSON だけ。1 行を配列の 1 要素にし、段落の区切りは "" の要素にする。')
    return "\n".join(lines)


def build_plan(n: int, seed: int = 42, personas: Sequence[dict] | None = None, max_repeat: int = 3) -> list[dict]:
    """ジャンルの重み付きラウンドロビン（先頭 k 件でも比率が保たれる）。サブジャンルは巡回、題材・書式・文体・長さは重み付き乱択。"""
    rng = random.Random(seed)
    personas = list(personas or [])
    credit = {g.key: 0.0 for g in GENRES}
    cursor = {g.key: 0 for g in GENRES}
    combo_count: Counter = Counter()
    rows: list[dict] = []
    for i in range(n):
        for g in GENRES:
            credit[g.key] += g.weight
        gkey = max(credit, key=lambda k: (credit[k], k))
        credit[gkey] -= 1.0
        g = GENRE_BY_KEY[gkey]
        sub = g.subgenres[cursor[gkey] % len(g.subgenres)]
        cursor[gkey] += 1
        for _ in range(20):
            topic = rng.choice(sub.topics)
            fmt = _weighted_choice(rng, sub.formats)
            style = _weighted_choice(rng, sub.styles)
            length_key = _weighted_choice(rng, LENGTH_WEIGHTS)
            combo = (sub.key, topic, fmt, style)
            if combo_count[combo] < max_repeat:
                break
        combo_count[combo] += 1
        persona = rng.choice(personas) if (sub.persona and personas) else None
        specs = build_slot_specs(rng, sub, g.key, persona)
        lo, hi = LENGTH_KEYS[length_key]
        row = {"request_id": f"syn_v11_{i:05d}", "index": i,
               "genre_key": g.key, "genre_label": g.label, "genre_extra": g.extra,
               "subgenre_key": sub.key, "subgenre_label": sub.label, "subgenre_extra": sub.extra,
               "topic": topic, "format_key": fmt, "style": style,
               "length_key": length_key, "length_min": lo, "length_max": hi, "max_tokens": MAX_TOKENS[length_key],
               "opening": rng.choice(OPENINGS), "slots": specs,
               "persona": ({k: persona.get(k) for k in ("uuid", "age", "sex", "occupation", "prefecture", "hobby")} if persona else None),
               "role_ok": g.key in ROLE_GENRES,
               "plan_version": PLAN_VERSION, "prompt_version": PROMPT_VERSION}
        row["prompt"] = build_user_prompt(row)
        row["prompt_sha256"] = sha256_short(SYSTEM_PROMPT + "\n" + row["prompt"])
        rows.append(row)
    return rows


def plan_manifest(rows: Sequence[dict], seed: int, plan_sha256: str, personas_sha256: str | None = None) -> dict:
    persons = [s for r in rows for s in r["slots"] if s["type"] == "PERSON"]
    return {
        "plan_version": PLAN_VERSION, "plan_tag": PLAN_TAG, "prompt_version": PROMPT_VERSION, "seed": seed, "n": len(rows),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"), "generator": "tools/synth_v11_plan.py plan",
        "plan_file": PLAN_FILE, "plan_sha256": plan_sha256, "personas_sha256": personas_sha256,
        "system_prompt": SYSTEM_PROMPT, "output_schema": OUTPUT_SCHEMA,
        "genre_weights": {g.key: g.weight for g in GENRES},
        "genre_counts": dict(Counter(r["genre_key"] for r in rows)),
        "subgenre_counts": dict(Counter(r["subgenre_key"] for r in rows)),
        "format_counts": dict(Counter(r["format_key"] for r in rows)),
        "style_counts": dict(Counter(r["style"] for r in rows)),
        "length_counts": dict(Counter(r["length_key"] for r in rows)),
        "persona_rows": sum(1 for r in rows if r["persona"]),
        "person_slots": len(persons),
        "person_script_counts": dict(Counter(s["script"] for s in persons)),
        "person_first_counts": dict(Counter(s["first"] for s in persons)),
        "location_form_counts": dict(Counter(s["form"] for r in rows for s in r["slots"] if s["type"] == "LOCATION")),
        "org_form_counts": dict(Counter(s["form"] for r in rows for s in r["slots"] if s["type"] == "ORG")),
        "address_form_counts": dict(Counter(s["form"] for r in rows for s in r["slots"] if s["type"] == "ADDRESS")),
        "unlabeled_rows": sum(1 for r in rows if any(s["type"] in UNLABELED_TYPES for s in r["slots"])),
        "length_ranges": LENGTH_KEYS, "max_tokens": MAX_TOKENS,
        "sampling_recommended": {"temperature": 0.85, "top_p": 0.9, "top_k": 40, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--n", type=int, default=8000); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--personas", default=None); p.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.join(drive_dir(), PLAN_DIR_NAME)
    personas_path = args.personas or os.path.join(out_dir, PERSONAS_FILE)
    personas = load_personas(personas_path) if os.path.exists(personas_path) else []
    if not personas:
        print(f"[warn] ペルソナスナップショットがありません: {personas_path}（ペルソナ無しで計画を作ります）")
    rows = build_plan(args.n, seed=args.seed, personas=personas)
    plan_path = os.path.join(out_dir, PLAN_FILE)
    write_jsonl(plan_path, rows)
    m = plan_manifest(rows, args.seed, file_sha256(plan_path), file_sha256(personas_path) if personas else None)
    with open(os.path.join(out_dir, PLAN_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: m[k] for k in ("n", "genre_counts", "person_script_counts", "person_first_counts", "location_form_counts", "persona_rows", "unlabeled_rows")}, ensure_ascii=False, indent=2))
    print("wrote", plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

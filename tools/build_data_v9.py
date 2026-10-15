"""学習データ v9 — v8 の切り分け（開発記録、非公開）を受けた調整。

v8 で分かったこと:
- LLM 負例は負例文書への過剰マスクを 5〜10 分の 1 にし OOD の一般語の誤検出を 4 割減らすが、17% では
  OpenPII の裸の町名（ADDRESS 6 件）と GSD のカタカナ著名人（2 件）を落とす
- 英数字（メールの一部）を PERSON に寄せる誤検出は負例のせいではなく、帳票フレームの `TO: {k}` `cc: {k}` `user={k}` が原因。
  負例がメール・URL を禁じていたので打ち消す例が無かった
- v8 のデータ構成自体（上限 6・帳票・新規半減）は v7 より再現率を大きく上げた（固定OOD 見逃し 24 → 10）

v9 の変更:

| 変更 | v8 | v9 |
|---|---|---|
| テンプレート上限 | 6 | **5**（wiki の比重をさらに上げる） |
| LLM 負例 | 4,000（17%） | **2,000（約 10%）**。NER が何かを検出した「ハード負例」423 件を優先し、残りはジャンル比で抽出 |
| 帳票フレーム | `TO: {k}` `cc: {k}` あり | **`TO:` `cc:` の姓フレームを外し、メールアドレス・URL を値にした行を非エンティティで追加**。`From: "{k}" <{email}>` で氏名だけ PERSON |
| カタカナ人名の文書 | なし | **600 文書**（外国名 姓・名・姓＋名、日本人のカタカナ名。報道・紹介文のフレーム） |
| 裸の町名の文書 | なし | **500 文書**（`{町}では` `{町}から{町}まで` を LOCATION。番地なし） |

4 評価セットの gold に出る PERSON 表層は姓・カタカナ名の候補から、LOCATION / ADDRESS 表層は町名の候補から機械的に除く。

使い方::

    PYTHONPATH=. .venv/bin/python tools/build_data_v9.py --out-dir <データフォルダ>/data_v9 [--cap 5] [--seed 42]
        [--chat 750] [--dialogue 750] [--halfwidth 250] [--forms 1100] [--katakana 600] [--towns 500]
        [--negatives <データフォルダ>/negatives_v8/filtered_v1/negatives_v8.jsonl] [--n-negatives 2000]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from augment_address_formats import augment as augment_addresses  # noqa: E402
from build_data_v6 import (  # noqa: E402
    LONG_TARGET_RATIO,
    PERSON_VARIANT_RATIO,
    _digest,
    _harvest_name_parts,
    augment_persons,
    cap_templates,
    data_card,
    make_long_docs,
    relabel_wiki_addresses,
    stats,
    write_jsonl,
)
from build_data_v7 import (  # noqa: E402
    BUSINESS_LINES,
    CHAT_UTTERANCES,
    EVAL_SURFACES,
    GIVEN_KATA_JA,
    GIVEN_KATA_SHORT,
    NARRATIVE_FILLERS,
    NEW_DOC_VALID_RATIO,
    SURNAMES_KATA_JA,
    VALID_BANK_RATIO,
    _Doc,
    _split_bank,
    check_no_eval_surfaces,
    make_chat_docs,
    make_dialogue_docs,
    make_halfwidth_docs,
)
from build_data_v8 import (  # noqa: E402
    FORM_FILLERS,
    FORM_FRAMES,
    FORM_NAME_SLOT_RATIO,
    FORM_PURE_NEGATIVE_RATIO,
    FORM_SOURCE,
    LLM_NEGATIVE_SOURCE,
    LOG_FILLERS,
    LOG_FRAMES,
    NEG_FORM_VALUES,
    NEG_LOG_VALUES,
    SHIFTS,
    _fill_values,
    _name_surface,
    check_no_eval_person_surfaces,
    eval_person_exclusions,
    load_negatives,
    surname_bank,
    without_llm_negatives,
)
from split_by_template import (  # noqa: E402
    BASE_LABELS,
    DATA_FILES,
    grouped_split,
    load_records,
    template_signature,
)
from privacy_masker.data.ja_pools import GIVEN_KANJI, GIVEN_KANJI_F, GIVEN_KATA, SURNAMES_KATA, TOWNS  # noqa: E402

SEED = 42
DEFAULT_CAP = 5
DEFAULT_CHAT = 750
DEFAULT_DIALOGUE = 750
DEFAULT_HALFWIDTH = 250
DEFAULT_FORMS = 1100
DEFAULT_KATAKANA = 600
DEFAULT_TOWNS = 500
DEFAULT_N_NEGATIVES = 2000
NEG_VALID_RATIO = 0.10
KATAKANA_SOURCE = "synth_katakana"
TOWN_SOURCE = "synth_town"
NEGATIVE_GENRE_WEIGHTS = {"business": 0.30, "news": 0.20, "life": 0.20, "literary": 0.15, "rules": 0.10, "chat": 0.05}

# ---------------------------------------------------------------- 帳票（v9）: メール・URL の非エンティティ値
FORM_FRAMES_V9: tuple[str, ...] = tuple(f for f in FORM_FRAMES if not f.startswith(("TO:", "cc:")))
# {email} {url} {domain} は非エンティティ。{k} は姓（PERSON か役割語）
EMAIL_FRAMES: tuple[str, ...] = (
    "To: {email}", "TO: {email}", "cc: {email}", "Cc: {email}", "From: {email}", "Reply-To: {email}", "Bcc: {email}",
    "連絡先: {email}", "問い合わせ: {email}", "返信先 {email}", "メール: {email}", "送信元 {email}", "宛先 {email}",
    "URL: {url}", "参照: {url}", "詳細は {url} を参照", "リンク: {url}", "DKIM pass d={domain}", "host={domain}",
    "user={email}", "owner={email}", "reporter={email}", "mail_from={email} result={res}",
    'From: "{k}" <{email}>', "{k} <{email}>", "To: {k} 様 ({email})", "担当: {k}（{email}）", "承認者 {k} / {email}",
    "問い合わせ先: {k}　{email}",
)
LOCAL_GENERIC: tuple[str, ...] = ("support", "info", "sales", "sales-team", "noreply", "no-reply", "contact", "admin", "help",
                                  "billing", "hr", "recruit", "press", "order", "service", "team", "dev", "ops", "notice")
DOMAIN_WORDS: tuple[str, ...] = ("north", "gate", "tech", "foods", "freight", "logistics", "works", "labs", "shop", "mall",
                                 "harbor", "river", "green", "blue", "sakura", "hikari", "aoba", "minato", "kaze", "hoshi",
                                 "kk", "co", "group", "trade", "system", "design", "medical", "clinic", "office", "net")
TLDS: tuple[str, ...] = ("example.com", "example.co.jp", "example.jp", "example.net", "example.org", "example.ne.jp")


def _handle_bank(latin_given: Sequence[str], latin_sur: Sequence[str]) -> tuple[list[str], list[str]]:
    lg = [g.lower() for g in latin_given if re.fullmatch(r"[A-Za-z]+", g)] or ["maria", "john", "ken", "yuki", "sora"]
    ls = [s.lower() for s in latin_sur if re.fullmatch(r"[A-Za-z]+", s)] or ["ivers", "smith", "sato", "mori", "hayashi"]
    return lg, ls


def make_email(rng: random.Random, lg: Sequence[str], ls: Sequence[str]) -> str:
    r = rng.random()
    if r < 0.30:
        local = rng.choice(LOCAL_GENERIC)
    elif r < 0.55:
        local = f"{rng.choice(lg)[0]}.{rng.choice(ls)}"
    elif r < 0.75:
        local = f"{rng.choice(lg)}.{rng.choice(ls)}"
    elif r < 0.88:
        local = f"{rng.choice(ls)}{rng.randint(1, 99)}"
    else:
        local = f"{rng.choice(lg)}_{rng.choice(ls)[:3]}"
    return f"{local}@{make_domain(rng, ls)}"


def make_domain(rng: random.Random, ls: Sequence[str]) -> str:
    r = rng.random()
    if r < 0.4:
        stem = f"{rng.choice(ls)}-{rng.choice(DOMAIN_WORDS)}"
    elif r < 0.7:
        stem = f"{rng.choice(DOMAIN_WORDS)}-{rng.choice(DOMAIN_WORDS)}"
    else:
        stem = rng.choice(ls)
    return f"{stem}.{rng.choice(TLDS)}"


def make_url(rng: random.Random, ls: Sequence[str]) -> str:
    path = rng.choice(("", "/", "/faq", "/support/ticket/{d}", "/docs/v{d}", "/news/2024/{d}", "/{w}/{w2}.html"))
    path = path.replace("{d}", str(rng.randint(1, 9999))).replace("{w}", rng.choice(DOMAIN_WORDS)).replace("{w2}", rng.choice(DOMAIN_WORDS))
    return f"{rng.choice(('https://', 'http://', 'https://www.'))}{make_domain(rng, ls)}{path}"


def make_form_docs_v9(n: int, rng: random.Random, surnames: Sequence[str], given: Sequence[str],
                      latin_given: Sequence[str], latin_sur: Sequence[str], *, exclusions: set[str] | None = None,
                      pure_negative_ratio: float = FORM_PURE_NEGATIVE_RATIO, name_slot_ratio: float = FORM_NAME_SLOT_RATIO,
                      email_line_ratio: float = 0.45, name_fn=None) -> list[dict]:
    """v8 の帳票・ログに、メール・URL を値にした非エンティティ行を混ぜる。`TO:` `cc:` の姓フレームは使わない。
    name_fn(rng, surnames, given, kata, exclusions) -> str で名の表層の作り方を差し替えられる（v10 の字間空白の姓など）。"""
    exclusions = exclusions or set()
    kata = [k for k in SURNAMES_KATA_JA if k not in exclusions] or list(SURNAMES_KATA_JA)
    lg, ls = _handle_bank(latin_given, latin_sur)
    _name_surface_fn = name_fn or _name_surface
    docs: list[dict] = []
    for _ in range(n):
        style = rng.choice(("log", "form", "form", "mixed"))
        pure_neg = rng.random() < pure_negative_ratio
        d = _Doc(FORM_SOURCE, f"form9:{style}:{'neg' if pure_neg else 'pos'}")
        n_lines = rng.randint(3, 7)
        frames = LOG_FRAMES if style == "log" else FORM_FRAMES_V9 if style == "form" else LOG_FRAMES + FORM_FRAMES_V9
        fillers = LOG_FILLERS if style == "log" else FORM_FILLERS if style == "form" else LOG_FILLERS + FORM_FILLERS
        n_frames = rng.randint(1, min(3, n_lines))
        lines = [(True, rng.choice(frames)) for _ in range(n_frames)] + \
                [(False, rng.choice(fillers)) for _ in range(n_lines - n_frames)]
        if rng.random() < email_line_ratio:
            lines.append((True, rng.choice(EMAIL_FRAMES)))
        rng.shuffle(lines)
        header = rng.choice(("", "", "", "点検記録\n", "作業日報\n", "audit log\n", "納 品 書\n", "引継ぎ表\n", "件名: 見積依頼\n"))
        d.text(header)
        has_name = False
        for is_frame, raw in lines:
            line = _fill_values(raw, rng)
            line = re.sub(r"\{email\}", lambda _: make_email(rng, lg, ls), line)
            line = re.sub(r"\{url\}", lambda _: make_url(rng, ls), line)
            line = re.sub(r"\{domain\}", lambda _: make_domain(rng, ls), line)
            if not is_frame:
                d.text(line)
                d.text("\n")
                continue
            is_log = "=" in raw or raw.startswith("[") or ("→" in raw and "handover" in raw)
            for tok in re.split(r"(\{k2?\}|\{shift\})", line):
                if tok == "{shift}":
                    d.text(rng.choice(SHIFTS))
                elif tok in ("{k}", "{k2}"):
                    if not pure_neg and rng.random() < name_slot_ratio:
                        d.ent(_name_surface_fn(rng, surnames, given, kata, exclusions), "PERSON")
                        has_name = True
                    else:
                        d.text(rng.choice(NEG_LOG_VALUES if is_log else NEG_FORM_VALUES))
                elif tok:
                    d.text(tok)
            d.text("\n")
        if not pure_neg and not has_name:
            d.text(rng.choice(("担当: ", "記録者: ", "operator=", "確認 ")))
            d.ent(_name_surface_fn(rng, surnames, given, kata, exclusions), "PERSON")
            d.text("\n")
        doc = d.build()
        doc["text"] = doc["text"].rstrip("\n")
        docs.append(doc)
    return docs


# ---------------------------------------------------------------- カタカナ人名の文書
KATAKANA_FRAMES: tuple[str, ...] = (
    "{n}は記者会見で今後の方針を語った。", "作家の{n}が来日し、{u}", "{n}監督の新作が公開された。", "{n}氏（{d}）は、{u}",
    "主演は{n}。", "{n}と{n2}が共演する。", "{n}が代表に選ばれた。", "{n}の著書を読んだ。", "元選手の{n}は解説者として出演した。",
    "{n}教授の講演を聴いた。", "科学者{n}の理論は{u}", "{n}は「{u}」と述べた。", "{n}さんが来店した。", "{n}さん、{u}",
    "投手の{n}が先発した。", "デザイナー{n}の展示が始まる。", "{n}氏と{n2}氏が会談した。", "研究者の{n}によれば、{u}",
    "{n}のアルバムを聴く。", "{n}は{d}年に生まれた。", "映画は{n}が脚本を書いた。", "受賞者は{n}。", "{n}が新記録を出した。",
    "{n}博士の研究室を訪ねた。", "{n}を招いて講演会を開く。", "{n}率いるチームが優勝した。", "{n}は{n2}に手紙を書いた。",
    "本日の講師は{n}です。", "{n}の小説『冬の旅』を読み返した。", "{n}選手が引退を表明した。",
)


def katakana_name_bank(exclusions: set[str], rng: random.Random) -> tuple[list[str], list[str]]:
    """外国名・日本人名のカタカナ表層。評価セットの PERSON 表層と一致するものは除く。"""
    given = list(dict.fromkeys(GIVEN_KATA + GIVEN_KATA_JA + GIVEN_KATA_SHORT))
    sur = list(dict.fromkeys(SURNAMES_KATA + SURNAMES_KATA_JA))
    names: list[str] = []
    for s in sur:
        names.append(s)
    for g in given:
        names.append(g)
    for _ in range(300):
        g, s = rng.choice(given), rng.choice(sur)
        names.append(rng.choice((f"{g}・{s}", f"{g} {s}", f"{s} {g}", f"{s}・{g}")))
    names = [x for x in dict.fromkeys(names) if x not in exclusions and unicodedata.normalize("NFKC", x) not in exclusions]
    rng.shuffle(names)
    k = max(1, int(len(names) * VALID_BANK_RATIO))
    return names[k:], names[:k]


def make_katakana_docs(n: int, rng: random.Random, names: Sequence[str], utterances: Sequence[str],
                       fillers: Sequence[str]) -> list[dict]:
    """カタカナ人名（外国名・日本名）を報道・紹介・会話のフレームで PERSON にする。敬称・助詞は span 外。"""
    docs = []
    for _ in range(n):
        d = _Doc(KATAKANA_SOURCE, "katakana")
        cast = rng.sample(list(names), 2)
        n_lines = rng.randint(2, 4)
        # 少なくとも 1 行は名を含むフレーム（全行が地の文だと負例になってしまう）
        frame_lines = {rng.randrange(n_lines)}
        for li in range(n_lines):
            if li not in frame_lines and rng.random() < 0.3:
                d.text(rng.choice(fillers) + "\n")
                continue
            frame = rng.choice(KATAKANA_FRAMES)
            u = rng.choice(utterances).rstrip("！。!")
            frame = frame.replace("{d}", str(rng.randint(19, 1999)))
            for tok in re.split(r"(\{n2?\}|\{u\})", frame):
                if tok == "{n}":
                    d.ent(cast[0], "PERSON")
                elif tok == "{n2}":
                    d.ent(cast[1], "PERSON")
                elif tok == "{u}":
                    d.text(u)
                elif tok:
                    d.text(tok)
            d.text("\n")
        doc = d.build()
        doc["text"] = doc["text"].rstrip("\n")
        docs.append(doc)
    return docs


# ---------------------------------------------------------------- 裸の町名の文書
EXTRA_TOWNS: tuple[str, ...] = (
    "新町", "元町", "錦町", "千代田", "日吉", "松原", "清水町", "宮前", "御幸町", "平和台", "緑ヶ丘", "光が丘", "若草町",
    "白山", "三田", "大森", "本郷", "田町", "梅ヶ丘", "桜ヶ丘", "藤が丘", "南台", "北台", "中町", "仲町", "昭和町", "明治町",
    "大正町", "春日町", "朝日町", "旭ヶ丘", "松が丘", "浜町", "船町", "堀之内", "宮の前", "神明町", "住吉", "天神", "山王",
    "八幡町", "本通", "大通", "新田", "小山", "上川", "下川", "中島", "川上", "岡", "台", "坂", "森", "林", "原",
)
# 1 字の町名は「森」「林」などの一般語と衝突するので除く
TOWN_MIN_CHARS = 2
TOWN_FRAMES: tuple[str, ...] = (
    "{t}では{u}", "{t}にある古い書店へ行った。", "{t}から{t2}までは徒歩で十分ほどだ。", "{t}周辺の道路が混雑した。",
    "{t}方面のバスは十五分間隔で出ている。", "会場は{t}の集会所。", "{t}一帯で停電が起きた。", "{t}に住む知人を訪ねた。",
    "今日は{t}まで歩いた。", "{t}の交差点で事故があった。", "{t}と{t2}の境に川が流れている。", "引っ越し先は{t}に決めた。",
    "{t}は昔、宿場町だった。", "{t}の商店街は夕方になると賑わう。", "{t}行きの最終便に間に合った。", "{t}側の出口から出る。",
    "{t}の空き地に花が咲いていた。", "配達は{t}と{t2}を回る。", "{t}で生まれ育った。", "{t}の祭りは十月にある。",
)


def town_bank(exclusions: set[str], rng: random.Random) -> tuple[list[str], list[str]]:
    towns = [t for t in dict.fromkeys(TOWNS + EXTRA_TOWNS) if len(t) >= TOWN_MIN_CHARS and t not in exclusions]
    rng.shuffle(towns)
    k = max(1, int(len(towns) * VALID_BANK_RATIO))
    return towns[k:], towns[:k]


def make_town_docs(n: int, rng: random.Random, towns: Sequence[str], utterances: Sequence[str],
                   fillers: Sequence[str]) -> list[dict]:
    """番地の無い裸の町名を LOCATION にする。"""
    docs = []
    for _ in range(n):
        d = _Doc(TOWN_SOURCE, "town")
        t1, t2 = rng.sample(list(towns), 2)
        n_lines = rng.randint(1, 3)
        frame_lines = {rng.randrange(n_lines)}   # 少なくとも 1 行は町名を含む
        for li in range(n_lines):
            if li not in frame_lines and rng.random() < 0.3:
                d.text(rng.choice(fillers) + "\n")
                continue
            frame = rng.choice(TOWN_FRAMES)
            u = rng.choice(utterances).rstrip("！。!") + "。"
            for tok in re.split(r"(\{t2?\}|\{u\})", frame):
                if tok == "{t}":
                    d.ent(t1, "LOCATION")
                elif tok == "{t2}":
                    d.ent(t2, "LOCATION")
                elif tok == "{u}":
                    d.text(u)
                elif tok:
                    d.text(tok)
            d.text("\n")
        doc = d.build()
        doc["text"] = doc["text"].rstrip("\n")
        docs.append(doc)
    return docs


# ---------------------------------------------------------------- LLM 負例の抽出

def negative_genre(row: dict) -> str:
    return (row.get("variant") or "").split("/")[0] or "unknown"


def select_negatives(rows: Sequence[dict], n: int, rng: random.Random, hard_texts: Iterable[str] = (),
                     weights: dict[str, float] = NEGATIVE_GENRE_WEIGHTS) -> list[dict]:
    """ジャンル比で n 件。NER が何かを検出したハード負例を各ジャンルで先に取り、残りを無作為に埋める。"""
    hard = set(hard_texts)
    by_genre: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_genre[negative_genre(r)].append(r)
    quota = {g: int(round(n * w)) for g, w in weights.items()}
    # 丸め誤差は最大ジャンルで吸収
    diff = n - sum(quota.values())
    quota[max(quota, key=quota.get)] += diff
    picked: list[dict] = []
    leftovers: list[dict] = []
    for g, q in quota.items():
        pool = list(by_genre.get(g, []))
        rng.shuffle(pool)
        pool.sort(key=lambda r: 0 if r["text"] in hard else 1)   # 安定ソートなのでハードが先、ハード内・非ハード内は無作為
        picked.extend(pool[:q])
        leftovers.extend(pool[q:])
    if len(picked) < n:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])
    return picked


def load_hard_negative_texts(drive: str) -> set[str]:
    """`negatives_v8/filtered_v1/ner_hits.csv` の request_id を raw 出力（lines 形式）に突き合わせて本文を得る。"""
    hits = os.path.join(drive, "negatives_v8", "filtered_v1", "ner_hits.csv")
    raw = os.path.join(drive, "qwen3_8_27b_negatives_v8_plan_v3_full_seed42", "qwen_negatives_raw.jsonl")
    if not (os.path.exists(hits) and os.path.exists(raw)):
        return set()
    from negatives_v8 import parsed_text
    ids = {r["request_id"] for r in csv.DictReader(open(hits, encoding="utf-8"))}
    texts: set[str] = set()
    with open(raw, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("request_id") in ids:
                t = parsed_text(r.get("parsed"))
                if t:
                    texts.add(t)
    return texts


def split_rows(rows: Sequence[dict], rng: random.Random, ratio: float = NEG_VALID_RATIO) -> tuple[list[dict], list[dict]]:
    xs = list(rows)
    rng.shuffle(xs)
    k = int(len(xs) * ratio)
    return xs[k:], xs[:k]


# ---------------------------------------------------------------- 評価セットの表層

def collect_eval_surfaces(labels: Sequence[str]) -> set[str]:
    surfaces: set[str] = set()
    try:
        import eval_ood_f1 as f1
        sets = f1.load_sets([])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 評価セットを読めません（{exc}）")
        sets = {}
    for name, (docs, _) in sets.items():
        n = 0
        for d in docs:
            for ent in d.get("entities", []):
                s, e, lab = (ent[0], ent[1], ent[2]) if isinstance(ent, (list, tuple)) else (ent["start"], ent["end"], ent.get("label"))
                if lab in labels:
                    surfaces.add(d["text"][s:e])
                    n += 1
        print(f"  評価セット {name}: {'/'.join(labels)} {n} 件")
    return surfaces


# ---------------------------------------------------------------- 全体

def build(records: list[dict], *, cap: int, seed: int, n_chat: int, n_dialogue: int, n_halfwidth: int, n_forms: int,
          n_katakana: int, n_towns: int, negatives: Sequence[dict], hard_texts: set[str], eval_person: set[str],
          eval_place: set[str], form_kwargs: dict | None = None, surname_bank_fn=None) -> tuple[list[dict], list[dict], dict]:
    """form_kwargs は make_form_docs_v9 への追加引数（v10: name_slot_ratio・name_fn）。
    surname_bank_fn(exclusions, rng) -> (train, valid) で姓の候補の作り方を差し替えられる（v10: 頻度の低い姓の比率）。"""
    form_kwargs = dict(form_kwargs or {})
    rng = random.Random(seed)
    before = stats(records)
    relabeled = relabel_wiki_addresses(records)
    capped, dropped = cap_templates(records, cap, rng)
    latin_given, latin_sur = _harvest_name_parts(capped)
    addr_aug, _, _ = augment_addresses(capped, per_doc=1, seed=seed)
    for r in addr_aug:
        r["variant"] = "address:" + r["variant"]
    person_aug, person_made = augment_persons(capped, ratio=PERSON_VARIANT_RATIO, rng=rng,
                                              latin_given=latin_given, latin_sur=latin_sur)
    combined = capped + addr_aug + person_aug
    train, valid = grouped_split(combined, loose=True, seed=seed)
    tsig = {template_signature(r, loose=True) for r in train}
    leak = sum(1 for r in valid if template_signature(r, loose=True) in tsig)

    utt_tr, utt_va = _split_bank(CHAT_UTTERANCES, rng, VALID_BANK_RATIO)
    fil_tr, fil_va = _split_bank(NARRATIVE_FILLERS, rng, VALID_BANK_RATIO)
    biz_tr, biz_va = _split_bank(BUSINESS_LINES, rng, VALID_BANK_RATIO)
    sur_tr, sur_va = (surname_bank_fn or surname_bank)(eval_person, rng)
    kata_tr, kata_va = katakana_name_bank(eval_person, rng)
    town_tr, town_va = town_bank(eval_place | eval_person, rng)
    given = list(dict.fromkeys(GIVEN_KANJI + GIVEN_KANJI_F))
    nv = lambda n: max(1, int(n * NEW_DOC_VALID_RATIO))  # noqa: E731
    new_train = (make_chat_docs(n_chat - nv(n_chat), rng, utt_tr, latin_given, latin_sur)
                 + make_dialogue_docs(n_dialogue - nv(n_dialogue), rng, fil_tr, utt_tr)
                 + make_halfwidth_docs(n_halfwidth - nv(n_halfwidth), rng, biz_tr)
                 + make_form_docs_v9(n_forms - nv(n_forms), rng, sur_tr, given, latin_given, latin_sur, exclusions=eval_person, **form_kwargs)
                 + make_katakana_docs(n_katakana - nv(n_katakana), rng, kata_tr, utt_tr, fil_tr)
                 + make_town_docs(n_towns - nv(n_towns), rng, town_tr, utt_tr, fil_tr))
    new_valid = (make_chat_docs(nv(n_chat), rng, utt_va, latin_given, latin_sur)
                 + make_dialogue_docs(nv(n_dialogue), rng, fil_va, utt_va)
                 + make_halfwidth_docs(nv(n_halfwidth), rng, biz_va)
                 + make_form_docs_v9(nv(n_forms), rng, sur_va, given, latin_given, latin_sur, exclusions=eval_person, **form_kwargs)
                 + make_katakana_docs(nv(n_katakana), rng, kata_va, utt_va, fil_va)
                 + make_town_docs(nv(n_towns), rng, town_va, utt_va, fil_va))
    bad = check_no_eval_surfaces(new_train + new_valid)
    if bad:
        raise SystemExit(f"生成物に盲検セットの表層が混ざっています: {sorted(set(bad))}")
    bad = check_no_eval_person_surfaces([d for d in new_train + new_valid if d["source"] in (FORM_SOURCE, KATAKANA_SOURCE)], eval_person)
    if bad:
        raise SystemExit(f"帳票・カタカナ文書に評価セットの PERSON 表層が混ざっています: {sorted(set(bad))[:10]}")
    bad_loc = [d["text"][s:e] for d in new_train + new_valid if d["source"] == TOWN_SOURCE
               for s, e, l in d["entities"] if d["text"][s:e] in eval_place]
    if bad_loc:
        raise SystemExit(f"町名文書に評価セットの地名表層が混ざっています: {sorted(set(bad_loc))[:10]}")
    new_leak = len({d["text"] for d in new_train} & {d["text"] for d in new_valid})
    train += new_train
    valid += new_valid

    long_train = make_long_docs(train, target_ratio=LONG_TARGET_RATIO, rng=rng)
    long_valid = make_long_docs(valid, target_ratio=LONG_TARGET_RATIO, rng=rng)
    train += long_train
    valid += long_valid

    neg_train, neg_valid = split_rows(negatives, rng)
    train += neg_train
    valid += neg_valid
    rng.shuffle(train)
    rng.shuffle(valid)
    new_all = new_train + new_valid

    def count(src: str) -> dict:
        ds = [d for d in new_all if d["source"] == src]
        return {"docs": len(ds), "entities": sum(len(d["entities"]) for d in ds), "negatives": sum(1 for d in ds if not d["entities"])}

    info = {"seed": seed, "cap": cap, "relabeled": relabeled, "dropped": dict(dropped),
            "addr_variants": len(addr_aug), "person_variants": len(person_aug),
            "person_made": person_made, "long_train": len(long_train), "long_valid": len(long_valid),
            "leak": leak, "before": before,
            "new_docs": {"chat": n_chat, "dialogue": n_dialogue, "halfwidth": n_halfwidth, "forms": n_forms,
                         "katakana": n_katakana, "towns": n_towns, "train": len(new_train), "valid": len(new_valid),
                         "text_overlap": new_leak},
            "forms": {**count(FORM_SOURCE), "email_lines": sum(d["text"].count("@") for d in new_all if d["source"] == FORM_SOURCE),
                      "surname_bank_train": len(sur_tr), "surname_bank_valid": len(sur_va)},
            "katakana": {**count(KATAKANA_SOURCE), "bank_train": len(kata_tr), "bank_valid": len(kata_va)},
            "towns": {**count(TOWN_SOURCE), "bank_train": len(town_tr), "bank_valid": len(town_va)},
            "eval_exclusions": {"person": len(eval_person), "place": len(eval_place)},
            "llm_negatives": {"selected": len(negatives), "hard": sum(1 for r in negatives if r["text"] in hard_texts),
                              "train": len(neg_train), "valid": len(neg_valid),
                              "by_genre": dict(Counter(negative_genre(r) for r in negatives))}}
    return train, valid, info


def v9_appendix(info: dict, train: Sequence[dict], valid: Sequence[dict]) -> str:
    tr, va = Counter(r["source"] for r in train), Counter(r["source"] for r in valid)
    ntr, nva = Counter(r["source"] for r in train if not r["entities"]), Counter(r["source"] for r in valid if not r["entities"])
    f, k, t, ln = info["forms"], info["katakana"], info["towns"], info["llm_negatives"]
    L = ["", "## v9 の追加分", "",
         f"- テンプレート上限 {info['cap']}（v8 は 6）。落とした文書: {sum(info['dropped'].values())}",
         f"- 新規文書: チャット {info['new_docs']['chat']} / 会話 {info['new_docs']['dialogue']} / 半角カナ {info['new_docs']['halfwidth']}"
         f" / 帳票・ログ {info['new_docs']['forms']} / **カタカナ人名 {info['new_docs']['katakana']}** / **裸の町名 {info['new_docs']['towns']}**"
         f"（train {info['new_docs']['train']} / valid {info['new_docs']['valid']}、本文一致 {info['new_docs']['text_overlap']}）",
         f"- 帳票・ログ: PERSON {f['entities']} 件、純負例 {f['negatives']} 文書、メールアドレスを含む行 {f['email_lines']}、"
         f"`TO:` `cc:` の姓フレームなし。姓の候補 train {f['surname_bank_train']} / valid {f['surname_bank_valid']}",
         f"- カタカナ人名: PERSON {k['entities']} 件、候補 train {k['bank_train']} / valid {k['bank_valid']}",
         f"- 裸の町名: LOCATION {t['entities']} 件、候補 train {t['bank_train']} / valid {t['bank_valid']}",
         f"- 評価セットの表層の除外: PERSON {info['eval_exclusions']['person']} 種、LOCATION / ADDRESS {info['eval_exclusions']['place']} 種",
         f"- LLM 負例（{LLM_NEGATIVE_SOURCE}）: {ln['selected']} 件（うちハード負例 {ln['hard']}）→ train {ln['train']} / valid {ln['valid']}。"
         f"ジャンル {ln['by_genre']}。`train_v9_noneg.jsonl` はこれを除いた版",
         "", "| 出所 | train | valid | うち負例 train | うち負例 valid |", "|---|---:|---:|---:|---:|"]
    for s in sorted(set(tr) | set(va)):
        L.append(f"| {s} | {tr.get(s, 0)} | {va.get(s, 0)} | {ntr.get(s, 0)} | {nva.get(s, 0)} |")
    n_tr = len(train)
    L += ["", f"- train の負例 {sum(ntr.values())} / {n_tr}（{sum(ntr.values()) / max(n_tr, 1) * 100:.1f}%）",
          f"- train の実テキスト out_ner {tr.get('out_ner', 0)} / {n_tr}（{tr.get('out_ner', 0) / max(n_tr, 1) * 100:.1f}%）、"
          f"LLM 負例を除く母数では {tr.get('out_ner', 0) / max(n_tr - tr.get(LLM_NEGATIVE_SOURCE, 0), 1) * 100:.1f}%"]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chat", type=int, default=DEFAULT_CHAT)
    ap.add_argument("--dialogue", type=int, default=DEFAULT_DIALOGUE)
    ap.add_argument("--halfwidth", type=int, default=DEFAULT_HALFWIDTH)
    ap.add_argument("--forms", type=int, default=DEFAULT_FORMS)
    ap.add_argument("--katakana", type=int, default=DEFAULT_KATAKANA)
    ap.add_argument("--towns", type=int, default=DEFAULT_TOWNS)
    ap.add_argument("--negatives", default=None, help="既定: <データフォルダ>/negatives_v8/filtered_v1/negatives_v8.jsonl")
    ap.add_argument("--n-negatives", type=int, default=DEFAULT_N_NEGATIVES)
    args = ap.parse_args(argv)
    if args.cap < 1:
        raise SystemExit("--cap は 1 以上")

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書")
    neg_path = args.negatives or os.path.join(gd, "negatives_v8", "filtered_v1", "negatives_v8.jsonl")
    all_neg = load_negatives(neg_path)
    hard = load_hard_negative_texts(gd)
    negatives = select_negatives(all_neg, args.n_negatives, random.Random(args.seed), hard)
    print(f"LLM 負例: {len(all_neg)} → {len(negatives)} 件（ハード {sum(1 for r in negatives if r['text'] in hard)} / {len(hard)}）")
    eval_person = eval_person_exclusions(collect_eval_surfaces(("PERSON",)))
    eval_place = collect_eval_surfaces(("LOCATION", "ADDRESS"))
    print(f"除外: PERSON 表層 {len(eval_person)} 種、LOCATION / ADDRESS 表層 {len(eval_place)} 種")

    train, valid, info = build(records, cap=args.cap, seed=args.seed, n_chat=args.chat, n_dialogue=args.dialogue,
                               n_halfwidth=args.halfwidth, n_forms=args.forms, n_katakana=args.katakana, n_towns=args.towns,
                               negatives=negatives, hard_texts=hard, eval_person=eval_person, eval_place=eval_place)
    if info["leak"]:
        raise SystemExit(f"train / valid にテンプレート重複が {info['leak']} 件あります。生成を中止")

    os.makedirs(args.out_dir, exist_ok=True)
    train_noneg = without_llm_negatives(train)
    write_jsonl(os.path.join(args.out_dir, "train_v9.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "train_v9_noneg.jsonl"), train_noneg)
    write_jsonl(os.path.join(args.out_dir, "valid_v9.jsonl"), valid)
    st_train, st_valid, st_all = stats(train), stats(valid), stats(train + valid)
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [f for _, f in DATA_FILES] + [os.path.relpath(neg_path, gd)],
        "seed": args.seed, "cap": args.cap, "labels": list(BASE_LABELS),
        "train": {"docs": len(train), "text_digest": _digest(train), **st_train},
        "train_noneg": {"docs": len(train_noneg), "text_digest": _digest(train_noneg)},
        "valid": {"docs": len(valid), "text_digest": _digest(valid), **st_valid},
        "processing": {k: v for k, v in info.items() if k not in ("before", "person_made")},
        "person_variants_made": dict(info["person_made"]),
        "files": {"train": "train_v9.jsonl", "train_noneg": "train_v9_noneg.jsonl", "valid": "valid_v9.jsonl"},
    }
    with open(os.path.join(args.out_dir, "manifest_v9.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    card = data_card("学習データ v9", info["before"], st_all, st_train, st_valid, info, version="v9")
    card = card.replace(
        "- LLM 生成の新規負例（CPU では作れない。`qwen_synth_pii_data` 系ノートブックで別途）",
        "- 評価セット由来の人名・地名表層（帳票・カタカナ・町名の候補から機械的に除外）")
    card += v9_appendix(info, train, valid)
    with open(os.path.join(args.out_dir, "DATA_CARD_v9.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

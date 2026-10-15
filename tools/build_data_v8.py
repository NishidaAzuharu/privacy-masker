"""学習データ v8 — v7 の部品に「帳票・ログの裸の姓」と「LLM 生成の負例」を足し、テンプレート合成を少し減らす。

v7（`build_data_v7.py`。結果は開発記録、非公開）:
盲検 v1 のハンドル名・短名・半角カナは直ったが、固定OOD 危険 14 → 24、GSD 21 → 32 と悪化。
落ちたのは (a) 公知の LOCATION / ORG（東京・中東・与党・デジタル庁）と (b) 帳票・ログの裸の姓
（`operator=<姓>` `記録者：夜勤 <姓>` `宅地建物取引士: <姓>`）。新規文書を 3,300 件入れて分布が偏り、
実テキスト（out_ner）の比重が下がった。負例は 1,035 件（3.6%）しかなく過剰マスクは併用で +14〜31%。

v8 の変更（設計は開発記録、非公開）:

| 変更 | v7 | v8 |
|---|---|---|
| テンプレート上限（loose 署名あたり） | 8 | **6**（v5c を 9,486 → 7,564。実テキストの比重を上げる。v6 の 4 は感度を落とした） |
| チャット / 会話 / 半角カナの新規文書 | 1,500 / 1,500 / 500 | **750 / 750 / 250**（半減） |
| **帳票・ログのフレーム**（`key=value` と `ラベル: 値` の裸の姓） | なし | **1,100 文書**。同じフレームで値が役割語・状態語の行を混ぜる（対の負例） |
| **LLM 生成の負例**（Qwen3.8-27B-FP8、`negatives_v8/filtered_v1`） | なし | **4,000 文書**（train 3,600 / valid 400）。`train_v8_noneg.jsonl` はこれを除いた版 |

固定OOD・GSD・OpenPII・盲検 v1 の gold に出る PERSON 表層は、帳票フレームの姓の候補から機械的に除く
（珍しい姓を含む）。盲検のハンドル名（`EVAL_SURFACES`。データフォルダの eval_surfaces.json から読む）は従来どおり全生成物で確認する。

使い方::

    PYTHONPATH=. .venv/bin/python tools/build_data_v8.py --out-dir <Drive>/data_v8 [--cap 6] [--seed 42]
        [--chat 750] [--dialogue 750] [--halfwidth 250] [--forms 1100]
        [--negatives <Drive>/negatives_v8/filtered_v1/negatives_v8.jsonl]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
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
    to_halfwidth_kana,
)
from split_by_template import (  # noqa: E402
    BASE_LABELS,
    DATA_FILES,
    grouped_split,
    load_records,
    template_signature,
)
from privacy_masker.data.ja_pools import GIVEN_KANJI, GIVEN_KANJI_F, SURNAMES_KANJI  # noqa: E402

SEED = 42
DEFAULT_CAP = 6
DEFAULT_CHAT = 750
DEFAULT_DIALOGUE = 750
DEFAULT_HALFWIDTH = 250
DEFAULT_FORMS = 1100
NEG_VALID_RATIO = 0.10
FORM_PURE_NEGATIVE_RATIO = 0.25      # 帳票文書のうち、名が 1 つも無い（役割語・状態語だけ）文書の割合
FORM_NAME_SLOT_RATIO = 0.70          # 正例文書の名スロットのうち、実際に名を入れる割合（残りは役割語・状態語）
LLM_NEGATIVE_SOURCE = "qwen38_negatives_v8"
FORM_SOURCE = "synth_form"

# ---------------------------------------------------------------- 姓の候補
# ja_pools の 140 姓（頻出）に、頻度の低い実在の姓を足す。評価セットの gold に出る表層は build 時に除く。
EXTRA_SURNAMES: tuple[str, ...] = (
    "生駒", "綿貫", "五味", "花房", "江尻", "海老原", "大曲", "小柴", "各務", "鏑木", "神林", "郡司", "佃", "手嶋", "十河",
    "中條", "難波", "二宮", "沼尻", "波多野", "一柳", "平賀", "深沢", "二階堂", "保科", "真田", "三宅", "八幡", "湯浅",
    "早乙女", "東海林", "安達", "相沢", "赤城", "秋元", "浅井", "芥川", "朝倉", "足立", "飯塚", "生田", "池谷", "石垣",
    "磯野", "井筒", "犬飼", "岩本", "植木", "牛島", "宇野", "榎本", "江口", "大和田", "岡崎", "奥山", "尾崎", "織田",
    "甲斐", "加賀", "柏木", "梶原", "春日", "片桐", "桂", "加納", "神谷", "川端", "神田", "菊池", "岸本", "北川",
    "木戸", "日下部", "久世", "国枝", "熊谷", "倉田", "黒岩", "桑原", "小暮", "小谷", "児玉", "小西", "榊", "佐久間",
    "桜庭", "志賀", "篠原", "渋谷", "島崎", "白井", "陣内", "末永", "菅野", "杉浦", "鷲尾", "関口", "瀬戸", "曽根",
    "高瀬", "滝沢", "竹村", "田所", "立花", "谷川", "玉置", "千野", "塚本", "筒井", "角田", "寺島", "土井", "徳永",
    "富岡", "鳥居", "長岡", "中原", "名取", "成瀬", "新田", "西岡", "額田", "根本", "野島", "萩原", "橋爪", "畠山",
    "八田", "花岡", "浜口", "早川", "原口", "半田", "樋渡", "日野", "平岩", "藤川", "舟木", "古田", "別府", "星野",
    "堀内", "本間", "前川", "牧野", "松浦", "三上", "水谷", "三谷", "宮下", "向井", "武藤", "村瀬", "森脇", "矢田",
    "柳田", "山岸", "弓削", "湯川", "横田", "吉岡", "米田", "若林", "脇田", "和久井", "渡会",
)

# ---------------------------------------------------------------- 帳票・ログのフレーム
# {k} {k2} = 名（PERSON か、役割語・状態語）、{shift} = 勤務帯（span 外）、{date} {time} {lvl} {act} {tgt} {res} {d} = 値
LOG_FRAMES: tuple[str, ...] = (
    "{date} {time} audit INFO operator={k} action={act} target={tgt}",
    "{date} {time} {lvl} user={k} result={res}",
    "[{time}] approved_by={k} status={res}",
    "{time} checked_by={k} count={d}",
    "{date} {lvl} operator={k} retry={d}",
    "登録者={k} 更新日={date}",
    "担当={k} 件数={d} 状態={res}",
    "{time} INFO reviewer={k} action={act}",
    "{date} {time} {lvl} job={tgt} owner={k}",
    "assignee={k} priority={d} status={res}",
    "{time} 承認者={k} 結果={res}",
    "worker={k} → {k2} (handover {time})",
)
FORM_FRAMES: tuple[str, ...] = (
    "記録者：{shift} {k}", "記録者: {k}", "担当: {k}", "担当：{k}", "作成者　{k}", "承認　{k}", "確認者 {k}",
    "受付 {k}", "受付担当：{k}", "宅地建物取引士: {k}", "処方医 {k}", "薬剤師 {k}", "担当医：{k}", "看護師 {k}",
    "対応者: {k}", "報告者：{k}", "申請者 {k}", "点検者：{k}", "運転者 {k}", "監督員：{k}", "検査員 {k}",
    "起票 {k}　承認 {k2}", "作成 {k}／確認 {k2}", "担当者名：{k}", "氏名　{k}", "宛先: {k} 様", "{k} 様", "TO: {k}",
    "cc: {k}", "コメント（{k}）: 期限内に対応済み。", "{k}（{shift}）記入", "引継ぎ: {k} → {k2}", "立会 {k}",
    "面談者：{k}", "講師 {k}", "審査 {k}　決裁 {k2}", "回答者: {k}", "入力 {k}", "配達員：{k}",
)
LOG_FILLERS: tuple[str, ...] = (
    "{date} {time} INFO job={tgt} status={res}", "{time} {lvl} retry={d} backoff={d}s", "{date} {time} INFO queue depth={d}",
    "{time} DEBUG cache hit ratio={d}%", "{date} {time} WARN latency={d}ms", "{time} INFO batch size={d}",
    "host=app-{d} pid={d}", "{date} {time} INFO checkpoint saved", "{time} ERROR timeout after {d}s",
)
FORM_FILLERS: tuple[str, ...] = (
    "対象: 月次バッチ", "状態: 完了", "件数: {d}", "処理時間: 00:{d2}:{d2}", "備考: 再実行なし", "区分: 定期", "優先度: 中",
    "結果: 正常終了", "次回: 翌営業日", "対象サーバー: 本番系", "エラー: なし", "影響範囲: なし", "版: 2.4.{d}", "温度: 4.{d}℃",
    "在庫: {d}箱", "数量: {d}", "単価: 2,400円", "納期: 10営業日", "検収: 5営業日以内", "受注No: {d}", "工程: 検品",
    "拠点: 第2倉庫", "時刻: {time}", "天候: 晴", "作業: 定期点検", "結果: 良", "所要: {d}分", "日付: {date}",
    "特記：明日、訪問診療あり。", "申し送り：夜間の巡回は 2 回。", "本人コメント: 期限内に提出できた。", "総合: A",
) + BUSINESS_LINES
NEG_LOG_VALUES: tuple[str, ...] = (
    "admin", "system", "batch", "auto", "cron", "root", "service", "guest", "monitor", "scheduler", "operator", "unknown", "none",
    "night_shift", "day_shift", "team_a", "ops",
)
NEG_FORM_VALUES: tuple[str, ...] = (
    "未定", "夜勤担当", "夜勤", "早番", "遅番", "当番", "担当者未定", "上長", "窓口", "総合窓口", "本人", "各自", "自動", "システム",
    "未記入", "空欄", "同上", "別紙", "―", "済", "承認済", "確認済", "外注", "委託先", "保留", "代理", "不在", "後日",
)
SHIFTS: tuple[str, ...] = ("夜勤", "早番", "日勤", "遅番", "準夜", "当直")
LEVELS: tuple[str, ...] = ("INFO", "WARN", "ERROR", "DEBUG", "info", "warn")
ACTIONS: tuple[str, ...] = ("log_download", "restart", "export", "login", "update", "approve", "rollback", "sync", "archive")
TARGETS: tuple[str, ...] = ("report_submission", "nightly_job", "invoice_batch", "user_sync", "backup_daily", "inventory_check")
RESULTS: tuple[str, ...] = ("OK", "NG", "done", "pending", "failed", "success", "skipped", "成功", "失敗", "完了", "保留")


def _rand_date(rng: random.Random) -> str:
    y, m, d = rng.randint(2019, 2027), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice((f"{y}-{m:02d}-{d:02d}", f"{y}/{m:02d}/{d:02d}", f"{y}年{m}月{d}日", f"{m:02d}/{d:02d}"))


def _rand_time(rng: random.Random) -> str:
    h, mi, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    return rng.choice((f"{h:02d}:{mi:02d}:{s:02d}", f"{h:02d}:{mi:02d}"))


def _fill_values(frame: str, rng: random.Random) -> str:
    """名スロット以外を埋める。{k} {k2} {shift} は残す。"""
    out = frame
    out = re.sub(r"\{date\}", lambda _: _rand_date(rng), out)
    out = re.sub(r"\{time\}", lambda _: _rand_time(rng), out)
    out = re.sub(r"\{lvl\}", lambda _: rng.choice(LEVELS), out)
    out = re.sub(r"\{act\}", lambda _: rng.choice(ACTIONS), out)
    out = re.sub(r"\{tgt\}", lambda _: rng.choice(TARGETS), out)
    out = re.sub(r"\{res\}", lambda _: rng.choice(RESULTS), out)
    out = re.sub(r"\{d2\}", lambda _: f"{rng.randint(0, 59):02d}", out)
    out = re.sub(r"\{d\}", lambda _: str(rng.randint(1, 999)), out)
    return out


def eval_person_exclusions(surfaces: Iterable[str]) -> set[str]:
    """評価セットの PERSON 表層から、姓候補として使ってはいけない文字列を作る。
    表層そのもの、空白区切りの先頭要素、3 文字以上の姓なら前方一致でも除く（<4 字の姓><カタカナ名> → <姓>）。"""
    out: set[str] = set()
    for s in surfaces:
        s = s.strip()
        if not s:
            continue
        out.add(s)
        parts = [p for p in re.split(r"[ 　・･]+", s) if p]
        if len(parts) >= 2:
            out.add(parts[0])
    return out


def surname_bank(exclusions: set[str], rng: random.Random) -> tuple[list[str], list[str]]:
    """姓の候補（頻出 140 + 追加）から評価セットの表層を除き、train / valid 用に分ける。"""
    pool = list(dict.fromkeys(SURNAMES_KANJI + EXTRA_SURNAMES))
    long_ex = [x for x in exclusions if len(x) >= 3]
    kept = [s for s in pool if s not in exclusions and not (len(s) >= 3 and any(x.startswith(s) for x in long_ex))]
    rng.shuffle(kept)
    k = max(1, int(len(kept) * VALID_BANK_RATIO))
    return kept[k:], kept[:k]


def make_form_docs(n: int, rng: random.Random, surnames: Sequence[str], given: Sequence[str],
                   *, pure_negative_ratio: float = FORM_PURE_NEGATIVE_RATIO,
                   name_slot_ratio: float = FORM_NAME_SLOT_RATIO, exclusions: set[str] | None = None,
                   kata_surnames: Sequence[str] = SURNAMES_KATA_JA) -> list[dict]:
    """帳票・ログ。`key=value` と `ラベル: 値` の値に裸の姓（PERSON）を置き、同じフレームで役割語・状態語の値も混ぜる。
    exclusions（評価セットの PERSON 表層）に当たる表層は、カタカナ姓・半角カナ姓・姓名の組み合わせも含めて作らない。"""
    exclusions = exclusions or set()
    kata = [k for k in kata_surnames if k not in exclusions] or list(kata_surnames)
    docs: list[dict] = []
    for _ in range(n):
        style = rng.choice(("log", "form", "form", "mixed"))
        pure_neg = rng.random() < pure_negative_ratio
        d = _Doc(FORM_SOURCE, f"form:{style}:{'neg' if pure_neg else 'pos'}")
        n_lines = rng.randint(3, 7)
        frames = LOG_FRAMES if style == "log" else FORM_FRAMES if style == "form" else LOG_FRAMES + FORM_FRAMES
        fillers = LOG_FILLERS if style == "log" else FORM_FILLERS if style == "form" else LOG_FILLERS + FORM_FILLERS
        n_frames = rng.randint(1, min(3, n_lines))
        lines = [(True, rng.choice(frames)) for _ in range(n_frames)] + \
                [(False, rng.choice(fillers)) for _ in range(n_lines - n_frames)]
        rng.shuffle(lines)
        header = rng.choice(("", "", "", "点検記録\n", "作業日報\n", "audit log\n", "納 品 書\n", "引継ぎ表\n", "処方内容\n"))
        d.text(header)
        has_name = False
        for is_frame, raw in lines:
            line = _fill_values(raw, rng)
            if not is_frame:
                d.text(line)
                d.text("\n")
                continue
            is_log = "=" in raw or raw.startswith("[") or "→" in raw and "handover" in raw
            for tok in re.split(r"(\{k2?\}|\{shift\})", line):
                if tok == "{shift}":
                    d.text(rng.choice(SHIFTS))
                elif tok in ("{k}", "{k2}"):
                    if not pure_neg and rng.random() < name_slot_ratio:
                        d.ent(_name_surface(rng, surnames, given, kata, exclusions), "PERSON")
                        has_name = True
                    else:
                        d.text(rng.choice(NEG_LOG_VALUES if is_log else NEG_FORM_VALUES))
                elif tok:
                    d.text(tok)
            d.text("\n")
        if not pure_neg and not has_name:
            # 正例文書には少なくとも 1 つ名を入れる
            d.text(rng.choice(("担当: ", "記録者: ", "operator=", "確認 ")))
            d.ent(_name_surface(rng, surnames, given, kata, exclusions), "PERSON")
            d.text("\n")
        doc = d.build()
        doc["text"] = doc["text"].rstrip("\n")
        docs.append(doc)
    return docs


def _name_surface(rng: random.Random, surnames: Sequence[str], given: Sequence[str],
                  kata: Sequence[str] = SURNAMES_KATA_JA, exclusions: set[str] | None = None) -> str:
    """名の表層。裸の姓 72% / 姓 名 12% / カタカナ姓 8% / 半角カナ姓 8%。評価セットの表層に当たったら引き直す。"""
    exclusions = exclusions or set()
    for _ in range(20):
        r = rng.random()
        sur = rng.choice(surnames)
        if r < 0.72:
            cand = sur
        elif r < 0.84:
            cand = f"{sur}{rng.choice((' ', '　', ''))}{rng.choice(given)}"
        elif r < 0.92:
            cand = rng.choice(kata)
        else:
            cand = to_halfwidth_kana(rng.choice(kata))
        if cand not in exclusions and unicodedata.normalize("NFKC", cand) not in exclusions:
            return cand
    return sur


def check_no_eval_person_surfaces(docs: Sequence[dict], exclusions: set[str]) -> list[str]:
    """生成した PERSON 表層（半角カナは NFKC でも）が評価セットの表層に当たっていないか。"""
    bad = []
    for d in docs:
        for s, e, l in d["entities"]:
            surf = d["text"][s:e]
            if l == "PERSON" and (surf in exclusions or unicodedata.normalize("NFKC", surf) in exclusions):
                bad.append(surf)
    return bad


# ---------------------------------------------------------------- LLM 負例

def load_negatives(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("entities"):
                raise SystemExit(f"負例ファイルにエンティティ付きの文書があります: {path}")
            rows.append({"text": r["text"], "entities": [], "source": r.get("source", LLM_NEGATIVE_SOURCE),
                         "variant": r.get("variant", "")})
    return rows


def split_negatives(rows: Sequence[dict], rng: random.Random, ratio: float = NEG_VALID_RATIO) -> tuple[list[dict], list[dict]]:
    xs = list(rows)
    rng.shuffle(xs)
    k = int(len(xs) * ratio)
    return xs[k:], xs[:k]


def without_llm_negatives(records: Sequence[dict]) -> list[dict]:
    return [r for r in records if r["source"] != LLM_NEGATIVE_SOURCE]


# ---------------------------------------------------------------- 全体

def build(records: list[dict], *, cap: int, seed: int, n_chat: int, n_dialogue: int, n_halfwidth: int,
          n_forms: int, negatives: Sequence[dict], eval_person: set[str]) -> tuple[list[dict], list[dict], dict]:
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
    sur_tr, sur_va = surname_bank(eval_person, rng)
    given = list(dict.fromkeys(GIVEN_KANJI + GIVEN_KANJI_F))
    nv = lambda n: max(1, int(n * NEW_DOC_VALID_RATIO))  # noqa: E731
    new_train = (make_chat_docs(n_chat - nv(n_chat), rng, utt_tr, latin_given, latin_sur)
                 + make_dialogue_docs(n_dialogue - nv(n_dialogue), rng, fil_tr, utt_tr)
                 + make_halfwidth_docs(n_halfwidth - nv(n_halfwidth), rng, biz_tr)
                 + make_form_docs(n_forms - nv(n_forms), rng, sur_tr, given, exclusions=eval_person))
    new_valid = (make_chat_docs(nv(n_chat), rng, utt_va, latin_given, latin_sur)
                 + make_dialogue_docs(nv(n_dialogue), rng, fil_va, utt_va)
                 + make_halfwidth_docs(nv(n_halfwidth), rng, biz_va)
                 + make_form_docs(nv(n_forms), rng, sur_va, given, exclusions=eval_person))
    bad = check_no_eval_surfaces(new_train + new_valid)
    if bad:
        raise SystemExit(f"生成物に盲検セットの表層が混ざっています: {sorted(set(bad))}")
    bad = check_no_eval_person_surfaces([d for d in new_train + new_valid if d["source"] == FORM_SOURCE], eval_person)
    if bad:
        raise SystemExit(f"帳票フレームに評価セットの PERSON 表層が混ざっています: {sorted(set(bad))[:10]}")
    new_leak = len({d["text"] for d in new_train} & {d["text"] for d in new_valid})
    train += new_train
    valid += new_valid

    long_train = make_long_docs(train, target_ratio=LONG_TARGET_RATIO, rng=rng)
    long_valid = make_long_docs(valid, target_ratio=LONG_TARGET_RATIO, rng=rng)
    train += long_train
    valid += long_valid

    neg_train, neg_valid = split_negatives(negatives, rng)
    train += neg_train
    valid += neg_valid
    rng.shuffle(train)
    rng.shuffle(valid)
    form_docs = [d for d in new_train + new_valid if d["source"] == FORM_SOURCE]
    info = {"seed": seed, "cap": cap, "relabeled": relabeled, "dropped": dict(dropped),
            "addr_variants": len(addr_aug), "person_variants": len(person_aug),
            "person_made": person_made, "long_train": len(long_train), "long_valid": len(long_valid),
            "leak": leak, "before": before,
            "new_docs": {"chat": n_chat, "dialogue": n_dialogue, "halfwidth": n_halfwidth, "forms": n_forms,
                         "train": len(new_train), "valid": len(new_valid), "text_overlap": new_leak},
            "forms": {"docs": len(form_docs), "person": sum(1 for d in form_docs for *_, l in d["entities"] if l == "PERSON"),
                      "pure_negative": sum(1 for d in form_docs if not d["entities"]),
                      "surname_bank_train": len(sur_tr), "surname_bank_valid": len(sur_va),
                      "eval_person_exclusions": len(eval_person)},
            "llm_negatives": {"total": len(negatives), "train": len(neg_train), "valid": len(neg_valid)}}
    return train, valid, info


def v8_appendix(info: dict, train: Sequence[dict], valid: Sequence[dict]) -> str:
    def src_counts(recs: Sequence[dict]) -> Counter:
        return Counter(r["source"] for r in recs)

    def neg_counts(recs: Sequence[dict]) -> Counter:
        return Counter(r["source"] for r in recs if not r["entities"])

    tr, va = src_counts(train), src_counts(valid)
    ntr, nva = neg_counts(train), neg_counts(valid)
    f, ln = info["forms"], info["llm_negatives"]
    L = ["", "## v8 の追加分", "",
         f"- テンプレート上限 {info['cap']}（v7 は 8）。落とした文書: {sum(info['dropped'].values())}",
         f"- 新規文書: チャット {info['new_docs']['chat']} / 会話 {info['new_docs']['dialogue']} / 半角カナ {info['new_docs']['halfwidth']}"
         f" / **帳票・ログ {info['new_docs']['forms']}**（train {info['new_docs']['train']} / valid {info['new_docs']['valid']}、本文一致 {info['new_docs']['text_overlap']}）",
         f"- 帳票・ログ: PERSON {f['person']} 件、名なしの純負例 {f['pure_negative']} 文書、姓の候補 train {f['surname_bank_train']} / valid {f['surname_bank_valid']}"
         f"（評価セットの PERSON 表層 {f['eval_person_exclusions']} 種を除外）",
         f"- LLM 負例（{LLM_NEGATIVE_SOURCE}）: {ln['total']} 件 → train {ln['train']} / valid {ln['valid']}。`train_v8_noneg.jsonl` はこれを除いた版",
         "", "| 出所 | train | valid | うち負例 train | うち負例 valid |", "|---|---:|---:|---:|---:|"]
    for s in sorted(set(tr) | set(va)):
        L.append(f"| {s} | {tr.get(s, 0)} | {va.get(s, 0)} | {ntr.get(s, 0)} | {nva.get(s, 0)} |")
    n_tr = len(train)
    L += ["", f"- train の負例 {sum(ntr.values())} / {n_tr}（{sum(ntr.values()) / max(n_tr, 1) * 100:.1f}%）、"
          f"LLM 負例を除くと {sum(ntr.values()) - ntr.get(LLM_NEGATIVE_SOURCE, 0)} / {n_tr - tr.get(LLM_NEGATIVE_SOURCE, 0)}",
          f"- train の実テキスト out_ner {tr.get('out_ner', 0)} / {n_tr}（{tr.get('out_ner', 0) / max(n_tr, 1) * 100:.1f}%）、"
          f"LLM 負例を除く母数では {tr.get('out_ner', 0) / max(n_tr - tr.get(LLM_NEGATIVE_SOURCE, 0), 1) * 100:.1f}%"]
    return "\n".join(L) + "\n"


def collect_eval_person_surfaces(extra_paths: Sequence[str] = ()) -> set[str]:
    """固定OOD・GSD・OpenPII・盲検 v1 の gold PERSON 表層。読めないセットは警告して飛ばす。"""
    surfaces: set[str] = set()
    try:
        import eval_ood_f1 as f1
        sets = f1.load_sets([f"extra{i}={p}" for i, p in enumerate(extra_paths)])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 評価セットを読めません（{exc}）。盲検の表層だけで除外します")
        sets = {}
    for name, (docs, _) in sets.items():
        n = 0
        for d in docs:
            for ent in d.get("entities", []):
                s, e, lab = (ent[0], ent[1], ent[2]) if isinstance(ent, (list, tuple)) else (ent["start"], ent["end"], ent.get("label"))
                if lab == "PERSON":
                    surfaces.add(d["text"][s:e])
                    n += 1
        print(f"  評価セット {name}: PERSON {n} 件")
    surfaces |= set(EVAL_SURFACES)
    return surfaces


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
    ap.add_argument("--negatives", default=None, help="既定: <Drive>/negatives_v8/filtered_v1/negatives_v8.jsonl")
    args = ap.parse_args(argv)
    if args.cap < 1:
        raise SystemExit("--cap は 1 以上")

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書")
    neg_path = args.negatives or os.path.join(gd, "negatives_v8", "filtered_v1", "negatives_v8.jsonl")
    negatives = load_negatives(neg_path)
    print(f"LLM 負例: {len(negatives)} 文書 ({neg_path})")
    eval_person = eval_person_exclusions(collect_eval_person_surfaces())
    print(f"帳票フレームから除く PERSON 表層: {len(eval_person)} 種")

    train, valid, info = build(records, cap=args.cap, seed=args.seed, n_chat=args.chat, n_dialogue=args.dialogue,
                               n_halfwidth=args.halfwidth, n_forms=args.forms, negatives=negatives, eval_person=eval_person)
    if info["leak"]:
        raise SystemExit(f"train / valid にテンプレート重複が {info['leak']} 件あります。生成を中止")

    os.makedirs(args.out_dir, exist_ok=True)
    train_noneg = without_llm_negatives(train)
    write_jsonl(os.path.join(args.out_dir, "train_v8.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "train_v8_noneg.jsonl"), train_noneg)
    write_jsonl(os.path.join(args.out_dir, "valid_v8.jsonl"), valid)
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
        "files": {"train": "train_v8.jsonl", "train_noneg": "train_v8_noneg.jsonl", "valid": "valid_v8.jsonl"},
    }
    with open(os.path.join(args.out_dir, "manifest_v8.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    card = data_card("学習データ v8", info["before"], st_all, st_train, st_valid, info, version="v8")
    # v6 のカードにある「LLM 負例は入れていない」は v8 では逆になる
    card = card.replace(
        "- LLM 生成の新規負例（CPU では作れない。`qwen_synth_pii_data` 系ノートブックで別途）",
        "- 評価セット由来の人名表層（帳票フレームの姓候補からも機械的に除外）")
    card += v8_appendix(info, train, valid)
    with open(os.path.join(args.out_dir, "DATA_CARD_v8.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

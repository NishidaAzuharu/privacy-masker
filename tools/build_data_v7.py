"""学習データ v7 — v6 の方針を保ったまま、盲検 v1 で落ちた 3 つの形を足す。

v6（`build_data_v6.py`）で 0 から学習した結果（開発記録、非公開）:
OpenPII の危険 99 → 8、住所書式の危険 191 → 2 と書式の穴は埋まったが、**盲検 v1（実テキスト）で
危険 52 → 60 と悪化**した。新たに被覆ゼロになった PERSON 14 件の内訳:

| 形 | 件数 | 例（盲検 v1。学習データには入れない） |
|---|---:|---|
| チャット・取引メッセージの行頭ハンドル名 | 9 | `<ハンドル名>: …` `<平仮名の名> 12:03`（表層は非公開） |
| 裸の 1〜2 字の名（会話の呼びかけ・物語文） | 3 | 「<1 字名>ちゃんを…」「<1 字名>の…」「<肩書き><2 字カナ名>の…」 |
| 半角カナの名 | 1 | `出荷担当 <半角カナの姓>` |
| 会話中の平仮名名 | 1 | `<平仮名の名>も行くなら` |

v45e にはこれらの形が（テンプレート反復の副作用で）多く含まれ、v6 のテンプレート上限で
正例が 44% 減ったことで感度が落ちた。v7 は上限を 4 → 8 に緩めて正例を戻しつつ、
上の 3 形を**新しい文書として**合成する（既存テンプレートの書き換えではなく、チャットログ・
会話文・業務文書のフレームから作る）。

新規文書の発話・文のバンクは train 用と valid 用に分けるので、v7 valid はこれらの形の
未知の文で評価できる。フレーム（`name: 本文` などの形式）は両側で共有する — 形式そのものを
学べたかを測るのが目的だから。

盲検 v1 の表層（ハンドル名・人名）は一切使わない（`EVAL_SURFACES` で機械的に確認する）。

使い方::

    PYTHONPATH=. .venv/bin/python tools/build_data_v7.py --out-dir <dir> [--cap 8] [--seed 42]
        [--chat 1500] [--dialogue 1500] [--halfwidth 500]
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
from typing import Sequence

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
from augment_address_formats import augment as augment_addresses  # noqa: E402
from build_data_v6 import (  # noqa: E402
    GIVEN_HIRA,
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
from split_by_template import (  # noqa: E402
    BASE_LABELS,
    DATA_FILES,
    grouped_split,
    load_records,
    template_signature,
)
from privacy_masker.data.ja_pools import (  # noqa: E402
    GIVEN_KANJI, GIVEN_KATA, SURNAMES_KANJI, SURNAMES_KATA,
)

SEED = 42
DEFAULT_CAP = 8
VALID_BANK_RATIO = 0.15      # 発話・文のバンクのうち valid 用に取り分ける割合
NEW_DOC_VALID_RATIO = 0.10   # 新規文書のうち valid に入れる割合

# 盲検 v1 で見た識別子的な表層（ハンドル名・珍しい半角カナ姓）と、チャットのハンドル名として出た平仮名の名。
# 生成物に混ざっていないことを機械的に確かめる（評価セットへの漏れ防止）。表層そのものはリポジトリに置かず、
# データフォルダの blind_eval_v1/eval_surfaces.json（{"surfaces": [...], "handle_kana": [...]}）から読む。
# ファイルが無ければ空集合（検査は素通り。main は警告する）。一般的な実名（ja_pools にも元からある姓名）は入れない
EVAL_SURFACES_FILE = os.path.join(drive_dir(require=False), "blind_eval_v1", "eval_surfaces.json")


def load_eval_surfaces(path: str = EVAL_SURFACES_FILE) -> tuple[frozenset[str], frozenset[str]]:
    """(識別子的な表層, ハンドル名として出た平仮名の名)。ファイルが無ければ両方とも空。"""
    if not os.path.exists(path):
        return frozenset(), frozenset()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return frozenset(d.get("surfaces", ())), frozenset(d.get("handle_kana", ()))


EVAL_SURFACES, EVAL_HANDLE_KANA = load_eval_surfaces()

# ---------------------------------------------------------------- 部品

# 1 文字の名（会話の呼びかけ「<名>ちゃん」型）。一般的なもので個人を指さない。盲検に出た 1 字名は入れない
GIVEN_ONE_KANJI: tuple[str, ...] = (
    "碧", "蓮", "凛", "翔", "湊", "陸", "葵", "楓", "光", "誠", "実", "舞", "愛", "遥", "樹", "空",
    "響", "澪", "律", "燦", "薫", "浩", "潤", "静", "純", "梓", "桜", "菫", "萌", "蛍",
)
# 2 文字のカタカナ名（あだ名・外国名）。盲検に出た短いカタカナ名は入れない
GIVEN_KATA_SHORT: tuple[str, ...] = (
    "リン", "ケン", "ユウ", "マキ", "アキ", "ジュン", "レイ", "カイ", "トム", "ボブ", "サム", "ジル",
    "ルナ", "ノア", "エマ", "リサ", "ナナ", "ミオ", "ヒロ", "タク", "ショウ", "ダイ", "リョウ", "コウ",
)
# 日本の姓のカタカナ表記（半角カナ用）。盲検に出た姓は入れない
SURNAMES_KATA_JA: tuple[str, ...] = (
    "タナカ", "サトウ", "スズキ", "タカハシ", "ワタナベ", "イトウ", "ヤマモト", "ナカムラ", "コバヤシ", "カトウ",
    "ヨシダ", "ヤマダ", "ササキ", "ヤマグチ", "マツモト", "イノウエ", "キムラ", "ハヤシ", "シミズ", "ヤマザキ",
    "モリ", "イケダ", "ハシモト", "アベ", "イシカワ", "ヤマシタ", "ナカジマ", "イシイ", "オガワ", "マエダ",
    "フジタ", "オカダ", "ゴトウ", "ハセガワ", "ムラカミ", "コンドウ", "イシダ", "サカモト", "エンドウ", "アオキ",
    "フジイ", "ニシムラ", "フクダ", "オオタ", "ミウラ", "フジワラ", "オカモト", "マツダ", "ナカガワ", "ナカノ",
)
GIVEN_KATA_JA: tuple[str, ...] = (
    "タロウ", "ハナコ", "ケンタ", "ユウキ", "サクラ", "ヒロシ", "アヤカ", "ダイスケ", "ミサキ", "リョウタ",
    "カズキ", "ユイ", "ショウタ", "ハルカ", "タクミ", "アオイ", "ソウタ", "ナオキ", "ミオ", "ユウト",
)
CHAT_ROLES: tuple[str, ...] = ("出品者", "購入者", "司会", "担当", "【キャラA】", "【キャラB】", "運営", "幹事")

# チャット・取引メッセージの発話。個人情報を含まない一般文
CHAT_UTTERANCES: tuple[str, ...] = (
    "はじめまして。購入させていただきました。よろしくお願いします！",
    "ご購入ありがとうございます。明日コンビニから匿名配送で発送します。",
    "助かります！ 受け取ったら評価しますね。",
    "浦安方面なら翌日には届くと思います。",
    "了解です。サイズは M で大丈夫でしたか？",
    "土曜の午後なら空いてます。何時にしますか？",
    "駅から近いし、雨でも行きやすいと思う。",
    "いいね！ じゃあ 13 時に改札前で。",
    "平日昼で 30 分待ちくらい。夕方がおすすめ。",
    "資料は共有フォルダに置いておきました。",
    "確認しました。この方向で進めましょう。",
    "先方の返答待ちです。届いたら共有します。",
    "参加者名を備考に書いてください。",
    "はい、で合ってますか？",
    "遅れてすみません、今向かってます。",
    "今日のプレゼン、時間配分が難しいと思うんだけど？",
    "写真追加しました。傷は写真の 3 枚目です。",
    "値下げは難しいですが、送料込みにします。",
    "お取り置きは 3 日までなら可能です。",
    "同じものをもう 1 点探しています。入荷予定はありますか？",
    "会議室は 2 階の奥、ホワイトボードがある方です。",
    "議事録は今日中に回します。",
    "その件、来週の定例で話しましょう。",
    "リンク先が開けません。権限を付けてもらえますか？",
    "テスト環境で再現しました。ログを添付します。",
    "明日の朝までに直せそうです。",
    "了解、こちらでレビューします。",
    "ありがとう、確認してマージします。",
    "健康チェック OK、監視は正常です。",
    "デプロイ完了。エラーは出ていません。",
    "行きたかったやつ！ 混んでた？",
    "誕生日プレゼントなので、9/20 までに届くと嬉しいです。",
    "当日は動きやすい服装でお願いします。",
    "集合場所を変更します。詳細は後で送ります。",
    "先に注文しておくね。何がいい？",
    "写真ありがとう、いい感じ！",
    "お疲れさまでした。次回もよろしくお願いします。",
    "在庫を確認して折り返します。",
    "本日発送しました。到着までお待ちください。",
    "評価ありがとうございました。またご縁があれば。",
    "その日は予定があって難しいです。別日でどうでしょう。",
    "見積もりを出し直しました。ご確認ください。",
    "承知しました。担当に伝えます。",
    "この間の件、どうなりましたか？",
    "無事に受け取りました。ありがとうございます。",
    "了解です。では 15 時に。",
    "今週中に一度すり合わせたいです。",
    "画面共有できますか？ 音声が少し途切れます。",
    "初参加です。よろしくお願いします。",
    "ここのラーメン、おすすめです。",
    "昨日の続き、どこまで進みました？",
    "はい、問題ありません。",
    "確認できたら教えてください。",
    "急ぎではないので、手が空いたときで大丈夫です。",
    "同じ症状が出ています。原因を調べます。",
    "スクショ貼ります。",
    "次の便で送ります。",
    "了解しました。ありがとうございます。",
    "ちょっと席を外します。戻ったら返します。",
    "明日は在宅です。連絡はチャットでお願いします。",
)
# 会話・物語文のフレーム。{n} が名、{u} が文。敬称・助詞は span に含めない
DIALOGUE_FRAMES: tuple[str, ...] = (
    "「{n}ちゃん、{u}」",
    "「{n}さん、{u}」と彼は言った。",
    "{n}は{u}とだけ言って、席を立った。",
    "そのことを、後から{n}に話した。",
    "{n}の恐るべき食欲。一晩で三人分を平らげた。",
    "{n}くんは黙ってうなずいた。",
    "「{n}を{n2}の後つぎにするという話は、本当ですか」",
    "{n}先生が教室に入ってきた。",
    "{n}氏はこう述べている。「{u}」",
    "彼女は{n}を見て、それから窓の外を見た。",
    "「{n}、{u}」",
    "{n}と{n2}は顔を見合わせた。",
    "母は{n}のことを話すと、いつも少し笑った。",
    "{n}様、{u}",
    "{n}も行くなら夕方がおすすめ。",
    "映画は{n}の『冬の旅』が好きだ。",
    "{n}の言葉は、しばらく途切れた。",
    "その夜、{n}は遅くまで起きていた。",
    "「{n}か。久しぶりだな」",
    "{n}に頼まれて、荷物を運んだ。",
)
NARRATIVE_FILLERS: tuple[str, ...] = (
    "雨は夕方まで降り続いた。", "駅前の通りはもう暗かった。", "誰も答えなかった。", "時計の音だけが聞こえた。",
    "窓の外では風が鳴っていた。", "それきり、話は途切れた。", "翌朝、町は静かだった。", "電車は少し遅れて着いた。",
    "台所から湯気が上がっていた。", "遠くで犬が鳴いた。", "会議は予定より長引いた。", "紙の束が机の端で崩れた。",
    "廊下の灯りが一つ消えていた。", "秋の匂いがした。", "店はまだ開いていなかった。", "彼は帽子を取った。",
)
# 業務文書のフレーム（半角カナ名）。{k} が半角カナの名
HALFWIDTH_FRAMES: tuple[str, ...] = (
    "出荷担当 {k}", "担当: {k}", "受付 {k}", "作成者 {k}", "ﾁｴｯｶｰ {k}", "確認 {k}　承認 {k2}",
    "問合せ先 {k}（内線 {d}）", "納品書　担当 {k}", "検品 {k}　梱包 {k2}", "ドライバー {k}",
    "受注 No.{d}　担当 {k}", "配送伝票　受取人 {k}", "現場責任者 {k}", "見積担当 {k}",
    "{k} 様", "宛名 {k}", "発注者 {k}　受注者 {k2}",
)
BUSINESS_LINES: tuple[str, ...] = (
    "品名　ステンレス製 配管継手（呼び径 50A）", "数量　120 個", "納期　受注後 10 営業日", "支払条件　月末締め翌月末払い",
    "備考　検査成績書を添付", "単価　別紙見積のとおり", "運送　宅配便（元払い）", "保証期間　納入後 1 年",
    "梱包　段ボール 6 箱", "受入時間　9:00〜17:00", "注記　図面改訂 B 適用", "工場出荷予定　来週水曜",
    "検収　納入後 5 営業日以内", "返品条件　未開封に限る", "品番　KT-2200", "ロット　R-0918",
)

# 半角カナ（U+FF66〜U+FF9D）から NFKC で全角を引き、逆表を作る。手打ちの文字列は
# 全角の「セ」が混ざる事故が起きたので、コードポイントの範囲から機械的に作る
_HW_TABLE: dict[str, str] = {}
for _cp in range(0xFF66, 0xFF9E):
    _ch = chr(_cp)
    _fw = unicodedata.normalize("NFKC", _ch)
    if len(_fw) == 1 and _fw != _ch:
        _HW_TABLE[_fw] = _ch
for _fw in "ガギグゲゴザジズゼゾダヂヅデドバビブベボヴ":
    _HW_TABLE[_fw] = _HW_TABLE[unicodedata.normalize("NFD", _fw)[0]] + "ﾞ"
for _fw in "パピプペポ":
    _HW_TABLE[_fw] = _HW_TABLE[unicodedata.normalize("NFD", _fw)[0]] + "ﾟ"
_HW_TABLE["・"] = "･"


# 平仮名の名。盲検 v1 にチャットのハンドル名として出た名（EVAL_HANDLE_KANA）は外す
KANA_NAMES: tuple[str, ...] = tuple(h for h in GIVEN_HIRA if h not in EVAL_HANDLE_KANA)


def to_halfwidth_kana(s: str) -> str:
    """全角カタカナを半角に。NFKC で元に戻る（往復一致を tests で確認）。"""
    return "".join(_HW_TABLE.get(c, c) for c in s)


class _Doc:
    """本文を継ぎ足しながら span を記録する。"""

    def __init__(self, source: str, variant: str) -> None:
        self.parts: list[str] = []
        self.n = 0
        self.entities: list[list] = []
        self.source = source
        self.variant = variant

    def text(self, s: str) -> None:
        self.parts.append(s)
        self.n += len(s)

    def ent(self, s: str, label: str) -> None:
        self.entities.append([self.n, self.n + len(s), label])
        self.text(s)

    def build(self) -> dict:
        t = "".join(self.parts)
        for s, e, l in self.entities:
            assert 0 <= s < e <= len(t), (s, e, len(t))
        return {"text": t, "entities": self.entities, "source": self.source, "variant": self.variant}


# ---------------------------------------------------------------- 生成器

def _handle_pool(rng: random.Random, latin_given: Sequence[str], latin_sur: Sequence[str]) -> list[tuple[str, str]]:
    """(表層, 種類) のハンドル名。英字は小文字化した英字名の組み合わせ、かなは名、混在は かな@出品 型。"""
    lg = [g.lower() for g in latin_given if re.fullmatch(r"[A-Za-z]+", g)] or ["maria", "john", "ken"]
    ls = [s.lower() for s in latin_sur if re.fullmatch(r"[A-Za-z]+", s)] or ["ivers", "smith", "sato"]
    out: list[tuple[str, str]] = []
    for _ in range(6):
        g, s = rng.choice(lg), rng.choice(ls)
        form = rng.choice(("{g}_{s}", "{s}_{g}", "{g}.{s}", "{g}{d}", "{s}{d}", "{g}_{d}", "{g}-{s}", "{s}", "{g}"))
        out.append((form.format(g=g, s=s, d=rng.randint(1, 99)), "latin"))
    for _ in range(3):
        out.append((rng.choice(KANA_NAMES), "kana"))
    for _ in range(2):
        out.append((rng.choice(SURNAMES_KANJI), "kanji"))
    for _ in range(2):
        out.append((f"{rng.choice(KANA_NAMES)}@{rng.choice(('出品', '店', 'shop', '事務局'))}", "mixed"))
    out.append((f"{rng.choice(lg)}@{rng.choice(('shop', 'work', 'home'))}", "mixed"))
    rng.shuffle(out)
    return out


def make_chat_docs(n: int, rng: random.Random, utterances: Sequence[str],
                   latin_given: Sequence[str], latin_sur: Sequence[str]) -> list[dict]:
    """チャット・取引メッセージのログ。行頭ハンドル名（`name: 本文`）を PERSON として注釈する。"""
    docs = []
    for _ in range(n):
        pool = _handle_pool(rng, latin_given, latin_sur)
        speakers = rng.sample(pool, k=rng.choice((2, 2, 3)))
        style = rng.choice(("colon", "colon", "bracket", "time", "role", "arrow"))
        d = _Doc("synth_chat", f"chat:{style}")
        header = rng.choice(("", "", "取引メッセージ\n", "---\n", "本日のログ\n", "#general\n"))
        d.text(header)
        for li in range(rng.randint(3, 8)):
            h, _kind = rng.choice(speakers)
            u = rng.choice(utterances)
            if style == "colon":
                d.ent(h, "PERSON"); d.text(": " + u)
            elif style == "bracket":
                d.text("["); d.ent(h, "PERSON"); d.text("] " + u)
            elif style == "time":
                d.ent(h, "PERSON"); d.text(f" {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}\n" + u)
            elif style == "role":
                d.ent(h, "PERSON"); d.text(f" ({rng.choice(CHAT_ROLES)}): 「{u}」")
            else:
                d.ent(h, "PERSON"); d.text("> " + u)
            # 発話の中で別の話者に @ で呼びかける（@ は span に含めない）
            if rng.random() < 0.25:
                other, _ = rng.choice(speakers)
                d.text(" @"); d.ent(other, "PERSON"); d.text(" " + rng.choice(("確認お願いします", "どう思う？", "見てもらえますか", "了解です")))
            d.text("\n")
        docs.append(d.build())
    return docs


def make_dialogue_docs(n: int, rng: random.Random, fillers: Sequence[str],
                       utterances: Sequence[str]) -> list[dict]:
    """会話・物語文。裸の 1〜2 字の名（漢字・カタカナ・平仮名）を PERSON、敬称・助詞は span 外。"""
    names_short = (list(GIVEN_ONE_KANJI) + [s for s in SURNAMES_KANJI if len(s) == 2]
                   + [g for g in GIVEN_KANJI if len(g) <= 2] + list(GIVEN_KATA_SHORT)
                   + [h for h in KANA_NAMES if len(h) <= 3])
    docs = []
    for _ in range(n):
        d = _Doc("synth_dialogue", "dialogue")
        cast = rng.sample(names_short, 2)
        for si in range(rng.randint(2, 5)):
            if rng.random() < 0.35:
                d.text(rng.choice(fillers) + "\n")
                continue
            frame = rng.choice(DIALOGUE_FRAMES)
            u = rng.choice(utterances).rstrip("！。!")
            # フレームを {n} {n2} {u} で分割して span を付けながら組む
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
        docs.append(d.build())
    return docs


def make_halfwidth_docs(n: int, rng: random.Random, lines: Sequence[str]) -> list[dict]:
    """業務文書（納品書・伝票・工程表）。半角カナの名を PERSON として注釈する。"""
    docs = []
    for _ in range(n):
        d = _Doc("synth_halfwidth", "halfwidth")
        k1 = to_halfwidth_kana(rng.choice(SURNAMES_KATA_JA))
        k2 = to_halfwidth_kana(rng.choice(SURNAMES_KATA_JA))
        if rng.random() < 0.3:      # 姓 名 の形も混ぜる
            k1 = k1 + " " + to_halfwidth_kana(rng.choice(GIVEN_KATA_JA))
        if rng.random() < 0.15:     # 外国名の半角カナ
            k2 = to_halfwidth_kana(rng.choice(GIVEN_KATA) + "･" + rng.choice(SURNAMES_KATA))
        body = rng.sample(list(lines), k=min(len(lines), rng.randint(2, 5)))   # valid 用バンクは小さい
        frames = rng.sample(HALFWIDTH_FRAMES, k=rng.randint(1, 3))
        order = body + frames
        rng.shuffle(order)
        for item in order:
            if "{k" in item:
                for tok in re.split(r"(\{k2?\}|\{d\})", item):
                    if tok == "{k}":
                        d.ent(k1, "PERSON")
                    elif tok == "{k2}":
                        d.ent(k2, "PERSON")
                    elif tok == "{d}":
                        d.text(str(rng.randint(100, 9999)))
                    elif tok:
                        d.text(tok)
            else:
                d.text(item)
            d.text("\n")
        docs.append(d.build())
    return docs


def _split_bank(items: Sequence[str], rng: random.Random, ratio: float) -> tuple[list[str], list[str]]:
    xs = list(items)
    rng.shuffle(xs)
    k = max(1, int(len(xs) * ratio))
    return xs[k:], xs[:k]


def check_no_eval_surfaces(docs: Sequence[dict]) -> list[str]:
    """生成した PERSON 表層に盲検 v1 の表層が混ざっていないか。混ざっていれば返す。"""
    bad = []
    for d in docs:
        for s, e, l in d["entities"]:
            if l == "PERSON" and d["text"][s:e] in EVAL_SURFACES:
                bad.append(d["text"][s:e])
    return bad


# ---------------------------------------------------------------- 全体

def build(records: list[dict], *, cap: int, seed: int, n_chat: int, n_dialogue: int,
          n_halfwidth: int) -> tuple[list[dict], list[dict], dict]:
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

    # 新規文書: 発話・文のバンクを先に train / valid に分け、valid 側は valid 用バンクだけで作る
    utt_tr, utt_va = _split_bank(CHAT_UTTERANCES, rng, VALID_BANK_RATIO)
    fil_tr, fil_va = _split_bank(NARRATIVE_FILLERS, rng, VALID_BANK_RATIO)
    biz_tr, biz_va = _split_bank(BUSINESS_LINES, rng, VALID_BANK_RATIO)
    nv = lambda n: max(1, int(n * NEW_DOC_VALID_RATIO))  # noqa: E731
    new_train = (make_chat_docs(n_chat - nv(n_chat), rng, utt_tr, latin_given, latin_sur)
                 + make_dialogue_docs(n_dialogue - nv(n_dialogue), rng, fil_tr, utt_tr)
                 + make_halfwidth_docs(n_halfwidth - nv(n_halfwidth), rng, biz_tr))
    new_valid = (make_chat_docs(nv(n_chat), rng, utt_va, latin_given, latin_sur)
                 + make_dialogue_docs(nv(n_dialogue), rng, fil_va, utt_va)
                 + make_halfwidth_docs(nv(n_halfwidth), rng, biz_va))
    bad = check_no_eval_surfaces(new_train + new_valid)
    if bad:
        raise SystemExit(f"生成物に評価セットの表層が混ざっています: {sorted(set(bad))}")
    # 新規文書の本文が両側で一致していないこと（バンクを分けた効果の確認）
    new_leak = len({d["text"] for d in new_train} & {d["text"] for d in new_valid})
    train += new_train
    valid += new_valid

    long_train = make_long_docs(train, target_ratio=LONG_TARGET_RATIO, rng=rng)
    long_valid = make_long_docs(valid, target_ratio=LONG_TARGET_RATIO, rng=rng)
    train += long_train
    valid += long_valid
    rng.shuffle(train)
    rng.shuffle(valid)
    info = {"seed": seed, "cap": cap, "relabeled": relabeled, "dropped": dict(dropped),
            "addr_variants": len(addr_aug), "person_variants": len(person_aug),
            "person_made": person_made, "long_train": len(long_train), "long_valid": len(long_valid),
            "leak": leak, "before": before,
            "new_docs": {"chat": n_chat, "dialogue": n_dialogue, "halfwidth": n_halfwidth,
                         "train": len(new_train), "valid": len(new_valid), "text_overlap": new_leak}}
    return train, valid, info


def v7_appendix(info: dict, train: Sequence[dict], valid: Sequence[dict]) -> str:
    def cnt(docs, src):
        n = sum(1 for d in docs if d["source"] == src)
        p = sum(1 for d in docs if d["source"] == src for *_, l in d["entities"] if l == "PERSON")
        return f"{n} 文書 / PERSON {p}"
    nd = info["new_docs"]
    return "\n".join([
        "", "## v7 で足した文書（盲検 v1 で落ちた形）", "",
        "| 種類 | train | valid |", "|---|---|---|",
        f"| チャット・取引ログ（行頭ハンドル名） | {cnt(train, 'synth_chat')} | {cnt(valid, 'synth_chat')} |",
        f"| 会話・物語文（裸の 1〜2 字の名） | {cnt(train, 'synth_dialogue')} | {cnt(valid, 'synth_dialogue')} |",
        f"| 業務文書（半角カナ名） | {cnt(train, 'synth_halfwidth')} | {cnt(valid, 'synth_halfwidth')} |",
        "",
        f"発話・文のバンクは valid 用に {int(VALID_BANK_RATIO * 100)}% を取り分け、valid の新規文書はそれだけで作った。",
        f"train / valid で本文が一致する新規文書: {nd['text_overlap']} 件（0 が期待値）。",
        "盲検 v1 の表層（`EVAL_SURFACES`）は生成物に含まれないことを機械的に確認している。",
        "",
    ])


def main(argv: Sequence[str] | None = None) -> int:
    if not EVAL_SURFACES:
        print(f"[warn] 盲検表層ファイルが無いので表層の漏れ検査は素通りです: {EVAL_SURFACES_FILE}", file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="loose 署名 1 つにつき残す最大文書数（既定 8）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chat", type=int, default=1500)
    ap.add_argument("--dialogue", type=int, default=1500)
    ap.add_argument("--halfwidth", type=int, default=500)
    args = ap.parse_args(argv)
    if args.cap < 1:
        raise SystemExit("--cap は 1 以上")

    gd = args.drive_dir or drive_dir()
    records = load_records(gd)
    print(f"読み込み: {len(records)} 文書")
    train, valid, info = build(records, cap=args.cap, seed=args.seed, n_chat=args.chat,
                               n_dialogue=args.dialogue, n_halfwidth=args.halfwidth)
    if info["leak"]:
        raise SystemExit(f"train / valid にテンプレート重複が {info['leak']} 件あります。生成を中止")

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train_v7.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "valid_v7.jsonl"), valid)
    st_train, st_valid, st_all = stats(train), stats(valid), stats(train + valid)
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [f for _, f in DATA_FILES], "seed": args.seed, "cap": args.cap,
        "labels": list(BASE_LABELS),
        "train": {"docs": len(train), "text_digest": _digest(train), **st_train},
        "valid": {"docs": len(valid), "text_digest": _digest(valid), **st_valid},
        "processing": {k: v for k, v in info.items() if k not in ("before", "person_made")},
        "person_variants_made": dict(info["person_made"]),
    }
    with open(os.path.join(args.out_dir, "manifest_v7.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    card = (data_card("学習データ v7", info["before"], st_all, st_train, st_valid, info, version="v7")
            + v7_appendix(info, train, valid))
    with open(os.path.join(args.out_dir, "DATA_CARD_v7.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"書き出し: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

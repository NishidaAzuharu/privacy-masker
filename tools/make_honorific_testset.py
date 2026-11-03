"""敬称境界テストセット（合成・架空の名前）を JSONL で生成する。

背景: 文字ヘッド NER は「田中さん」が語彙で 1 ピースになるとき PERSON を『田中さ』（1 文字ずれ）と出すことが
ある。このセットは「名前 + 敬称」の境界だけを狙って作る。gold は名前だけ（敬称はスパン外）で、文型に含まれる
地名（大阪府堺市 → LOCATION）も gold に入れる。評価は tools/eval_boundary_errors.py で行う。

構成:
  姓 60（語彙で『姓+さん』が 1 ピースになる 田中・佐藤・鈴木 と、分かれる 山田・高橋・伊藤・渡辺・中村・小林・加藤 など、
  まれな姓 12）× 名 20 × 敬称 6（さん/様/氏/さま/くん/ちゃん）× 文型 10 × 名前形 2（姓のみ / 姓名）。
  既定は全組合せから (敬称, 文型, 名前形) の 120 層で層化サンプリングして 2,000 件程度。姓は seed 付きの巡回で
  各層に割り当てる。『語 + 敬称』がどれかの敬称で 1 ピースになる姓・名（--model の tokenizer で判定。30m では
  田中・佐藤・鈴木・原・谷 … と 太郎・一郎・恵 …）は --boost-merged 倍（既定 3）の重みで出す。境界が崩れる側の
  事例を十分に含めるためで、--boost-merged 1 で一様になる。

各行: {"id", "text", "entities": [[start, end, label], ...], "neutral": [], "meta": {...}}
  meta.surname / given（姓のみなら null）/ honorific / template_id / name_form（"surname" | "full"）
  meta.merged: 決定的トークン分割（--model の tokenizer）で、名前の末尾文字を含むピースが敬称まで伸びるか
  2 人目の名前がある文型（T08）は surname2 / given2 / merged2 も付ける

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_honorific_testset.py [--out <データ>/honorific_boundary_v1.jsonl] \\
      [--model az225/privacy-masker-ja-pii-ner-30m] [--seed 42] [--n 2000] [--boost-merged 3]
  --n 0 で全組合せ（姓名形は名を seed で 1 つ選ぶ）。--model はローカルのモデルフォルダか Hub ID（tokenizer だけ使う）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402

DEFAULT_MODEL = "az225/privacy-masker-ja-pii-ner-30m"
SET_NAME = "honorific_boundary_v1"

# 姓 60。先頭 3 つは『姓+さん』が 1 ピースになるもの（必須）、次の 7 つは分かれるもの。末尾 12 はまれな姓。
SURNAMES = (
    "田中", "佐藤", "鈴木",
    "山田", "高橋", "伊藤", "渡辺", "中村", "小林", "加藤",
    "吉田", "山本", "松本", "井上", "木村", "林", "斎藤", "清水", "山口", "森",
    "池田", "橋本", "阿部", "石川", "山崎", "中島", "前田", "藤田", "小川", "岡田",
    "後藤", "長谷川", "村上", "近藤", "石井", "坂本", "遠藤", "青木", "藤井", "西村",
    "福田", "太田", "原", "谷", "三浦", "岡本", "松田", "中川",
    "榊原", "雲井", "朝比奈", "五十嵐", "東海林", "御手洗", "小鳥遊", "勅使河原", "百々", "九十九", "栗花落", "薬袋",
)
GIVEN_NAMES = (
    "太郎", "花子", "一郎", "健太", "美咲", "翔太", "さくら", "陽菜", "大輔", "彩",
    "拓也", "優子", "悠人", "結衣", "直樹", "恵", "蓮", "葵", "智子", "和也",
)
HONORIFICS = ("さん", "様", "氏", "さま", "くん", "ちゃん")
NAME_FORMS = ("surname", "full")
# 文型。{n} = 1 人目の名前、{n2} = 2 人目、{h} = 敬称、[表層|LABEL] = 名前以外の gold
TEMPLATES = {
    "T01": "お世話になっております。{n}{h}に確認します。",
    "T02": "{n}{h}が来社されました。",
    "T03": "先ほど{n}{h}と話しました。",
    "T04": "担当は{n}{h}です。",
    "T05": "{n}{h}へ",
    "T06": "{n}{h}の件",
    "T07": "{n}{h}。",
    "T08": "{n}{h}、{n2}{h}にも連絡しました。",
    "T09": "【{n}{h}】",
    "T10": "{n}{h}は[大阪府堺市|LOCATION]の出身です。",
}
_TOKEN_RE = re.compile(r"\{n2\}|\{n\}|\{h\}|\[([^\]|]+)\|([A-Z_]+)\]")

assert len(SURNAMES) == 60 and len(set(SURNAMES)) == 60, "姓は 60 件・重複なし"
assert len(GIVEN_NAMES) == 20 and len(set(GIVEN_NAMES)) == 20, "名は 20 件・重複なし"


def render(template: str, names: list[str], honorific: str) -> tuple[str, list[list], list[tuple[int, int]]]:
    """文型を展開して text / entities / 名前スパン（名前 i の [start, end)）を返す。"""
    text_parts: list[str] = []
    pos = 0
    entities: list[list] = []
    name_spans: list[tuple[int, int]] = []
    last = 0
    for m in _TOKEN_RE.finditer(template):
        lit = template[last:m.start()]
        text_parts.append(lit)
        pos += len(lit)
        tok = m.group(0)
        if tok == "{n}" or tok == "{n2}":
            idx = 0 if tok == "{n}" else 1
            name = names[idx]
            text_parts.append(name)
            entities.append([pos, pos + len(name), "PERSON"])
            name_spans.append((pos, pos + len(name)))
            pos += len(name)
        elif tok == "{h}":
            text_parts.append(honorific)
            pos += len(honorific)
        else:
            surface, label = m.group(1), m.group(2)
            text_parts.append(surface)
            entities.append([pos, pos + len(surface), label])
            pos += len(surface)
        last = m.end()
    tail = template[last:]
    text_parts.append(tail)
    text = "".join(text_parts)
    for s, e, _ in entities:
        assert 0 <= s < e <= len(text)
    return text, entities, name_spans


def load_tokenizer(model: str):
    """モデルフォルダか Hub ID から fast tokenizer を読む（推論に必要なファイルだけ取得してキャッシュ）。"""
    from transformers import AutoTokenizer

    from privacy_masker.hub import resolve_model_dir

    return AutoTokenizer.from_pretrained(resolve_model_dir(model), use_fast=True)


def merged_flags(tokenizer, text: str, name_spans: list[tuple[int, int]]) -> list[bool]:
    """名前の末尾文字を含むピースが名前の終端を越えて（敬称側へ）伸びているか。名前ごとに返す。"""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out = []
    for s, e in name_spans:
        last = e - 1
        piece = next(((a, b) for a, b in offsets if a <= last < b), None)
        if piece is None:
            raise RuntimeError(f"名前の末尾文字 {last} がどのピースにも載っていません（text len={len(text)}）")
        out.append(piece[1] > e)
    return out


def merges_with_honorific(tokenizer, words) -> set[str]:
    """『語 + 敬称』がどれかの敬称で、語の末尾文字を含むピースが敬称側へ伸びる語の集合（boost の対象）。"""
    if tokenizer is None:
        return set()
    out = set()
    for w in words:
        for h in HONORIFICS:
            if merged_flags(tokenizer, w + h + "。", [(0, len(w))])[0]:
                out.add(w)
                break
    return out


def all_combos() -> list[tuple[str, str, str, str]]:
    """(honorific, template_id, name_form, surname) の全組合せ。層化の単位は先頭 3 つ。"""
    return [(h, t, f, s) for h in HONORIFICS for t in TEMPLATES for f in NAME_FORMS for s in SURNAMES]


def stratified_sample(n: int, rng: random.Random, weights: dict[str, int] | None = None) -> list[tuple[str, str, str, str]]:
    """(敬称, 文型, 名前形) の各層にほぼ同数を割り当て、姓は seed 付きの巡回で配る（weights で姓ごとの重み）。n=0 は全組合せ。"""
    if n <= 0:
        return all_combos()
    weights = weights or {}
    strata = [(h, t, f) for h in HONORIFICS for t in TEMPLATES for f in NAME_FORMS]
    base, extra = divmod(n, len(strata))
    order = list(range(len(strata)))
    rng.shuffle(order)
    quota = {strata[i]: base + (1 if rank < extra else 0) for rank, i in enumerate(order)}
    # 姓の巡回: 1 周ごとに seed 付きで並べ替え、どの姓もほぼ同じ回数使われるようにする
    cycle: list[str] = []

    def next_surname() -> str:
        if not cycle:
            for s in SURNAMES:
                cycle.extend([s] * max(1, int(weights.get(s, 1))))
            rng.shuffle(cycle)
        return cycle.pop()

    picked = []
    for st in strata:
        used: set[str] = set()
        for _ in range(quota[st]):
            s = next_surname()
            tries = 0
            while s in used and tries < len(SURNAMES):   # 1 層に同じ姓を 2 回入れない（層の件数が 60 を超えるときだけ許す）
                cycle.insert(0, s)
                s = next_surname()
                tries += 1
            used.add(s)
            picked.append((*st, s))
    return picked


def build(combos, rng: random.Random, tokenizer, given_weights: dict[str, int] | None = None) -> list[dict]:
    gw = [max(1, int((given_weights or {}).get(g, 1))) for g in GIVEN_NAMES]
    docs = []
    for i, (h, tid, form, surname) in enumerate(combos):
        given = rng.choices(GIVEN_NAMES, weights=gw)[0] if form == "full" else None
        names = [surname + (given or "")]
        meta = {"surname": surname, "given": given, "honorific": h, "template_id": tid, "name_form": form}
        if "{n2}" in TEMPLATES[tid]:
            surname2 = rng.choice([s for s in SURNAMES if s != surname])
            given2 = rng.choices(GIVEN_NAMES, weights=gw)[0] if form == "full" else None
            names.append(surname2 + (given2 or ""))
            meta.update({"surname2": surname2, "given2": given2})
        text, entities, name_spans = render(TEMPLATES[tid], names, h)
        flags = merged_flags(tokenizer, text, name_spans) if tokenizer is not None else [None] * len(name_spans)
        meta["merged"] = flags[0]
        if len(flags) > 1:
            meta["merged2"] = flags[1]
        docs.append({"id": f"{SET_NAME}_{i:05d}", "text": text, "entities": entities, "neutral": [],
                     "source": SET_NAME, "meta": meta})
    return docs


def summary(docs: list[dict]) -> str:
    by_h = Counter(d["meta"]["honorific"] for d in docs)
    by_t = Counter(d["meta"]["template_id"] for d in docs)
    by_f = Counter(d["meta"]["name_form"] for d in docs)
    merged = Counter((d["meta"]["honorific"], d["meta"]["name_form"]) for d in docs if d["meta"]["merged"])
    merged_surnames = Counter(d["meta"]["surname"] for d in docs if d["meta"]["merged"] and d["meta"]["name_form"] == "surname")
    lines = [f"文書 {len(docs)}  gold {sum(len(d['entities']) for d in docs)}",
             "敬称別: " + "  ".join(f"{h}={by_h[h]}" for h in HONORIFICS),
             "文型別: " + "  ".join(f"{t}={by_t[t]}" for t in TEMPLATES),
             "名前形: " + "  ".join(f"{f}={by_f[f]}" for f in NAME_FORMS),
             "merged（名前末尾のピースが敬称まで伸びる）件数 (敬称, 名前形): "
             + ("  ".join(f"{h}/{f}={c}" for (h, f), c in sorted(merged.items())) or "なし"),
             "merged になった姓（姓のみ形）: " + ("  ".join(f"{s}={c}" for s, c in merged_surnames.most_common()) or "なし")]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"出力 JSONL（既定: <データフォルダ>/{SET_NAME}.jsonl）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="meta.merged の計算に使う tokenizer のモデルフォルダか Hub ID。空文字で計算しない（merged は null）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=2000, help="件数上限（層化サンプリング）。0 で全組合せ")
    ap.add_argument("--boost-merged", type=int, default=3, help="『語+敬称』が 1 ピースになる姓・名の重み（1 で一様）")
    args = ap.parse_args()
    out = args.out or os.path.join(drive_dir(), f"{SET_NAME}.jsonl")

    tokenizer = load_tokenizer(args.model) if args.model else None
    boost_s = merges_with_honorific(tokenizer, SURNAMES)
    boost_g = merges_with_honorific(tokenizer, GIVEN_NAMES)
    rng = random.Random(args.seed)
    combos = stratified_sample(args.n, rng, {s: args.boost_merged for s in boost_s})
    docs = build(combos, rng, tokenizer, {g: args.boost_merged for g in boost_g})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"書き出し: {out}")
    print(f"boost ×{args.boost_merged} の姓: {' '.join(sorted(boost_s)) or 'なし'}  名: {' '.join(sorted(boost_g)) or 'なし'}")
    print(summary(docs))


if __name__ == "__main__":
    main()

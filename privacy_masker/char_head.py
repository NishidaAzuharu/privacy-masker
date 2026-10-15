"""文字単位BIOヘッド: サブワード encoder の上に文字ごとの分類層を載せる。

なぜ必要か
----------
token classification では tokenizer の分割が gold 境界をまたぐと厳密なスパンを
表現できない。実測で gold 両端がトークン境界に乗る割合は 86〜90%（固定OOD 89.7 /
盲検 90.3 / GSD 88.0 / OpenPII 86.0）で、残りは原理的に表現不能。

代表例: "医療法人伊織会の理事長" は ['医療法人', '伊織', '会の', '理事', '長'] と割れ、
"会の" が 1 トークンなので ORG を "医療法人伊織会" で切ることが token 単位では不可能。
これまではルールベースの SpanRefiner で末尾の助詞を削っていたが、品詞では
「〜会の」(所属の助詞) と「〜の会」(名前の一部) を区別できないためエンティティ型で
場合分けするしかなく、汎用性がない。

文字ごとに BIO を出せば境界表現率は定義上 100% になる。

文字表現
--------
文字 c の表現は 4 項の和（すべて char_dim 次元）:
  1. h_tok[tok(c)]   : c を含むトークンの encoder 出力を射影したもの
  2. e_piece[c]      : c 単体を 1 文字トークンとして引いた **encoder の入力埋め込み**
                       （102400x256 の事前学習済みテーブルを流用。新規パラメータは射影行列のみ）
  3. script(c)       : 文字種 8 クラスの埋め込み
  4. intra-token 位置 : トークン内の前方/後方オフセット（各 0..7 でクリップ）の埋め込み

1 だけでは同一トークン内の文字が同一ベクトルになり境界が原理的に表現できない。
2 と 4 がそれを解く。続いて文字方向の depthwise separable 混合層を 1 層だけ通す
（受容野 ±2 文字。境界判定は局所問題で、長距離文脈は encoder が h_tok に入れている）。

安全側の設計
------------
プロトタイプ 3 案すべてがチャンク統合と truncation の自作部分で PII 漏れを起こした
（未被覆文字を無警告で O にする / overlap の確率平均でスパンが縮む / 断片が完全な
スパンを棄却する）。ここでは:

  * token-safe 分割: 文字数で切る幅を **トークン数を実測しながら** 決め、max_length を
    超えるチャンクは返さない。truncation は起こさない。上限を満たせない病的な文字が
    あれば黙って切り捨てず ``CharCoverageError`` を投げる。
  * fail-closed: それでも被覆できない文字が残れば ``CharCoverageError`` を投げる。
    黙って O にはしない。
  * ラベル平坦化: 単一 BIO 列は重複スパンを表現できないので、``flatten_entities()``
    が「未知型は除外（型検査を文字の占有より先に行う）／重なるスパンは和集合を
    1 スパンに統合」まで落としてから BIO にする。負けたスパンを丸ごと落とすと
    その部分が O になり「隠さなくてよい」と教えるため、過剰マスク側に倒す。
  * 統合: 確率平均をしない。overlap 部分の扱いは ``merge_strategy`` で選ぶ。
    ``"safe_union"``（既定）は「自分の切り詰められた端に接していないスパン」＝
    エンティティ全体が見えているチャンクの判断だけを採用し、文字単位で和集合を取る。
    ``"core"`` は各文字を担当チャンク 1 つに割り当てる旧方式で、**エンティティ全体の
    保持は保証しない**（担当チャンクが端で切られていて検出できないと、別チャンクの
    正しいスパンの一部が採用されない。``core_ranges`` の説明を参照）。
"""

from __future__ import annotations

import math
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ラベル定義（既存 token モデルと同一順序・同一 id。移植と比較のため揃える）
# ---------------------------------------------------------------------------

LABEL_LIST: list[str] = [
    "O",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-FACILITY",
    "I-FACILITY",
    "B-LOCATION",
    "I-LOCATION",
    "B-ORG",
    "I-ORG",
    "B-PERSON",
    "I-PERSON",
]
LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)
ENTITY_TYPES: list[str] = [lab[2:] for lab in LABEL_LIST if lab.startswith("B-")]

# 補助カテゴリ層（第 2 段）の語彙。privacy_masker.operators.semantic.CATEGORIES と同じ並び（このモジュールは
# ノートブックが Drive から単体 import するので、パッケージ相対 import はせず複製する。一致はテストで固定）
CATEGORY_KINDS: tuple[str, ...] = ("org", "facility", "location")
CATEGORY_NAMES: dict[str, tuple[str, ...]] = {
    "org": ("manufacturing", "it", "finance", "retail", "construction", "transport", "service", "medical_corp",
            "school_corp", "government", "association", "media", "party", "department"),
    "facility": ("medical", "education", "transport", "commerce", "public", "building", "culture", "other"),
    "location": ("country", "prefecture", "city", "town", "district", "foreign", "nature", "other"),
}
CATEGORY_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(CATEGORY_KINDS)}
CATEGORY_INDEX: dict[str, dict[str, int]] = {k: {n: i for i, n in enumerate(names)} for k, names in CATEGORY_NAMES.items()}

IGNORE_INDEX = -100

CHUNK_CHARS = 400
CHUNK_OVERLAP = 50
MAX_LENGTH = 512
N_INTRA = 8   # トークン内オフセットのクリップ上限
N_SCRIPT = 8  # 文字種クラス数


class CharCoverageError(RuntimeError):
    """1 文字でもモデルに載せられなかったときに投げる（fail-closed）。"""


# ---------------------------------------------------------------------------
# 文字種
# ---------------------------------------------------------------------------

_SCRIPT_OTHER = 0
_SCRIPT_KANJI = 1
_SCRIPT_HIRAGANA = 2
_SCRIPT_KATAKANA = 3
_SCRIPT_LATIN = 4
_SCRIPT_DIGIT = 5
_SCRIPT_PUNCT = 6
_SCRIPT_SPACE = 7


def script_class(ch: str) -> int:
    """文字 1 つを 8 クラスの文字種に落とす（境界判定に効く最小の素性）。"""
    if ch.isspace():
        return _SCRIPT_SPACE
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return _SCRIPT_HIRAGANA
    if 0x30A0 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9D:
        return _SCRIPT_KATAKANA
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or o in (0x3005, 0x3007):
        return _SCRIPT_KANJI
    if ch.isdigit():
        return _SCRIPT_DIGIT
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or (0xFF21 <= o <= 0xFF3A) or (0xFF41 <= o <= 0xFF5A):
        return _SCRIPT_LATIN
    if unicodedata.category(ch)[0] in ("P", "S"):
        return _SCRIPT_PUNCT
    return _SCRIPT_OTHER


# ---------------------------------------------------------------------------
# スパン <-> 文字ラベル
# ---------------------------------------------------------------------------

# 重なりを 1 スパンにまとめるときの型の優先順位。小さいほど優先。
# 基準は「policy 側で Keep（マスクしない）になり得るか」。ADDRESS は
# policy.DEFAULT_PUBLIC_TYPES に含まれないので決して Keep にならない。PERSON は
# IdentifiabilityAnnotator が公知人物と判定したときだけ Keep になる。残りは
# 公知判定で容易に Keep になるので最も危険。policy.py を import しないのは
# レイヤを逆流させないためで、DEFAULT_PUBLIC_TYPES を変えたらここも見直すこと。
_KEEP_RISK_ORDER: dict[str, int] = {"ADDRESS": 0, "PERSON": 1}


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    label: str          # BIO 接頭辞なしの型名
    score: float = 1.0


def _as_spans(entities: Iterable[Any]) -> list[CharSpan]:
    """dict / Entity / tuple / CharSpan を CharSpan に正規化する。"""
    out: list[CharSpan] = []
    for e in entities:
        if isinstance(e, CharSpan):
            out.append(e)
            continue
        if isinstance(e, dict):
            s, t = e.get("start"), e.get("end")
            lab = e.get("label") or e.get("entity_type") or e.get("entity_group") or e.get("type")
        elif isinstance(e, (tuple, list)) and len(e) >= 3:
            s, t, lab = e[0], e[1], e[2]
        else:
            s = getattr(e, "start", None)
            t = getattr(e, "end", None)
            lab = getattr(e, "entity_type", None) or getattr(e, "label", None)
        if s is None or t is None or lab is None:
            continue
        lab = str(lab)
        if lab.startswith(("B-", "I-")):
            lab = lab[2:]
        out.append(CharSpan(int(s), int(t), lab, float(getattr(e, "score", 1.0) if not isinstance(e, dict) else e.get("score", 1.0))))
    return out


OVERLAP_POLICIES = ("merge", "error")
UNKNOWN_TYPE_POLICIES = ("drop", "error")


def flatten_entities(
    text: str,
    entities: Iterable[Any],
    *,
    overlap: str = "merge",
    unknown_types: str = "drop",
    stats: dict[str, Any] | None = None,
) -> list[CharSpan]:
    """単一 BIO 列で厳密に表現できる「重複なし・既知型のみ」のスパン列に落とす。

    平坦化方針（学習ラベルと評価 gold の両方でこの関数を通すこと）
    ------------------------------------------------------------
    1. 範囲外・逆順のスパンを落とす。
    2. **型の検査を先に行う**。``LABEL_LIST`` にない型は ``unknown_types`` に従って
       落とす（既定）か例外にする。文字の占有を決める前に落とすので、未知型が
       既知型の正解を消すことはない（例: UNKNOWN[0,4) と PERSON[1,3) を渡すと
       PERSON が残る）。
    3. 残ったスパンのうち **重なり合うものは 1 つの連結成分にまとめ**、その和集合を
       1 スパンにする（``overlap="merge"``）。型は ``_KEEP_RISK_ORDER`` の順（マスク
       されなくなる危険が低い型を優先）で選び、同順なら長い方・先に現れた方。
       ``overlap="error"`` なら例外。

    型を「長い方」で選んではいけない理由
    ----------------------------------
    長さで選ぶと ORG[0,6) が PERSON[4,9) を吸収して ORG[0,9) になる。ORG は
    ``policy.DEFAULT_PUBLIC_TYPES`` に含まれるので、``IdentifiabilityAnnotator`` が
    公知と判定すると ``ORG_PUBLIC`` になり ``Policy.preset("minimal"/"standard")`` では
    **Keep（マスクしない）** になる。つまり人名が丸ごと素通しになる。実測で
    ``医療法人伊織会山田太郎`` が Keep になり PERSON が一切マスクされないことを確認した。
    そのため型は「Keep になり得るか」の順で選ぶ。``ADDRESS`` は
    ``DEFAULT_PUBLIC_TYPES`` に無いので決して Keep にならず最優先、``PERSON`` は
    公知人物と判定された場合だけ Keep なので次、残りは同格。

    3 で「長い方だけを残して短い方を丸ごと落とす」ことはしない。ORG[0,6) と
    PERSON[4,9) で PERSON を落とすと 6〜8 文字目が O になり「隠さなくてよい」と
    教えることになる。匿名化では見逃しが最悪の失敗なので、和集合を取る側（過剰
    マスク側）に倒す。逆に「文字ごとに長い方を優先」（この関数の以前の挙動）だと
    PERSON[6,9) という **元データに存在しない部分スパン** を学習させてしまい、
    gold とも一致しなくなる。どちらも避けるための和集合方針である。

    実データ（out_ner / synthetic_pii_v4 / v5c / v5e の 30,343 文書・100,361
    エンティティ）では重複ペアは 0 件だったので、既定を merge にしても現行の
    学習ラベルは変わらない。入れ子（ADDRESS ⊃ LOCATION 等）の結果も、以前の
    「文字ごとに長い方を優先」と一致する。

    ``stats`` に dict を渡すと落とした/まとめた件数が入る（呼び出し側が気付けるように）。
    """
    if overlap not in OVERLAP_POLICIES:
        raise ValueError(f"overlap は {OVERLAP_POLICIES} のいずれか: {overlap!r}")
    if unknown_types not in UNKNOWN_TYPE_POLICIES:
        raise ValueError(f"unknown_types は {UNKNOWN_TYPE_POLICIES} のいずれか: {unknown_types!r}")

    n = len(text)
    raw = _as_spans(entities)
    in_range = [s for s in raw if 0 <= s.start < s.end <= n]

    known: list[CharSpan] = []
    bad_types: list[str] = []
    for sp in in_range:
        if ("B-" + sp.label) in LABEL2ID:
            known.append(sp)
        else:
            bad_types.append(sp.label)
    if bad_types and unknown_types == "error":
        raise ValueError(
            f"未知のエンティティ型 {sorted(set(bad_types))} が {len(bad_types)} 件あります"
            f"（既知の型: {ENTITY_TYPES}）"
        )

    # 入力順を tie-break に使うので、並べ替える前に元の位置を持たせる
    indexed = sorted(enumerate(known), key=lambda it: (it[1].start, it[1].end))
    out: list[CharSpan] = []
    n_groups = n_absorbed = 0
    group: list[tuple[int, CharSpan]] = []
    group_end = -1

    def _flush() -> None:
        nonlocal n_groups, n_absorbed
        if not group:
            return
        if len(group) == 1:
            out.append(group[0][1])
            return
        n_groups += 1
        n_absorbed += len(group) - 1
        rep = min(
            group,
            key=lambda it: (
                _KEEP_RISK_ORDER.get(it[1].label, len(_KEEP_RISK_ORDER)),
                -(it[1].end - it[1].start),
                it[1].start,
                it[0],
            ),
        )[1]
        out.append(CharSpan(group[0][1].start, max(sp.end for _, sp in group), rep.label, rep.score))

    for idx, sp in indexed:
        if group and sp.start < group_end:
            group.append((idx, sp))
            group_end = max(group_end, sp.end)
            continue
        _flush()
        group = [(idx, sp)]
        group_end = sp.end
    _flush()

    if n_groups and overlap == "error":
        raise ValueError(
            f"重複するエンティティが {n_groups} 箇所あります（単一 BIO 列では表現できません）。"
            " データ側で重複を解消するか overlap='merge' を指定してください。"
        )

    if stats is not None:
        stats["n_input"] = len(raw)
        stats["n_out"] = len(out)
        stats["dropped_out_of_range"] = len(raw) - len(in_range)
        stats["dropped_unknown_type"] = len(bad_types)
        stats["unknown_type_names"] = sorted(set(bad_types))
        stats["merged_overlap_groups"] = n_groups
        stats["merged_overlap_spans"] = n_absorbed
    return out


def char_labels_from_entities(
    text: str,
    entities: Iterable[Any],
    *,
    overlap: str = "merge",
    unknown_types: str = "drop",
    stats: dict[str, Any] | None = None,
) -> list[int]:
    """文字数と同じ長さの BIO ラベル id 配列を作る。

    平坦化（重複の統合・未知型の除外）は :func:`flatten_entities` が行う。評価側の
    gold も同じ関数を通せば、ここで作るラベルから復元されるスパンと厳密に一致する。
    """
    spans = flatten_entities(
        text, entities, overlap=overlap, unknown_types=unknown_types, stats=stats
    )
    labels = [LABEL2ID["O"]] * len(text)
    for sp in spans:
        labels[sp.start] = LABEL2ID["B-" + sp.label]
        for i in range(sp.start + 1, sp.end):
            labels[i] = LABEL2ID["I-" + sp.label]
    return labels


# ---------------------------------------------------------------------------
# 分割: トークン数を実測しながら左から窓を置く（truncation を発生させない）
# ---------------------------------------------------------------------------

def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def _fit_chunk_chars(text: str, tokenizer, start: int, max_chars: int, max_length: int) -> int:
    """``text[start:start+L)`` が max_length token 以内に収まる文字数 L を返す（入らなければ 0）。

    まず max_chars をそのまま試すので、再分割が要らない通常のテキストでは tokenizer 呼び出し
    1 回で済む。超えた場合だけ二分探索する。tokenizer の token 数は文字数に対して厳密に単調とは
    限らない（語彙にある長い連続で減ることがある）ので、返す L は「上限内だと実測した値」であり
    最大であることは保証しない。返却チャンクが上限以内であることは実測で保証される。
    """
    if max_chars <= 0:
        return 0
    if _token_len(text[start : start + max_chars], tokenizer) <= max_length:
        return max_chars
    best = 0
    lo, hi = 1, max_chars - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _token_len(text[start : start + mid], tokenizer) <= max_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def token_safe_chunks(
    text: str,
    tokenizer,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_length: int = MAX_LENGTH,
    min_chunk_chars: int = 8,
) -> list[tuple[int, str]]:
    """``[(offset, chunk_text), ...]`` を返す。各チャンクは必ず max_length token 以内。

    既存 detector と同じ「400 文字 / overlap 50」で切るが、byte-fallback される文字種
    （絵文字 2.0 tok/char・タイ語 1.9・韓国語 1.7 を実測）では 400 文字が 512 token を
    超えるため、そのチャンクだけ token 数を見ながら短く切る。日本語 0.62・英語 0.20 では
    短縮は発生せず、従来と同じ「400 文字 / step 350」の配置になる。

    左から順に「上限内に収まる最大幅の窓」を置いていく。窓を再帰的に半分に割る方式は、
    重なりの付け方次第で子区間が親と同一になり同じ区間を無限に再処理するうえ、重なりが
    再帰の各段で二重に積み上がってチャンク数が文字数に対して超線形に増える。前進走査なら
    1 周ごとに必ず開始・終了が前へ進むので停止し、開始・終了ともに単調増加（階段状）に
    なるため ``core_ranges`` の担当割り当ても必ずチャンクの内側に収まる。

    上限を満たせない場合は ``CharCoverageError`` を投げる（fail-closed）。後段の
    truncation に任せると切り落とされた範囲が無警告で「エンティティなし」になり PII が
    漏れる。上限を満たせないのは「1 文字が max_length を超える token 数へ膨張する」病的な
    場合だけで、既定設定 (400/50/512) では日本語・英語・絵文字・タイ語・韓国語・簡体字の
    いずれでも起きない（実 tokenizer で 400〜2000 文字を実測）。

    ``min_chunk_chars`` は API 互換のために受け取るだけで、token 上限の免除には使わない。
    上限超過チャンクを返さないため、必要ならこの値より短く切る。
    """
    if chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be smaller than chunk_chars")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    n = len(text)
    if n == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        take = _fit_chunk_chars(text, tokenizer, start, min(chunk_chars, n - start), max_length)
        if take == 0:
            one = _token_len(text[start : start + 1], tokenizer)
            raise CharCoverageError(
                f"text[{start}:{start + 1}]={text[start]!r} の 1 文字が {one} token に膨張し、"
                f"max_length={max_length} をどう分割しても満たせません。"
                " max_length を増やすか、この文字を正規化・除去してから渡してください。"
            )
        end = start + take
        if spans and end <= spans[-1][1]:
            # 重なりを付けた窓が直前のチャンクの内側に収まってしまった（この付近は token 密度が
            # 高い）。重なりを捨てて直前の終端から続け、前進と終端の単調増加を保証する。
            start = spans[-1][1]
            continue
        spans.append((start, end))
        if end >= n:
            break
        # 重なりは幅の半分までに制限する。制限しないと次の開始位置が前へ進まない。
        start = end - min(chunk_overlap, take // 2)

    # fail-closed: 全文字被覆と階段状を確認する。破れていれば core 割り当てが成立しない。
    broken = spans[0][0] != 0 or spans[-1][1] != n
    for (pa, pb), (ca, cb) in zip(spans, spans[1:]):
        if not (pa < ca <= pb < cb):
            broken = True
            break
    if broken:
        raise CharCoverageError(
            f"分割結果が [0,{n}) を隙間なく階段状に覆えていません: {spans}。"
            " chunk_chars / chunk_overlap / max_length の組み合わせを見直してください。"
        )
    return [(a, text[a:b]) for a, b in spans]


def core_ranges(chunks: Sequence[tuple[int, str]], n_chars: int) -> list[tuple[int, int]]:
    """各チャンクが「予測を担当する」区間を返す。区間は [0, n_chars) をちょうど分割する。

    overlap 部分は文脈提供のみに使い、投票には参加させない。境界はチャンクの重なりの
    中点に置くので、担当区間の両側には必ず文脈が付く。

    これは **全文字への担当割り当て** であって、エンティティ全体の保持保証ではない。
    反例（``chunk_chars=400 / chunk_overlap=50``、文書長 460）:

        chunk A=[0,400) core=[0,375) / chunk B=[350,460) core=[375,460)
        gold PERSON=[360,410)
        A は 400 で切られて gold 全体が見えず検出しない。B は [360,410) を正しく出す。
        しかし 360〜374 の担当は A なので、最終出力は [375,410) となり先頭 15 文字が漏れる。

    この漏れを避けるため ``predict_char_spans`` の既定は ``merge_strategy="safe_union"``
    で、core 割り当ては ``merge_strategy="core"`` を選んだときだけ使う。
    """
    m = len(chunks)
    if m == 0:
        return []
    if m == 1:
        return [(0, n_chars)]
    bounds = [0]
    for k in range(1, m):
        prev_a, prev_txt = chunks[k - 1]
        cur_a = chunks[k][0]
        prev_b = prev_a + len(prev_txt)
        mid = (cur_a + min(prev_b, n_chars)) // 2      # 重なりの中点
        mid = max(bounds[-1], min(mid, n_chars))       # 単調性を強制
        bounds.append(mid)
    bounds.append(n_chars)
    return [(bounds[k], bounds[k + 1]) for k in range(m)]


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def single_char_piece_id(tokenizer, ch: str, cache: dict[str, int] | None = None) -> int:
    """文字 1 つ -> 語彙上の 1 文字トークン id（無ければ unk）。

    encoder の入力埋め込みを文字埋め込みとして流用するための索引。ruri-v3 の
    SentencePiece 語彙には日本語 1 文字がほぼ全て入っている（漢字40/40・ひらがな46/46・
    カタカナ46/46・英数62/62・記号16/16 で unk ゼロを実測）。
    """
    if cache is not None:
        got = cache.get(ch)
        if got is not None:
            return got
    tid = tokenizer.convert_tokens_to_ids(ch)
    unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    if tid is None or tid < 0:
        tid = unk
    if cache is not None:
        cache[ch] = tid
    return tid


def encode_chunk(
    text: str,
    tokenizer,
    entities: Iterable[Any] | None = None,
    max_length: int = MAX_LENGTH,
    piece_cache: dict[str, int] | None = None,
    strict: bool = True,
    strict_labels: bool = False,
    category_spans: Iterable[tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    """1 チャンクを学習/推論サンプルに変換する。

    ``strict=True`` なら、トークンに載らなかった文字が 1 つでもあれば
    ``CharCoverageError`` を投げる（fail-closed）。学習データ作成では
    ``strict=False`` にして該当文字を IGNORE_INDEX にする運用も選べる。

    ``strict_labels=True`` にすると、未知のエンティティ型と重複スパンを例外にする
    （データ側を直す運用）。既定は :func:`flatten_entities` の既定（未知型は除外、
    重複は和集合に統合）で、落とした/統合した件数は戻り値の ``label_stats`` に入る。

    ``category_spans`` はチャンク相対の文字位置 ``[start, end)`` と kind / category の添字
    （``CATEGORY_KINDS`` / ``CATEGORY_NAMES`` の並び）。学習の補助損失用で、推論では空。範囲外は ``ValueError``。
    戻り値の ``cat_spans`` に検査済みの組がそのまま入る（省略時は ``[]``）。
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])
    offsets = list(enc["offset_mapping"])

    n = len(text)
    char_to_token = [0] * n
    char_pos_fwd = [0] * n
    char_pos_bwd = [0] * n
    char_mask = [0] * n
    for ti, (a, b) in enumerate(offsets):
        if b <= a:                     # special token は offset (0,0)
            continue
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        for i in range(a, b):
            char_to_token[i] = ti
            char_pos_fwd[i] = min(i - a, N_INTRA - 1)
            char_pos_bwd[i] = min(b - 1 - i, N_INTRA - 1)
            char_mask[i] = 1

    uncovered = n - sum(char_mask)
    if uncovered and strict:
        raise CharCoverageError(
            f"{uncovered}/{n} 文字が max_length={max_length} に収まらず未被覆。"
            " token_safe_chunks() で分割してから渡してください。"
        )

    cache = piece_cache if piece_cache is not None else {}
    char_piece_ids = [single_char_piece_id(tokenizer, c, cache) for c in text]
    char_script = [script_class(c) for c in text]

    labels: list[int] | None = None
    label_stats: dict[str, Any] | None = None
    if entities is not None:
        label_stats = {}
        raw = char_labels_from_entities(
            text,
            entities,
            overlap="error" if strict_labels else "merge",
            unknown_types="error" if strict_labels else "drop",
            stats=label_stats,
        )
        labels = [raw[i] if char_mask[i] else IGNORE_INDEX for i in range(n)]

    cat_spans: list[tuple[int, int, int, int]] = []
    for sp in (category_spans or ()):
        s, e, k, c = (int(x) for x in sp)
        if not (0 <= s < e <= n) or not (0 <= k < len(CATEGORY_KINDS)) or not (0 <= c < len(CATEGORY_NAMES[CATEGORY_KINDS[k]])):
            raise ValueError(f"invalid category span {sp!r} for a chunk of {n} chars")
        cat_spans.append((s, e, k, c))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_to_token": char_to_token,
        "char_piece_ids": char_piece_ids,
        "char_script": char_script,
        "char_pos_fwd": char_pos_fwd,
        "char_pos_bwd": char_pos_bwd,
        "char_mask": char_mask,
        "labels": labels,
        "n_chars": n,
        "uncovered": uncovered,
        "label_stats": label_stats,
        "cat_spans": cat_spans,
    }


class CharNerCollator:
    """token 長・文字長ともに可変長 padding する collator（Trainer にも素ループにも使える）。

    補助カテゴリ損失用に ``cat_spans``（``(B, S_max, 4)`` long、S_max = max(件数, 1)、padding 0）と
    ``cat_mask``（``(B, S_max)`` bool）を常に返す。推論では全部偽で、モデルの ``forward`` は使わない。
    """

    _CHAR_KEYS = ("char_to_token", "char_piece_ids", "char_script", "char_pos_fwd", "char_pos_bwd", "char_mask")

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        bsz = len(features)
        max_t = max(len(f["input_ids"]) for f in features)
        max_c = max(max(len(f["char_to_token"]), 1) for f in features)

        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.full((bsz, max_t), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((bsz, max_t), dtype=torch.long),
        }
        for k in self._CHAR_KEYS:
            batch[k] = torch.zeros((bsz, max_c), dtype=torch.long)
        has_labels = all(f.get("labels") is not None for f in features)
        if has_labels:
            batch["labels"] = torch.full((bsz, max_c), IGNORE_INDEX, dtype=torch.long)
        max_s = max(max(len(f.get("cat_spans") or ()) for f in features), 1)
        batch["cat_spans"] = torch.zeros((bsz, max_s, 4), dtype=torch.long)
        batch["cat_mask"] = torch.zeros((bsz, max_s), dtype=torch.bool)

        for i, f in enumerate(features):
            t = len(f["input_ids"])
            batch["input_ids"][i, :t] = torch.tensor(f["input_ids"], dtype=torch.long)
            batch["attention_mask"][i, :t] = torch.tensor(f["attention_mask"], dtype=torch.long)
            cs = f.get("cat_spans") or []
            if cs:
                batch["cat_spans"][i, : len(cs)] = torch.tensor(cs, dtype=torch.long)
                batch["cat_mask"][i, : len(cs)] = True
            c = len(f["char_to_token"])
            if not c:
                continue
            for k in self._CHAR_KEYS:
                batch[k][i, :c] = torch.tensor(f[k], dtype=torch.long)
            if has_labels:
                batch["labels"][i, :c] = torch.tensor(f["labels"], dtype=torch.long)
        return batch


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def depthwise_shift_mix(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """depthwise 1D convolution と等価で、CPU で速い実装。

    ``out[b, t, d] = sum_k weight[k, d] * x[b, t + k - K//2, d]``（範囲外は 0 padding）。

    ``nn.Conv1d(D, D, K, groups=D)`` と同じ演算だが、PyTorch の CPU grouped-conv は
    「1 チャネル conv を D 回」に分解されオーバーヘッド支配になり実測で 15 倍遅い
    （5.72ms 対 0.38ms, batch4 x 1600文字, char_dim=128）。nn.Conv1d との数値差は
    forward 1.2e-7 / grad 3.8e-6 で float32 の丸め相当。
    """
    k_size, dim = weight.shape
    r = k_size // 2
    n = x.shape[1]
    out = x * weight[r]
    for k in range(k_size):
        s = k - r
        if s == 0:
            continue
        w = weight[k].expand(1, 1, dim)
        if s > 0:
            if n > s:
                out[:, : n - s, :].addcmul_(x[:, s:, :], w)
        elif n > -s:
            out[:, -s:, :].addcmul_(x[:, : n + s, :], w)
    return out


class CharMixHead(nn.Module):
    """文字表現の構築 + 文字方向の軽量文脈混合 1 層 + 分類。

    行列積は (1) tok 射影 (2) piece 射影 (3) depthwise 混合 (4) pointwise (5) classifier のみ。
    ``use_mix=False`` にすると混合層を外した ablation になる。
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = NUM_LABELS,
        char_dim: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_mix: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.char_dim = char_dim
        self.use_mix = use_mix
        self.kernel_size = kernel_size
        self.tok_proj = nn.Linear(hidden_size, char_dim)
        self.piece_proj = nn.Linear(hidden_size, char_dim, bias=False)
        self.script_emb = nn.Embedding(N_SCRIPT, char_dim)
        self.pos_fwd_emb = nn.Embedding(N_INTRA, char_dim)
        self.pos_bwd_emb = nn.Embedding(N_INTRA, char_dim)
        self.in_norm = nn.LayerNorm(char_dim)

        if use_mix:
            self.mix_weight = nn.Parameter(torch.empty(kernel_size, char_dim))
            self.mix_bias = nn.Parameter(torch.zeros(char_dim))
            nn.init.uniform_(self.mix_weight, -1.0 / math.sqrt(kernel_size), 1.0 / math.sqrt(kernel_size))
            self.pw = nn.Linear(char_dim, char_dim)
            self.mix_norm = nn.LayerNorm(char_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(char_dim, num_labels)

        for emb in (self.script_emb, self.pos_fwd_emb, self.pos_bwd_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(
        self,
        token_hidden: torch.Tensor,   # (B, T, H)
        piece_embeds: torch.Tensor,   # (B, C, H)
        char_to_token: torch.Tensor,  # (B, C)
        char_script: torch.Tensor,
        char_pos_fwd: torch.Tensor,
        char_pos_bwd: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, t, _ = token_hidden.shape
        # 非 in-place clamp（呼び出し側の batch テンソルを書き換えない）
        idx = char_to_token.clamp(min=0, max=max(t - 1, 0))
        # tok_proj を gather より先に掛ける。射影の対象が文字数 C ではなく token 数 T に
        # なり、日本語では C > T なので行列積が減る（Linear は位置ごとに独立なので
        # gather との順序交換は数学的に同一。float32 の丸め差だけが出る）。
        gathered = torch.gather(
            self.tok_proj(token_hidden), 1, idx.unsqueeze(-1).expand(-1, -1, self.char_dim)
        )

        x = (
            gathered
            + self.piece_proj(piece_embeds)
            + self.script_emb(char_script)
            + self.pos_fwd_emb(char_pos_fwd)
            + self.pos_bwd_emb(char_pos_bwd)
        )
        x = F.gelu(self.in_norm(x))

        m = char_mask.to(x.dtype).unsqueeze(-1)
        x = x * m                      # padding を混合層の受容野に入れない
        if self.use_mix:
            y = depthwise_shift_mix(x, self.mix_weight) + self.mix_bias
            y = self.pw(F.gelu(y))
            x = self.mix_norm(x + self.dropout(y)) * m
        else:
            x = self.dropout(x)
        return self.classifier(x)      # (B, C, num_labels)


class CharBioNerModel(nn.Module):
    """事前学習（＋token NER fine-tune）済み encoder + 文字BIOヘッド。"""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int = NUM_LABELS,
        char_dim: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_mix: bool = True,
        class_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.num_labels = num_labels
        self.head = CharMixHead(
            hidden_size=encoder.config.hidden_size,
            num_labels=num_labels,
            char_dim=char_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_mix=use_mix,
        )
        # 文字単位では B- が I- に対して相対的に希薄になる（7 文字の ORG は B 1 個に I 6 個。
        # token 単位なら B 1 個に I 2 個）。境界の学習が弱い場合はここで B- を重み付けする。
        if class_weights is None:
            self.register_buffer("class_weights", None, persistent=False)
        else:
            if len(class_weights) != num_labels:
                raise ValueError(f"class_weights は {num_labels} 要素必要です")
            self.register_buffer("class_weights", torch.tensor(list(class_weights), dtype=torch.float), persistent=False)

    @classmethod
    def from_pretrained_encoder(
        cls,
        model_path: str,
        num_labels: int = NUM_LABELS,
        char_dim: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_mix: bool = True,
        class_weights: Sequence[float] | None = None,
        **encoder_kwargs: Any,
    ) -> "CharBioNerModel":
        """token 分類ヘッド付きの学習済みディレクトリから **encoder 重みだけ** 読み込む。"""
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_path, **encoder_kwargs)
        return cls(
            encoder,
            num_labels=num_labels,
            char_dim=char_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_mix=use_mix,
            class_weights=class_weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_to_token: torch.Tensor,
        char_piece_ids: torch.Tensor,
        char_script: torch.Tensor,
        char_pos_fwd: torch.Tensor,
        char_pos_bwd: torch.Tensor,
        char_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_unused: Any,
    ) -> dict[str, torch.Tensor]:
        token_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        piece_embeds = self.encoder.get_input_embeddings()(char_piece_ids)
        logits = self.head(
            token_hidden=token_hidden,
            piece_embeds=piece_embeds,
            char_to_token=char_to_token,
            char_script=char_script,
            char_pos_fwd=char_pos_fwd,
            char_pos_bwd=char_pos_bwd,
            char_mask=char_mask,
        )
        out: dict[str, torch.Tensor] = {"logits": logits}
        out["token_hidden"] = token_hidden        # スパンベクトル用（設計 §2）。学習ループは無視する
        if labels is not None:
            w = self.class_weights
            flat_logits = logits.reshape(-1, self.num_labels).float()
            flat_labels = labels.reshape(-1)
            n_valid = (flat_labels != IGNORE_INDEX).sum()
            # 有効ラベル数を必ず返す。学習ループはこれを見て「更新すべきでない batch」を
            # 判別し、optimizer/scheduler を進めず loss 平均の分母にも入れない。
            out["n_valid_labels"] = n_valid
            if int(n_valid) == 0:
                # 全ラベルが IGNORE_INDEX の batch。reduction="mean" は分母が 0 になって
                # NaN を返し、それが loss 集計や（AMP の）scaler 経由で伝播する。
                # ここでは有効ラベル 0 件の cross_entropy を空スライスで明示的に計算する。
                # 空入力なので logits が非有限でも NaN は混ざらず、勾配グラフだけ残った 0 に
                # なる（呼び出し側が誤って backward しても更新量 0）。
                # 全ラベルが O の負例は n_valid > 0 なのでこの分岐に入らない（除外しない）。
                out["loss"] = F.cross_entropy(
                    flat_logits[:0],
                    flat_labels[:0],
                    weight=None if w is None else w.to(logits.device, torch.float),
                    reduction="sum",
                )
            else:
                out["loss"] = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    weight=None if w is None else w.to(logits.device, torch.float),
                    ignore_index=IGNORE_INDEX,
                )
        return out

    def num_added_parameters(self) -> int:
        return sum(p.numel() for p in self.head.parameters())


# ---------------------------------------------------------------------------
# デコード
# ---------------------------------------------------------------------------

def decode_char_labels(
    text: str,
    type_ids: Sequence[int],
    scores: Sequence[float],
    starts: Sequence[bool],
    seam: Sequence[bool] | None = None,
    *,
    suppress_seam_b: bool = False,
) -> list[CharSpan]:
    """文字ごとの「型 id / スコア / B か」からスパンを組む。

    型が同じ連続文字は 1 スパンにまとめ、``starts[i]``（B- 予測）が立っているときに
    区切る。**既定では継ぎ目でも B- を尊重する。**

    以前は ``seam[i]``（担当チャンクが切り替わる位置）の B- を無条件に無効化していたが、
    それだと継ぎ目にちょうど別実体の境界が来たときに 2 実体が 1 つに融合した
    （位置 371-374 と 375-378 に別人、正しい BIO ``B I I I B I I I``、seam=375 で
    ``[(371,379)]`` になる）。継ぎ目位置を担当するのは後続チャンクで、既定設定なら
    ``chunk_overlap // 2`` 文字の左文脈を持って判断している。左文脈ゼロで偽 B が出るのは
    チャンクのテキスト先頭だが、そこは担当区間の外なので採用されない。しかも偽 B が
    継ぎ目より右にずれた場合はこの抑制では救えない（抑制は継ぎ目 1 文字だけに効く）。

    ``suppress_seam_b=True`` は旧挙動。マスキング優先で「継ぎ目由来の断片化は絶対に
    避けたい／隣接別実体の融合は許容する」用途のための明示的オプションで、
    ``seam`` を渡したときだけ意味を持つ。
    """
    n = len(text)
    spans: list[CharSpan] = []
    i = 0
    while i < n:
        t = type_ids[i]
        if t <= 0:
            i += 1
            continue
        j = i + 1
        acc = [scores[i]]
        while j < n and type_ids[j] == t:
            if starts[j] and not (suppress_seam_b and seam is not None and seam[j]):
                break
            acc.append(scores[j])
            j += 1
        spans.append(CharSpan(i, j, ENTITY_TYPES[t - 1], sum(acc) / len(acc)))
        i = j
    return spans


def merge_adjacent_same_type(spans: Sequence[CharSpan]) -> list[CharSpan]:
    """接している同型スパンを 1 つにまとめる。

    文字単位 BIO では、スパン内部の 1 文字に誤って B- が立つとスパンが 2 つに割れる
    （例: `河村敬史` が `河` + `村敬史`）。マスキング用途では覆う文字は同じなので
    漏れにはならないが、プレースホルダが 2 個出て見苦しく、厳密スパン一致の指標も落ちる。

    副作用: 区切り文字なしで隣接する同型の別エンティティ（表組みの `山田太郎佐藤花子` 等）は
    1 つに融合する。マスクは掛かるので漏れないが、Resolver が別人に別 ID を振れなくなる。
    既定で有効にはせず、呼び出し側が選べるようにしてある。
    """
    out: list[CharSpan] = []
    for sp in sorted(spans, key=lambda s: (s.start, s.end)):
        if out and out[-1].label == sp.label and out[-1].end == sp.start:
            prev = out[-1]
            n_prev, n_cur = prev.end - prev.start, sp.end - sp.start
            out[-1] = CharSpan(
                prev.start, sp.end, prev.label,
                (prev.score * n_prev + sp.score * n_cur) / (n_prev + n_cur),
            )
        else:
            out.append(sp)
    return out


MERGE_STRATEGIES: tuple[str, ...] = ("safe_union", "core")


def _merge_overlapping_same_type(spans: Sequence[CharSpan]) -> list[CharSpan]:
    """同型で **重なる** スパンだけを 1 つにまとめる。

    複数チャンクが同じ実体を見ると同じ（か、ほぼ同じ）スパンを重複して出す。それは
    まとめたい。一方「接しているだけ」の同型スパンは、B- で分かれた別実体の可能性が
    あるのでまとめない（``merge_adjacent_same_type`` と役割を分けている）。
    スコアは高い方を採る（下流の閾値で落とさないため安全側）。
    """
    out: list[CharSpan] = []
    for sp in sorted(spans, key=lambda s: (s.label, s.start, s.end)):
        if out and out[-1].label == sp.label and sp.start < out[-1].end:
            prev = out[-1]
            out[-1] = CharSpan(prev.start, max(prev.end, sp.end), prev.label, max(prev.score, sp.score))
        else:
            out.append(sp)
    return sorted(out, key=lambda s: (s.start, s.end, s.label))


def _subtract(sp: CharSpan, blockers: Sequence[CharSpan]) -> list[CharSpan]:
    """``sp`` から ``blockers`` に覆われた部分を除いた残りを返す。"""
    pieces = [(sp.start, sp.end)]
    for b in blockers:
        nxt: list[tuple[int, int]] = []
        for a, z in pieces:
            if b.end <= a or b.start >= z:
                nxt.append((a, z))
                continue
            if a < b.start:
                nxt.append((a, b.start))
            if b.end < z:
                nxt.append((b.end, z))
        pieces = nxt
        if not pieces:
            break
    return [CharSpan(a, z, sp.label, sp.score) for a, z in pieces if z > a]


def _resolve_type_conflicts(spans: Sequence[CharSpan]) -> list[CharSpan]:
    """型が違うスパンが重なったとき、**Keep 危険度が低い型**を丸ごと残し、他は残りだけ残す。

    以前はスコアだけで勝者を決めていた。それだと ORG[355,395) がスコア 0.90、
    PERSON[360,390) がスコア 0.60 のとき PERSON が完全に内包されて**消える**。
    「どちらの型でもマスクは掛かる」という前提は ``Policy.preset("minimal"/"standard")``
    では成立せず、``ORG_PUBLIC`` と判定されると Keep（マスクしない）になるので、
    人名部分まで素通しになる。``flatten_entities`` で学習ラベル側だけこの順序を入れて
    推論側を放置していたのは不整合だったので揃える。

    ``_KEEP_RISK_ORDER`` の順に処理するので、より機密な型が削られることはない。
    同順の型どうし（ORG と FACILITY 等）はスコアで決めるが、どちらも同じ Keep 危険度で
    文字の被覆は保たれるので漏れの向きに差は出ない。

    なお ``Engine`` 経由なら最終的な重なり解消は ``EntityResolver`` が行う。ここでの解消は
    単一 BIO 列として出力するために必要な最小限のもの。

    同型の重なりは呼び出し前に ``_merge_overlapping_same_type`` で解消済みの前提。

    重なりで連結成分に切ってから成分ごとに解く。素朴に全採用済みスパンと突き合わせると
    スパン数の 2 乗になり、未学習モデルのように大量のスパンが出る入力で極端に遅くなる。
    """
    out: list[CharSpan] = []
    group: list[CharSpan] = []
    reach = -1
    for sp in sorted(spans, key=lambda s: (s.start, s.end, s.label)):
        if group and sp.start >= reach:
            out.extend(_resolve_group(group))
            group = []
        group.append(sp)
        reach = max(reach, sp.end)
    if group:
        out.extend(_resolve_group(group))
    return out


def _resolve_group(group: Sequence[CharSpan]) -> list[CharSpan]:
    if len(group) == 1:
        return list(group)
    acc: list[CharSpan] = []
    order = sorted(
        group,
        key=lambda s: (
            _KEEP_RISK_ORDER.get(s.label, len(_KEEP_RISK_ORDER)),
            -s.score,
            s.start,
            -(s.end - s.start),
            s.label,
        ),
    )
    for sp in order:
        acc.extend(_subtract(sp, acc))
    return sorted(acc, key=lambda s: (s.start, s.end, s.label))


def _has_complete_view(sp: CharSpan, chunks: Sequence[tuple[int, str]], skip: int, n_chars: int) -> bool:
    """``sp`` の範囲を「自分の切り詰められた端に触れない形で」丸ごと見られた別チャンクがあるか。

    あるなら、そのチャンクの判断（スパンを出した／O と判断した）の方が信頼できるので、
    端で切られた断片を採用する必要はない。無いなら誰も全体を見ていないので、断片を
    採らないと丸ごと漏れる（チャンクの切り詰め端にちょうど始まるエンティティ等）。
    """
    for ci, (off, piece) in enumerate(chunks):
        if ci == skip:
            continue
        end = off + len(piece)
        if sp.start < off or sp.end > end:
            continue
        left_ok = off == 0 or sp.start > off
        right_ok = end >= n_chars or sp.end < end
        if left_ok and right_ok:
            return True
    return False


FEATURES: tuple[str, ...] = ("span_mean", "span_ctx")
# span_ctx の文脈幅（スパンの左右それぞれのサブワード数）
CTX_TOKENS = 16


def _unit(v: "torch.Tensor") -> "torch.Tensor":
    norm = float(v.norm())
    return v / norm if norm > 0 else v


def span_vector(
    sp: CharSpan,
    chunks: Sequence[tuple[int, str]],
    feat_idx: Sequence[int],
    feats: Sequence[dict[str, Any]],
    local_hidden: Sequence["torch.Tensor | None"],
    features: str = "span_mean",
) -> bytes | None:
    """スパンを丸ごと含む最初のチャンク（無ければ最も重なるチャンク）で encoder の隠れ状態からスパンの特徴量を作り、
    float16 little-endian のバイト列にする。

    ``features="span_mean"``: スパンに重なるサブワードの平均を L2 正規化（従来どおり、256 次元）。
    ``features="span_ctx"``: [span_mean; スパン直前 ``CTX_TOKENS`` サブワードの平均; 直後 ``CTX_TOKENS`` サブワードの平均] を
    区画ごとに L2 正規化して連結（768 次元）。文脈は同じチャンクの、文字が乗っているサブワード（特殊トークンは除く）の範囲内だけ。
    チャンクの端で足りなければある分だけ、無ければゼロベクトル。

    スパンに重なるチャンクが無い場合、隠れ状態が無い場合、スパンに重なるサブワードが無い場合は None。
    """
    if features not in FEATURES:
        raise ValueError(f"unknown features {features!r} (use one of {FEATURES})")
    best: int | None = None
    best_overlap = 0
    for ci, (off, piece) in enumerate(chunks):
        lo, hi = max(sp.start, off), min(sp.end, off + len(piece))
        if hi <= lo:
            continue
        if off <= sp.start and sp.end <= off + len(piece):
            best = ci
            break
        if hi - lo > best_overlap:
            best, best_overlap = ci, hi - lo
    if best is None:
        return None
    hidden = local_hidden[feat_idx[best]]
    if hidden is None:
        return None
    off, piece = chunks[best]
    f = feats[feat_idx[best]]
    c2t, mask = f["char_to_token"], f["char_mask"]
    toks = sorted({c2t[li] for li in range(max(sp.start - off, 0), min(sp.end - off, len(piece))) if mask[li]})
    toks = [t for t in toks if t < hidden.shape[0]]
    if not toks:
        return None
    v = _unit(hidden[toks].float().mean(dim=0))
    if features == "span_mean":
        return v.numpy().astype("<f2").tobytes()
    content = sorted({c2t[li] for li in range(len(piece)) if mask[li]})
    content = [t for t in content if t < hidden.shape[0]]
    lo_t, hi_t = content[0], content[-1]
    left = list(range(max(lo_t, toks[0] - CTX_TOKENS), toks[0]))
    right = list(range(toks[-1] + 1, min(hi_t, toks[-1] + CTX_TOKENS) + 1))
    zero = torch.zeros_like(v)
    lv = _unit(hidden[left].float().mean(dim=0)) if left else zero
    rv = _unit(hidden[right].float().mean(dim=0)) if right else zero
    return torch.cat([v, lv, rv]).numpy().astype("<f2").tobytes()


def _unit_grad(v: torch.Tensor) -> torch.Tensor:
    """学習用の L2 正規化。forward の値は :func:`_unit` と同じ（ゼロベクトルはゼロのまま）。

    :func:`_unit` は ``float(v.norm())`` で分母を Python の数にするので、逆伝播では分母が定数扱いになり、
    正規化の射影項が勾配から抜ける。さらに GPU ではスパンごとに同期が起きる。ここではノルムをテンソルのまま
    割るので、勾配が分母にも流れ、同期も起きない。推論の :func:`span_vector` は保存済みベクトルのバイトを
    変えないよう従来どおり :func:`_unit` を使う。ノルムが 1e-12 未満の実質ゼロのベクトルは 1e-12 で割る
    （ゼロベクトルの勾配が NaN にならないため）。
    """
    return v / v.norm().clamp_min(1e-12)


def span_ctx_batch(
    token_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    char_to_token: torch.Tensor,
    char_mask: torch.Tensor,
    cat_spans: torch.Tensor,
    cat_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """学習用: batch 内のカテゴリスパンについて ``span_ctx`` 特徴量（3H）をまとめて計算する。

    規則は :func:`span_vector` の ``features="span_ctx"`` と同じ（スパン文字のサブワード平均、左右 ``CTX_TOKENS``
    サブワードの平均、各区画 L2 正規化、文脈は同じチャンクの文字が乗るサブワードの範囲内、無ければゼロ）。
    勾配は ``token_hidden`` に流れる。``cat_spans`` は ``(B, S, 4)`` = [start, end, kind, category]（文字位置はチャンク相対）、
    ``cat_mask`` が真のスパンだけを行優先 (b, s) の順に並べて返す。サブワードに重ならないスパンはゼロ行にし、
    2 つ目の戻り値 ``valid``（bool, (N,)）を偽にする（損失側で除く）。

    正規化は :func:`_unit_grad`（分母にも勾配が流れる）。計算は fp32 以上で行う（fp16 / bf16 は fp32 に上げ、
    float64 はそのまま。float64 は勾配の数値検査用）。添字側のテンソルは最初に 1 回だけ Python のリストにし、
    スパンごとの GPU 同期を起こさない。
    """
    hidden = token_hidden if token_hidden.dtype == torch.float64 else token_hidden.float()
    B, T, H = hidden.shape
    rows: list[torch.Tensor] = []
    valid: list[bool] = []
    zero = hidden.new_zeros(H)
    t_valid_all = attention_mask.sum(dim=1).tolist()
    c2t_all = char_to_token.tolist()
    cmask_all = char_mask.tolist()
    spans_all = cat_spans.tolist()
    smask_all = cat_mask.tolist()
    for b in range(B):
        t_valid = int(t_valid_all[b])
        c2t = c2t_all[b]
        cmask = cmask_all[b]
        n_chars = len(c2t)
        content = sorted({c2t[i] for i in range(n_chars) if cmask[i] and c2t[i] < t_valid})
        for s in range(len(spans_all[b])):
            if not smask_all[b][s]:
                continue
            start, end = int(spans_all[b][s][0]), int(spans_all[b][s][1])
            toks = sorted({c2t[i] for i in range(max(start, 0), min(end, n_chars)) if cmask[i] and c2t[i] < t_valid})
            if not toks or not content:
                rows.append(torch.cat([zero, zero, zero]))
                valid.append(False)
                continue
            lo_t, hi_t = content[0], content[-1]
            idx = torch.tensor(toks, device=hidden.device)
            v = _unit_grad(hidden[b, idx].mean(dim=0))
            left = list(range(max(lo_t, toks[0] - CTX_TOKENS), toks[0]))
            right = list(range(toks[-1] + 1, min(hi_t, toks[-1] + CTX_TOKENS) + 1))
            lv = _unit_grad(hidden[b, torch.tensor(left, device=hidden.device)].mean(dim=0)) if left else zero
            rv = _unit_grad(hidden[b, torch.tensor(right, device=hidden.device)].mean(dim=0)) if right else zero
            rows.append(torch.cat([v, lv, rv]))
            valid.append(True)
    if not rows:
        return hidden.new_zeros((0, 3 * H)), torch.zeros(0, dtype=torch.bool, device=hidden.device)
    return torch.stack(rows), torch.tensor(valid, dtype=torch.bool, device=hidden.device)


class CategoryAuxHead(nn.Module):
    """第 2 段の補助カテゴリ層: kind ごとの ``Linear(3H, k)``。学習時の損失にだけ使い、推論の保存形式には含めない。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleDict({kind: nn.Linear(3 * hidden_size, len(CATEGORY_NAMES[kind])) for kind in CATEGORY_KINDS})

    def forward(self, feats: torch.Tensor, kind_idx: torch.Tensor) -> list[torch.Tensor]:
        return [self.layers[kind](feats[kind_idx == i]) for i, kind in enumerate(CATEGORY_KINDS)]


def category_loss(
    aux: CategoryAuxHead,
    token_hidden: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    class_weights: Mapping[str, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, int]:
    """補助カテゴリ損失。batch の有効なカテゴリスパンについて kind ごとの cross entropy を計算し、
    スパン数で重み付けして 1 本に平均する。有効スパンが無ければ (0, 0)（勾配グラフだけ残る 0）。

    有効スパン 0 件の 0 は ``token_hidden`` の空スライスの和で作る（``token_hidden.sum() * 0`` だと
    fp16 の和が inf に溢れたとき NaN になる。``CharBioNerModel.forward`` の有効ラベル 0 件と同じ扱い）。
    """
    feats, valid = span_ctx_batch(token_hidden, batch["attention_mask"], batch["char_to_token"], batch["char_mask"],
                                  batch["cat_spans"], batch["cat_mask"])
    meta = batch["cat_spans"][batch["cat_mask"]]            # (N, 4) 行優先 = feats と同じ順
    if feats.shape[0] == 0 or not bool(valid.any()):
        return token_hidden.float().reshape(-1)[:0].sum(), 0
    feats, meta = feats[valid], meta[valid.to(meta.device)]
    total = feats.new_zeros(())
    n_total = 0
    for i, kind in enumerate(CATEGORY_KINDS):
        sel = meta[:, 2] == i
        n = int(sel.sum().item())
        if n == 0:
            continue
        w = class_weights.get(kind) if class_weights else None
        logits = aux.layers[kind](feats[sel.to(feats.device)])
        target = meta[sel, 3].to(logits.device)
        total = total + F.cross_entropy(
            logits, target, weight=(None if w is None else w.to(logits.device, torch.float)), reduction="sum"
        )
        n_total += n
    return total / n_total, n_total


def sqrt_class_weights(counts: Mapping[str, int], categories: Sequence[str], cap: float = 5.0) -> list[float]:
    """クラス頻度の平方根の逆数を平均 1 に正規化し ``cap`` で切る（`tools/train_category_head._class_weights(mode="sqrt")` と同値）。無いクラスは 0。"""
    inv = [(1.0 / math.sqrt(counts[c])) if counts.get(c, 0) > 0 else 0.0 for c in categories]
    total = sum(inv)
    k = len(categories)
    if total <= 0:
        return [1.0] * k
    return [min(w * k / total, cap) for w in inv]


def category_spans_by_text(
    labels: Iterable[Mapping[str, Any]],
    text_by_doc_id: Mapping[str, str],
    splits: Sequence[str] = ("head_train", "head_dev"),
) -> tuple[dict[str, list[tuple[int, int, int, int]]], dict[str, Any]]:
    """`category_labels_v1.jsonl` の行を本文に結合する（学習レコードは dedup・shuffle されるので本文で引く）。

    戻り値は (本文 → [(start, end, kind_idx, cat_idx)], stats)。stats は ``n_rows``（``splits`` に入る行数）、
    ``n_used``、``n_unjoined``（``doc_id`` の本文が無い行）、``by_kind``、``by_category``（``"kind/name"``）。
    ``splits`` に入らない行（head_judge など）は数えない。未知の kind / category は ``ValueError``。
    """
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    st: dict[str, Any] = {"n_rows": 0, "n_used": 0, "n_unjoined": 0, "by_kind": {}, "by_category": {}}
    for row in labels:
        if row.get("split") not in splits:
            continue
        st["n_rows"] += 1
        kind, cat = row["kind"], row["category"]
        if kind not in CATEGORY_INDEX or cat not in CATEGORY_INDEX[kind]:
            raise ValueError(f"unknown category {cat!r} for kind {kind!r}")
        text = text_by_doc_id.get(row["doc_id"])
        if text is None:
            st["n_unjoined"] += 1
            continue
        out.setdefault(text, []).append((int(row["start"]), int(row["end"]), CATEGORY_KIND_INDEX[kind], CATEGORY_INDEX[kind][cat]))
        st["n_used"] += 1
        st["by_kind"][kind] = st["by_kind"].get(kind, 0) + 1
        key = f"{kind}/{cat}"
        st["by_category"][key] = st["by_category"].get(key, 0) + 1
    return out, st


def vector_to_floats(data: bytes) -> tuple[float, ...]:
    """``span_vector`` のバイト列を float の列に戻す（float16 little-endian）。"""
    import struct

    n = len(data) // 2
    return struct.unpack(f"<{n}e", data[: n * 2])


def _chunk_and_encode(
    texts: Sequence[str],
    tokenizer,
    chunk_chars: int,
    chunk_overlap: int,
    max_length: int,
) -> tuple[
    list[dict[str, Any]],
    list[list[tuple[int, str]]],
    list[list[int]],
    list[list[tuple[int, int]]],
]:
    """全文書をチャンクに割って符号化し、``(feats, doc_chunks, doc_feats, cores)`` を返す
    （``predict_char_spans`` と ``encode_span_vectors`` で共有）。"""
    piece_cache: dict[str, int] = {}
    feats: list[dict[str, Any]] = []
    doc_chunks: list[list[tuple[int, str]]] = []
    doc_feats: list[list[int]] = []
    cores: list[list[tuple[int, int]]] = []
    for di, text in enumerate(texts):
        chunks = token_safe_chunks(text, tokenizer, chunk_chars, chunk_overlap, max_length)
        doc_chunks.append(chunks)
        cores.append(core_ranges(chunks, len(text)))
        doc_feats.append([])
        for off, piece in chunks:
            doc_feats[di].append(len(feats))
            feats.append(encode_chunk(piece, tokenizer, None, max_length, piece_cache, strict=True))
    return feats, doc_chunks, doc_feats, cores


def encode_span_vectors(
    texts: Sequence[str],
    spans_per_doc: Sequence[Sequence[tuple[int, int]]],
    model,
    tokenizer,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_length: int = MAX_LENGTH,
    features: str = "span_mean",
) -> list[list["bytes | None"]]:
    """与えられた ``(start, end)`` スパンの encoder ベクトル（``span_vector`` と同じ float16 LE）。

    ``predict_char_spans(vectors_out=...)`` が返すベクトルと同じ計算だが、スパンをモデルの
    予測からではなく呼び出し側から受け取る。計画 B の学習ツール（分類ヘッドの入力ベクトル
    作成）用。モデルが ``token_hidden`` を返さない場合はすべて ``None``。``features`` は
    ``FEATURES`` のどれか（``span_vector`` 参照）。
    """
    if len(texts) != len(spans_per_doc):
        raise ValueError("texts and spans_per_doc must have the same length")
    if features not in FEATURES:
        raise ValueError(f"unknown features {features!r} (use one of {FEATURES})")
    model.eval()
    collate = CharNerCollator(tokenizer.pad_token_id or 0)
    feats, doc_chunks, doc_feats, _ = _chunk_and_encode(texts, tokenizer, chunk_chars, chunk_overlap, max_length)

    local_hidden: list[torch.Tensor | None] = [None] * len(feats)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idxs = order[s : s + batch_size]
            batch = collate([feats[i] for i in idxs])
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model(**batch)
            hidden = result.get("token_hidden") if isinstance(result, dict) else None
            if hidden is None:
                continue
            hidden_cpu = hidden.detach().to("cpu", torch.float16)
            for row, fi in enumerate(idxs):
                local_hidden[fi] = hidden_cpu[row]

    out: list[list[bytes | None]] = []
    for di, spans in enumerate(spans_per_doc):
        out.append(
            [
                span_vector(CharSpan(s, e, "X"), doc_chunks[di], doc_feats[di], feats, local_hidden, features=features)
                for s, e in spans
            ]
        )
    return out


@torch.no_grad()
def predict_char_spans(
    texts: Sequence[str],
    model: CharBioNerModel,
    tokenizer,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_length: int = MAX_LENGTH,
    threshold: float = 0.0,
    merge_adjacent: bool = False,
    *,
    merge_strategy: str = "safe_union",
    suppress_seam_b: bool = False,
    fallback_out: list[tuple[int, CharSpan]] | None = None,
    warn_fallback: bool = False,
    vectors_out: list[list[bytes | None]] | None = None,
    vector_features: str = "span_mean",
) -> list[list[CharSpan]]:
    """文書ごとに文字スパンを返す。確率平均はしない。

    ``merge_strategy``
        ``"safe_union"``（既定）
            core 割り当ての結果を土台に、各チャンクの「自分の切り詰められた端に接して
            いないスパン」＝エンティティ全体が見えたうえでの判断を文字単位で和集合に
            する。エンティティ全体が見えているチャンクが 1 つでも正しく出せば、その
            全体が残るので、core 方式の「担当チャンクが端で切られていて検出できず、
            別チャンクの正しいスパンの一部だけが採用される」漏れ（``core_ranges`` の
            反例）が起きない。PII マスキングでは漏れ > 過剰マスクなので既定にした。
            代償は 2 つある。(1) チャンクが左文脈不足で出した偽陽性も採用されうる。
            (2) あるチャンクが実体内部の B- を落として 1 本の長いスパンを出すと、
            別チャンクが正しく 2 本に分けていても重なりで結合される。人工モデルでの
            実測ではどちらも「漏れの減り方」に比べて小さいが、実データでは未評価。
        ``"core"``
            旧方式。各文字の予測を担当チャンク 1 つから取る。後方互換・比較用。
            **エンティティ全体の保持は保証しない**（``core_ranges`` 参照）。

    ``warn_fallback=True`` にすると、その fallback が起きた文書で ``UserWarning`` を出す。
    既定で出さないのは、ランダム系の重みでは 4% 程度の文書で発火してバッチ処理のログを
    埋めるため。情報が要るなら ``fallback_out`` を渡す（診断はどちらも要求されたときだけ
    計算する）。

    ``fallback_out`` に list を渡すと、「全体が見えたチャンク」の裏付けが無く core 割り当て
    だけで拾った部分が ``(doc_idx, CharSpan)`` で入る（超長エンティティが全チャンクの端に
    掛かる場合など）。分割幅より長いエンティティが疑われるサイン。

    ``suppress_seam_b=True`` は ``"core"`` 用の旧デコード挙動（``decode_char_labels``）。
    ``merge_adjacent=True`` なら最後に接する同型スパンをまとめる。

    ``vectors_out`` に list を渡すと、文書ごと・返すスパンごとに encoder の隠れ状態（モデルが
    ``token_hidden`` を返す場合）をスパンに重なるサブワードで平均し L2 正規化した float16 LE の
    バイト列を追加する（``span_vector``）。モデルが ``token_hidden`` を返さなければ ``None``。
    ``vector_features`` で特徴量を選ぶ（既定 ``span_mean``、``FEATURES`` 参照）。
    """
    if merge_strategy not in MERGE_STRATEGIES:
        raise ValueError(f"merge_strategy は {MERGE_STRATEGIES} のいずれか: {merge_strategy!r}")
    if vector_features not in FEATURES:
        raise ValueError(f"unknown features {vector_features!r} (use one of {FEATURES})")
    model.eval()
    collate = CharNerCollator(tokenizer.pad_token_id or 0)

    # 全文書のチャンクを 1 本に並べてからバッチ化する（長さ順ソートで padding を削る）
    feats, doc_chunks, doc_feats, cores = _chunk_and_encode(texts, tokenizer, chunk_chars, chunk_overlap, max_length)

    # チャンクごとの局所予測。safe_union は core 外の判断も使うので全位置を保持する。
    local_types: list[list[int]] = [[] for _ in feats]
    local_conf: list[list[float]] = [[] for _ in feats]
    local_isb: list[list[bool]] = [[] for _ in feats]
    local_hidden: list[torch.Tensor | None] = [None] * len(feats)

    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    for s in range(0, len(order), batch_size):
        idxs = order[s : s + batch_size]
        batch = collate([feats[i] for i in idxs])
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(**batch)
        logits = result["logits"]
        hidden_cpu: torch.Tensor | None = None
        if vectors_out is not None and isinstance(result, dict) and result.get("token_hidden") is not None:
            hidden_cpu = result["token_hidden"].detach().to("cpu", torch.float16)
        # fail-closed: 有効文字位置に NaN/Inf があると argmax が 0(=O) を返して
        # 「エンティティなし」が成功として返る。softmax 前に弾く。padding は検査しない。
        valid = batch["char_mask"].bool()
        if bool(valid.any()) and not bool(torch.isfinite(logits[valid]).all()):
            raise FloatingPointError(
                "文字NERの有効文字位置に NaN または Inf があります（モデル出力の数値異常）"
            )
        probs = torch.softmax(logits.float(), dim=-1)
        # 文字ごとに argmax を呼ぶのではなく batch 単位でまとめて取る
        best_scores, best_labels = probs.max(dim=-1)
        best_scores = best_scores.cpu().tolist()
        best_labels = best_labels.cpu().tolist()
        for row, fi in enumerate(idxs):
            nb = len(feats[fi]["char_to_token"])
            row_lab = best_labels[row]
            row_sc = best_scores[row]
            t_ids = [0] * nb
            c_sc = [0.0] * nb
            b_fl = [False] * nb
            for li in range(nb):
                lab = row_lab[li]
                sc = row_sc[li]
                if lab != 0 and sc >= threshold:
                    name = LABEL_LIST[lab]
                    t_ids[li] = ENTITY_TYPES.index(name[2:]) + 1
                    b_fl[li] = name.startswith("B-")
                c_sc[li] = sc
            local_types[fi] = t_ids
            local_conf[fi] = c_sc
            local_isb[fi] = b_fl
            if hidden_cpu is not None:
                local_hidden[fi] = hidden_cpu[row]

    out: list[list[CharSpan]] = []
    for di, text in enumerate(texts):
        n = len(text)
        chunks = doc_chunks[di]

        # core 割り当て: 各文字を担当チャンク 1 つから取る。safe_union でも土台に使う。
        type_ids = [0] * n
        conf = [0.0] * n
        is_b = [False] * n
        seam = [False] * n
        filled = [False] * n
        for ci, fi in enumerate(doc_feats[di]):
            off = chunks[ci][0]
            c_from, c_to = cores[di][ci]
            nb = len(local_types[fi])
            for gi in range(c_from, c_to):
                li = gi - off
                if not (0 <= li < nb):
                    continue
                type_ids[gi] = local_types[fi][li]
                conf[gi] = local_conf[fi][li]
                is_b[gi] = local_isb[fi][li]
                filled[gi] = True
            if c_from > 0:
                seam[c_from] = True
        _check_filled(di, text, filled)
        spans = decode_char_labels(
            text, type_ids, conf, is_b, seam, suppress_seam_b=suppress_seam_b
        )

        if merge_strategy == "safe_union":
            # 各チャンクの「自分の切り詰められた端に接していないスパン」= エンティティ
            # 全体が見えたうえでの判断だけを寄与させ、core の結果と文字単位で和集合にする。
            # core を土台に残すのは (1) どのチャンクも全体を見られない領域（超長エンティティが
            # 全チャンクの端に掛かる等）を落とさないため、(2) チャンク先頭の左文脈不足な
            # 領域だけを寄与にすると断片が増えるため（実測: 偽 B を出すモデルで断片 5 → 1）。
            whole: list[CharSpan] = []
            orphan: list[CharSpan] = []
            for ci, fi in enumerate(doc_feats[di]):
                off, piece = chunks[ci]
                cut_left = off > 0
                cut_right = off + len(piece) < n
                for sp in decode_char_labels(
                    piece, local_types[fi], local_conf[fi], local_isb[fi]
                ):
                    g = CharSpan(sp.start + off, sp.end + off, sp.label, sp.score)
                    if (cut_left and sp.start == 0) or (cut_right and sp.end == len(piece)):
                        # 端で切られた判断は普通は採らないが、他のどのチャンクもこの範囲を
                        # 丸ごと見られていないなら、採らないと丸ごと漏れる
                        if not _has_complete_view(g, chunks, ci, n):
                            orphan.append(g)
                        continue
                    whole.append(g)
            union = _resolve_type_conflicts(_merge_overlapping_same_type(spans + whole + orphan))
            # 「全体が見えたチャンク」の裏付けが無く core だけが支えている部分を報告する。
            # 診断なので、要求されていないときは計算しない（下の _uncovered_runs は線形だが、
            # 素朴な区間差分は O(|union| x |whole|) で 16000 文字の文書で 6.5 秒かかった）。
            if fallback_out is not None or warn_fallback:
                gaps = _uncovered_runs(union, whole, n)
                if gaps:
                    if fallback_out is not None:
                        fallback_out.extend((di, sp) for sp in gaps)
                    if warn_fallback:
                        warnings.warn(
                            f"doc[{di}]: 全体が見えたチャンクが無く core 割り当てだけで拾った"
                            f"スパンが {len(gaps)} 件"
                            f"（文字数 {sum(sp.end - sp.start for sp in gaps)}）。"
                            " 分割幅・overlap の見直しを検討してください",
                            stacklevel=2,
                        )
            spans = union

        final = merge_adjacent_same_type(spans) if merge_adjacent else spans
        out.append(final)
        if vectors_out is not None:
            vectors_out.append(
                [
                    span_vector(sp, chunks, doc_feats[di], feats, local_hidden, features=vector_features)
                    for sp in final
                ]
            )
    return out


def _uncovered_runs(spans: Sequence[CharSpan], covered_by: Sequence[CharSpan], n_chars: int) -> list[CharSpan]:
    """``spans`` のうち ``covered_by`` に覆われていない連続区間を返す。

    区間差分を素朴に二重ループで書くと O(|spans| x |covered_by|) になり、
    予測スパンが多い文書（未学習チェックポイント等）で支配的なコストになる
    （実測: 16000 文字で 6.58 秒、うち 99% がこの差分）。文書長の bool 配列を
    1 度作って走査すれば文字数に線形になる。
    """
    if not spans:
        return []
    covered = bytearray(n_chars)
    for sp in covered_by:
        a = max(0, min(sp.start, n_chars))
        b = max(0, min(sp.end, n_chars))
        if b > a:
            covered[a:b] = b"\x01" * (b - a)
    out: list[CharSpan] = []
    for sp in spans:
        i = max(0, sp.start)
        end = min(sp.end, n_chars)
        while i < end:
            if covered[i]:
                i += 1
                continue
            j = i + 1
            while j < end and not covered[j]:
                j += 1
            out.append(CharSpan(i, j, sp.label, sp.score))
            i = j
    return out


def _check_filled(di: int, text: str, filled: Sequence[bool]) -> None:
    """1 文字でも予測が入っていなければ落とす（黙って O にしない）。"""
    if text and not all(filled):
        miss = sum(1 for f in filled if not f)
        raise CharCoverageError(f"doc[{di}]: {miss}/{len(text)} 文字が未予測（チャンク統合の不整合）")


# ---------------------------------------------------------------------------
# 保存 / 読み込み（encoder は HF 標準形式、ヘッドは別ファイル）
# ---------------------------------------------------------------------------

# save_char_model が自分で書くキー。extra で上書きさせない。
_RESERVED_CONFIG_KEYS: frozenset[str] = frozenset({
    "char_dim", "kernel_size", "use_mix", "num_labels", "label_list",
    "n_intra", "n_script", "added_parameters", "dropout", "class_weights",
})

HEAD_WEIGHTS_NAME = "char_head.safetensors"
HEAD_CONFIG_NAME = "char_head_config.json"


def save_char_model(model: CharBioNerModel, save_dir: str, tokenizer=None, extra: dict | None = None) -> None:
    """encoder を HF 標準形式で、ヘッドを別ファイルで保存する。

    encoder が ``AutoModel.from_pretrained`` で普通に読めるので、
    既存の token 分類モデルと同じ運用（任意のフォルダに置いてパス指定、または Hub のリポジトリ ID を指定）がそのまま使える。

    保存対象は **推論に必要な重みと構成、および学習設定（dropout 率・class_weights）だけ**。
    optimizer / scheduler / GradScaler / RNG の状態は保存しないので、これだけでは
    厳密な学習再開（同じ更新軌跡の再現）はできない。中断からの再開が必要なら
    別途チェックポイントを取ること。
    """
    import json
    import os

    from safetensors.torch import save_file

    os.makedirs(save_dir, exist_ok=True)
    model.encoder.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    head_sd = {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}
    save_file(head_sd, os.path.join(save_dir, HEAD_WEIGHTS_NAME))
    cfg = {
        "char_dim": model.head.char_dim,
        "kernel_size": model.head.kernel_size,
        "use_mix": model.head.use_mix,
        "num_labels": model.num_labels,
        "label_list": LABEL_LIST,
        "n_intra": N_INTRA,
        "n_script": N_SCRIPT,
        "added_parameters": model.num_added_parameters(),
        # 以下は推論には不要だが、記録しないと継続学習で元の設定に戻せない。
        # dropout は state_dict に出ないため、保存し忘れると復元時に既定値へ化ける。
        "dropout": float(model.head.dropout.p),
        "class_weights": (
            None if model.class_weights is None else [float(v) for v in model.class_weights.tolist()]
        ),
    }
    if extra:
        # 学習設定を extra で黙って上書きできると、復元時に誤った dropout /
        # class_weights が無警告で使われる（レビュー文書が避けたかった「既定値へ黙って
        # 化ける」の別経路）。衝突は明示的に落とす。
        clash = sorted(set(extra) & _RESERVED_CONFIG_KEYS)
        if clash:
            raise ValueError(
                f"extra に予約キー {clash} を入れられません（保存される学習設定を"
                " 上書きしてしまうため）"
            )
        cfg.update(extra)
    with open(os.path.join(save_dir, HEAD_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_char_model(
    save_dir: str,
    device: str | torch.device = "cpu",
    *,
    for_training: bool = False,
    dropout: float | None = None,
    class_weights: Sequence[float] | None = None,
    **encoder_kwargs: Any,
) -> CharBioNerModel:
    """``save_char_model`` で保存したディレクトリから復元する。

    既定（``for_training=False``）は **推論専用の復元** である。dropout=0・``eval()``・
    class_weights なしで返すので、同じ入力に対して保存前と同一の出力になる。

    ``for_training=True`` は継続学習向けで、保存された dropout 率と class_weights を
    戻して ``train()`` で返す。ただし optimizer / scheduler / GradScaler / RNG の状態は
    ``save_char_model`` が保存しないので、**厳密な学習再開（同じ更新軌跡の再現）はできない**。
    復元されるのはモデル重みと学習設定までである。

    ``dropout`` / ``class_weights`` は保存された値を上書きする明示指定用。
    古い形式（これらが ``char_head_config.json`` に無い）を ``for_training=True`` で
    読むときは、既定値へ黙って化けるのを避けるため明示指定を要求する。
    """
    import json
    import os

    from safetensors.torch import load_file

    cfg_path = os.path.join(save_dir, HEAD_CONFIG_NAME)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{HEAD_CONFIG_NAME} がありません: {save_dir}\n"
            "  → 文字ヘッド付きモデルではなく token 分類モデルを指している可能性があります"
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("label_list") and list(cfg["label_list"]) != LABEL_LIST:
        raise ValueError(f"ラベル空間が不一致です: 保存={cfg['label_list']} / 現在={LABEL_LIST}")

    if for_training:
        if dropout is None:
            if cfg.get("dropout") is None:
                raise ValueError(
                    f"{HEAD_CONFIG_NAME} に dropout が記録されていません（古い形式の保存物）。\n"
                    "  → 学習時の値を dropout= で明示してください"
                    "（推論だけなら for_training=False で読めます）"
                )
            dropout = float(cfg["dropout"])
        if class_weights is None:
            class_weights = cfg.get("class_weights")   # 記録が無い/None なら重みなしで学習
    else:
        # 推論経路で dropout を有効にすると出力が非決定的になるため、指定自体を拒否する。
        if dropout is not None or class_weights is not None:
            raise ValueError(
                "dropout / class_weights は for_training=True のときだけ指定できます"
                "（既定は推論専用の復元です）"
            )
        dropout = 0.0

    model = CharBioNerModel.from_pretrained_encoder(
        save_dir,
        num_labels=int(cfg.get("num_labels", NUM_LABELS)),
        char_dim=int(cfg.get("char_dim", 128)),
        kernel_size=int(cfg.get("kernel_size", 5)),
        dropout=float(dropout),
        use_mix=bool(cfg.get("use_mix", True)),
        class_weights=class_weights,
        **encoder_kwargs,
    )
    model.head.load_state_dict(load_file(os.path.join(save_dir, HEAD_WEIGHTS_NAME)), strict=True)
    model = model.to(device)
    return model.train() if for_training else model.eval()

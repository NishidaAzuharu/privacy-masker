"""学習データ v7 の新規文書生成器（tools/build_data_v7.py）。Drive を読まない。"""
import os
import random

import pytest
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from build_data_v7 import (  # noqa: E402
    BUSINESS_LINES,
    CHAT_UTTERANCES,
    EVAL_SURFACES,
    GIVEN_KATA_JA,
    NARRATIVE_FILLERS,
    SURNAMES_KATA_JA,
    check_no_eval_surfaces,
    make_chat_docs,
    make_dialogue_docs,
    make_halfwidth_docs,
    to_halfwidth_kana,
)

LATIN_G = ["Maria", "Kenan", "Robert"]
LATIN_S = ["Ivers", "Szanto", "Normark"]


def spans_ok(docs):
    for d in docs:
        t = d["text"]
        prev = -1
        for s, e, l in d["entities"]:
            assert 0 <= s < e <= len(t), (s, e, len(t))
            assert s >= prev, "span が重なっている"
            prev = e
            assert t[s:e] == t[s:e].strip(), "span の端に空白がある"
            assert l == "PERSON"


# ------------------------------------------------------------ 半角カナ

def test_halfwidth_conversion_round_trips_via_nfkc():
    for w in ("ハマグチ", "タナカ", "ジョン・スミス", "ヴィヴィアン", "パーク", "ハセガワ", "セキ", "ソノダ"):
        hw = to_halfwidth_kana(w)
        assert all(0xFF61 <= ord(c) <= 0xFF9F for c in hw), hw     # 全部半角カナ領域（全角が残っていない）
        assert unicodedata.normalize("NFKC", hw) == w


def test_halfwidth_docs_have_halfwidth_person_spans():
    docs = make_halfwidth_docs(40, random.Random(1), BUSINESS_LINES)
    spans_ok(docs)
    assert len(docs) == 40
    for d in docs:
        assert d["entities"], "半角カナ名が 1 つも無い文書"
        for s, e, l in d["entities"]:
            surf = d["text"][s:e]
            assert re.search(r"[ｦ-ﾟ]", surf), surf


# ------------------------------------------------------------ チャットログ

def test_chat_docs_annotate_line_start_handles():
    docs = make_chat_docs(60, random.Random(2), CHAT_UTTERANCES, LATIN_G, LATIN_S)
    spans_ok(docs)
    styles = {d["variant"] for d in docs}
    assert {"chat:colon", "chat:time"} <= styles
    for d in docs:
        t = d["text"]
        for s, e, l in d["entities"]:
            surf = t[s:e]
            assert "@" not in surf[:1], "@ を span に含めない"
            # 行頭か、「@」の直後か
            assert s == 0 or t[s - 1] in "\n[@", (surf, repr(t[max(0, s - 3):s]))


def test_chat_docs_include_mixed_handles():
    docs = make_chat_docs(200, random.Random(3), CHAT_UTTERANCES, LATIN_G, LATIN_S)
    surfs = {d["text"][s:e] for d in docs for s, e, _ in d["entities"]}
    assert any("@" in x for x in surfs), "かな@出品 型のハンドルが無い"
    assert any(re.fullmatch(r"[a-z]+[._-][a-z]+", x) for x in surfs), "英字スネーク型が無い"
    assert any(re.fullmatch(r"[ぁ-ゖ]+", x) for x in surfs), "かなハンドルが無い"


# ------------------------------------------------------------ 会話・物語文

def test_dialogue_docs_exclude_honorifics_from_spans():
    docs = make_dialogue_docs(80, random.Random(4), NARRATIVE_FILLERS, CHAT_UTTERANCES)
    spans_ok(docs)
    for d in docs:
        t = d["text"]
        for s, e, l in d["entities"]:
            assert not t[s:e].endswith(("ちゃん", "さん", "くん", "氏", "先生", "様")), t[s:e]
            assert len(t[s:e]) <= 3, "裸の短い名だけを作る"


def test_dialogue_docs_contain_one_char_names():
    docs = make_dialogue_docs(200, random.Random(5), NARRATIVE_FILLERS, CHAT_UTTERANCES)
    lens = {e - s for d in docs for s, e, _ in d["entities"]}
    assert 1 in lens and 2 in lens


# ------------------------------------------------------------ 評価セットの表層

def test_generated_docs_do_not_contain_blind_set_surfaces():
    rng = random.Random(6)
    docs = (make_chat_docs(300, rng, CHAT_UTTERANCES, LATIN_G, LATIN_S)
            + make_dialogue_docs(300, rng, NARRATIVE_FILLERS, CHAT_UTTERANCES)
            + make_halfwidth_docs(100, rng, BUSINESS_LINES))
    assert check_no_eval_surfaces(docs) == []
    assert not (set(SURNAMES_KATA_JA) & EVAL_SURFACES)
    assert not ({to_halfwidth_kana(s) for s in SURNAMES_KATA_JA} & EVAL_SURFACES)
    assert not (set(GIVEN_KATA_JA) & EVAL_SURFACES)
    # 盲検でハンドル名として出た平仮名名は、ハンドル名の学習例に使わない
    from build_data_v7 import EVAL_HANDLE_KANA, KANA_NAMES
    assert not (set(KANA_NAMES) & EVAL_HANDLE_KANA)
    handles = {d["text"][s:e] for d in docs if d["source"] == "synth_chat" for s, e, _ in d["entities"]}
    assert not (handles & EVAL_HANDLE_KANA)


def test_check_no_eval_surfaces_detects_leak():
    # 盲検の表層はテストに平文で書かない（tests/test_no_eval_surfaces.py）。正本の集合から決定的に 1 件取る
    if not EVAL_SURFACES:
        pytest.skip("盲検表層ファイル（データフォルダ）が無い環境")
    surface = sorted(EVAL_SURFACES)[0]
    bad = {"text": f"{surface}: こんにちは", "entities": [[0, len(surface), "PERSON"]], "source": "x", "variant": "x"}
    assert check_no_eval_surfaces([bad]) == [surface]

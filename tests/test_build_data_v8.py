"""学習データ v8 の帳票・ログ生成器と負例の結合（tools/build_data_v8.py）。Drive を読まない。"""
import os
import random
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from build_data_v8 import (  # noqa: E402
    EXTRA_SURNAMES,
    FORM_SOURCE,
    LLM_NEGATIVE_SOURCE,
    NEG_FORM_VALUES,
    NEG_LOG_VALUES,
    check_no_eval_person_surfaces,
    eval_person_exclusions,
    make_form_docs,
    split_negatives,
    surname_bank,
    without_llm_negatives,
)
from build_data_v7 import EVAL_SURFACES, check_no_eval_surfaces  # noqa: E402

GIVEN = ["太郎", "花子", "健一", "美咲"]


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


def test_eval_person_exclusions_cover_surface_first_part_and_prefix():
    ex = eval_person_exclusions({"蒼波", "朱鷺野", "久遠寺ミチル", "山田 太郎", "雪見沢"})
    assert {"蒼波", "朱鷺野", "雪見沢", "久遠寺ミチル", "山田"} <= ex
    tr, va = surname_bank(ex, random.Random(0))
    bank = set(tr) | set(va)
    assert "山田" not in bank and "蒼波" not in bank
    # 3 文字以上の姓は前方一致でも除く（久遠寺ミチル → 久遠寺）
    tr2, va2 = surname_bank(eval_person_exclusions({"久遠寺ミチル"}) | set(), random.Random(0))
    assert "久遠寺" not in set(tr2) | set(va2)
    assert not set(tr) & set(va) and len(bank) > 200
    assert all(s in bank or s in ex for s in EXTRA_SURNAMES)


def test_form_docs_spans_sources_and_contrast():
    rng = random.Random(3)
    surnames = ["田中", "鈴木", "生駒", "綿貫", "早乙女"]
    docs = make_form_docs(300, rng, surnames, GIVEN)
    spans_ok(docs)
    assert len(docs) == 300 and all(d["source"] == FORM_SOURCE for d in docs)
    variants = Counter(d["variant"].split(":")[1] for d in docs)
    assert set(variants) <= {"log", "form", "mixed"} and len(variants) == 3
    pure_neg = [d for d in docs if d["variant"].endswith(":neg")]
    assert all(not d["entities"] for d in pure_neg)
    assert 0.15 <= len(pure_neg) / len(docs) <= 0.40
    pos = [d for d in docs if d["variant"].endswith(":pos")]
    assert all(d["entities"] for d in pos), "正例文書に名が無い"
    # 同じフレームに役割語・状態語の値も出る（対の負例）
    neg_vals = set(NEG_LOG_VALUES) | set(NEG_FORM_VALUES)
    assert sum(1 for d in pos if any(v in d["text"] for v in neg_vals)) >= len(pos) * 0.3
    # 名の表層: 裸の姓が中心、姓+名・カタカナ・半角カナも出る
    surfaces = [d["text"][s:e] for d in docs for s, e, _ in d["entities"]]
    bare = sum(1 for x in surfaces if x in surnames)
    assert bare >= len(surfaces) * 0.55
    assert any(re.search(r"[ｦ-ﾟ]", x) for x in surfaces), "半角カナ姓が無い"
    assert any(re.fullmatch(r"[ァ-ヶー]+", x) for x in surfaces), "カタカナ姓が無い"
    assert any(any(g in x for g in GIVEN) for x in surfaces), "姓 名 の形が無い"
    # フレームの痕跡が残っていない
    assert not any("{" in d["text"] or "}" in d["text"] for d in docs)
    # key=value と ラベル: 値 の両方が出る
    assert any(re.search(r"(operator|user|approved_by|checked_by|owner|assignee)=", d["text"]) for d in docs)
    assert any(re.search(r"(記録者|担当|作成者|承認)[:：　 ]", d["text"]) for d in docs)


def test_form_docs_respect_exclusions_and_blind_surfaces():
    rng = random.Random(5)
    # カタカナ姓・半角カナ姓・姓名の組み合わせも除外に当たったら作らない
    ex = eval_person_exclusions({"蒼波", "朱鷺野", "雪見沢", "タカハシ", "サトウ", "佐藤 太郎", "佐藤太郎"})
    tr, va = surname_bank(ex, rng)
    docs = make_form_docs(400, rng, tr, GIVEN, exclusions=ex)
    assert check_no_eval_person_surfaces(docs, ex) == []
    surfaces = {unicodedata.normalize("NFKC", d["text"][s:e]) for d in docs for s, e, _ in d["entities"]}
    assert "タカハシ" not in surfaces and "サトウ" not in surfaces
    # 除外なしなら半角カナの ﾀｷﾞ… も NFKC で照合されて検出される
    bad_docs = [{"text": "担当: ﾀｶﾊｼ", "entities": [[4, 8, "PERSON"]]}]
    assert check_no_eval_person_surfaces(bad_docs, ex) == ["ﾀｶﾊｼ"]
    assert check_no_eval_surfaces(docs) == []
    assert not any(s in d["text"] for d in docs for s in EVAL_SURFACES)
    # 姓の候補は渡した bank の中だけ（valid 用 bank の姓は出ない）
    kanji = {d["text"][s:e] for d in docs for s, e, _ in d["entities"] if re.fullmatch(r"[一-龥]{1,4}", d["text"][s:e])}
    assert not kanji & set(va)


def test_halfwidth_surname_round_trips():
    rng = random.Random(9)
    docs = make_form_docs(150, rng, ["佐藤"], GIVEN)
    hw = [d["text"][s:e] for d in docs for s, e, _ in d["entities"] if re.search(r"[ｦ-ﾟ]", d["text"][s:e])]
    assert hw
    for x in hw:
        assert re.fullmatch(r"[ァ-ヶー]+", unicodedata.normalize("NFKC", x)), x


def test_split_negatives_and_noneg_variant():
    negs = [{"text": f"負例 {i} の本文。", "entities": [], "source": LLM_NEGATIVE_SOURCE, "variant": "x"} for i in range(100)]
    tr, va = split_negatives(negs, random.Random(1))
    assert len(tr) == 90 and len(va) == 10 and not {r["text"] for r in tr} & {r["text"] for r in va}
    tr2, va2 = split_negatives(negs, random.Random(1))
    assert [r["text"] for r in tr2] == [r["text"] for r in tr]
    others = [{"text": "正例", "entities": [[0, 2, "PERSON"]], "source": "synth_v5c"},
              {"text": "帳票の純負例", "entities": [], "source": FORM_SOURCE}]
    kept = without_llm_negatives(others + tr)
    assert kept == others, "LLM 負例だけを除き、帳票の純負例は残す"

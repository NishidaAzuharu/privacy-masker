"""学習データ v10 の比率調整（tools/build_data_v10.py）。Drive を読まない。"""
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from build_data_v10 import EXTRA_SURNAME_WEIGHT, name_surface_v10, surname_bank_v10  # noqa: E402
from build_data_v8 import EXTRA_SURNAMES, eval_person_exclusions  # noqa: E402
from build_data_v9 import make_form_docs_v9  # noqa: E402
from privacy_masker.data.ja_pools import SURNAMES_KANJI  # noqa: E402

GIVEN = ["太郎", "花子"]


def test_surname_bank_v10_weights_extra_and_splits_by_kind():
    ex = eval_person_exclusions({"蒼波", "朱鷺野", "雪見沢"})
    tr, va = surname_bank_v10(ex, random.Random(0))
    assert not set(tr) & set(va), "同じ姓が train と valid の両方にある"
    c = Counter(tr)
    extra_in = [s for s in EXTRA_SURNAMES if s in c]
    common_in = [s for s in SURNAMES_KANJI if s in c and s not in EXTRA_SURNAMES]
    assert extra_in and common_in
    assert all(c[s] == EXTRA_SURNAME_WEIGHT for s in extra_in) and all(c[s] == 1 for s in common_in)
    assert "蒼波" not in c and "朱鷺野" not in c and "雪見沢" not in c


def test_name_surface_v10_has_spaced_surnames_and_respects_exclusions():
    rng = random.Random(3)
    ex = {"田中", "佐藤 太郎"}
    surfaces = [name_surface_v10(rng, ["田中", "生駒", "綿貫", "早乙女"], GIVEN, exclusions=ex) for _ in range(3000)]
    assert "田中" not in surfaces and "田 中" not in surfaces and "田　中" not in surfaces
    spaced = [s for s in surfaces if re.fullmatch(r"[一-龥]([ 　][一-龥])+", s)]
    assert 0.02 <= len(spaced) / len(surfaces) <= 0.09, len(spaced) / len(surfaces)
    bare = sum(1 for s in surfaces if s in ("生駒", "綿貫", "早乙女"))
    assert bare / len(surfaces) > 0.55


def test_form_docs_with_v10_name_fn_and_ratio():
    rng = random.Random(5)
    docs = make_form_docs_v9(300, rng, ["生駒", "綿貫", "早乙女", "各務"], GIVEN, ["Maria"], ["Ivers"],
                             name_slot_ratio=0.85, name_fn=name_surface_v10, email_line_ratio=0.45)
    for d in docs:
        t = d["text"]
        for s, e, l in d["entities"]:
            assert l == "PERSON" and 0 <= s < e <= len(t) and t[s:e] == t[s:e].strip()
    surfaces = [d["text"][s:e] for d in docs for s, e, _ in d["entities"]]
    assert any(re.fullmatch(r"[一-龥][ 　][一-龥]", x) for x in surfaces), "字間空白の姓が無い"
    # 名スロット比率 85%: 正例文書の中で役割語・状態語が値になる行は少なめ
    pos = [d for d in docs if d["variant"].endswith(":pos")]
    with_neg_value = sum(1 for d in pos if any(v in d["text"] for v in ("担当者未定", "夜勤担当", "admin", "batch")))
    assert with_neg_value / len(pos) < 0.5

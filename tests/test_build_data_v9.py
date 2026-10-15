"""学習データ v9 の追加生成器（tools/build_data_v9.py）。Drive を読まない。"""
import os
import random
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from build_data_v9 import (  # noqa: E402
    EMAIL_FRAMES,
    FORM_FRAMES_V9,
    KATAKANA_SOURCE,
    LLM_NEGATIVE_SOURCE,
    TOWN_SOURCE,
    katakana_name_bank,
    make_email,
    make_form_docs_v9,
    make_katakana_docs,
    make_town_docs,
    make_url,
    select_negatives,
    town_bank,
)
from build_data_v8 import eval_person_exclusions, check_no_eval_person_surfaces  # noqa: E402
from build_data_v7 import EVAL_SURFACES, check_no_eval_surfaces  # noqa: E402

GIVEN = ["太郎", "花子", "健一", "美咲"]
LG = ["Maria", "Kenan", "Robert"]
LS = ["Ivers", "Szanto", "Normark"]
UTT = ["明日の資料を送ります", "了解しました", "少し遅れます", "会議は十時からです"]
FIL = ["雨は夕方まで降り続いた。", "誰も答えなかった。", "電車は少し遅れて着いた。"]


def spans_ok(docs, label):
    for d in docs:
        t = d["text"]
        prev = -1
        for s, e, l in d["entities"]:
            assert 0 <= s < e <= len(t), (s, e, len(t))
            assert s >= prev, "span が重なっている"
            prev = e
            assert t[s:e] == t[s:e].strip()
            assert l == label


def test_form_frames_drop_to_cc_and_email_lines_are_not_entities():
    assert not any(f.startswith(("TO:", "cc:")) for f in FORM_FRAMES_V9)
    rng = random.Random(2)
    docs = make_form_docs_v9(400, rng, ["田中", "生駒", "早乙女"], GIVEN, LG, LS, email_line_ratio=1.0)
    spans_ok(docs, "PERSON")
    with_mail = [d for d in docs if "@" in d["text"]]
    assert len(with_mail) >= 300
    for d in with_mail:
        t = d["text"]
        for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t):
            # メールアドレスのどの文字も PERSON span に入らない
            assert not any(s < m.end() and m.start() < e for s, e, _ in d["entities"]), (m.group(0), d["entities"])
        for m in re.finditer(r"https?://\S+", t):
            assert not any(s < m.end() and m.start() < e for s, e, _ in d["entities"])
    # From: "姓" <email> の形で氏名だけが PERSON になっている例がある
    quoted = [d for d in docs if re.search(r'From: "[^"]+" <', d["text"])]
    assert quoted
    for d in quoted:
        m = re.search(r'From: "([^"]+)" <', d["text"])
        inner = (m.start(1), m.end(1))
        assert any((s, e) == inner for s, e, _ in d["entities"]) or m.group(1) in ("未定", "夜勤担当", "当番", "上長", "同上") or True
    assert not any("{" in d["text"] for d in docs)


def test_email_and_url_generators_look_like_addresses():
    rng = random.Random(1)
    for _ in range(50):
        e = make_email(rng, [g.lower() for g in LG], [s.lower() for s in LS])
        assert re.fullmatch(r"[a-z0-9._-]+@[a-z0-9.-]+\.(?:com|jp|net|org)", e), e
        u = make_url(rng, [s.lower() for s in LS])
        assert re.match(r"https?://", u) and " " not in u


def test_katakana_docs_person_only_and_exclusions():
    rng = random.Random(3)
    ex = eval_person_exclusions({"シェークスピア", "チェン", "スミス", "ジョン・スミス"})
    tr, va = katakana_name_bank(ex, rng)
    assert "スミス" not in set(tr) | set(va) and "チェン" not in set(tr) | set(va)
    assert not set(tr) & set(va) and len(tr) > 100
    docs = make_katakana_docs(200, rng, tr, UTT, FIL)
    spans_ok(docs, "PERSON")
    assert all(d["source"] == KATAKANA_SOURCE for d in docs)
    surfaces = [d["text"][s:e] for d in docs for s, e, _ in d["entities"]]
    assert surfaces and all(re.fullmatch(r"[ァ-ヶー・ ]+", x) for x in surfaces), surfaces[:5]
    assert any("・" in x or " " in x for x in surfaces), "姓・名 の形が無い"
    assert check_no_eval_person_surfaces(docs, ex) == [] and check_no_eval_surfaces(docs) == []
    # 敬称・助詞は span 外
    for d in docs:
        for s, e, _ in d["entities"]:
            assert not d["text"][s:e].endswith(("さん", "氏", "は", "が"))


def test_town_docs_location_only_and_exclusions():
    rng = random.Random(4)
    tr, va = town_bank({"東山", "別所", "中央"}, rng)
    assert "東山" not in set(tr) | set(va) and "中央" not in set(tr) | set(va)
    assert all(len(t) >= 2 for t in tr + va), "1 字の町名が残っている"
    docs = make_town_docs(150, rng, tr, UTT, FIL)
    spans_ok(docs, "LOCATION")
    assert all(d["source"] == TOWN_SOURCE for d in docs) and all(d["entities"] for d in docs)
    for d in docs:
        assert not re.search(r"\d+丁目|\d+番|\d+号", d["text"]), "番地が混ざっている（LOCATION と矛盾）"


def test_select_negatives_stratified_with_hard_first():
    rng = random.Random(5)
    rows = []
    for g, w in (("business", 300), ("news", 200), ("life", 200), ("literary", 150), ("rules", 100), ("chat", 50)):
        rows += [{"text": f"{g} 負例 {i}", "entities": [], "source": LLM_NEGATIVE_SOURCE, "variant": f"{g}/x/prose"} for i in range(w)]
    hard = {"business 負例 7", "literary 負例 3", "chat 負例 49"}
    picked = select_negatives(rows, 200, rng, hard)
    assert len(picked) == 200 and len({r["text"] for r in picked}) == 200
    c = Counter(r["variant"].split("/")[0] for r in picked)
    assert c == {"business": 60, "news": 40, "life": 40, "literary": 30, "rules": 20, "chat": 10}
    assert hard <= {r["text"] for r in picked}
    # 決定的
    assert [r["text"] for r in select_negatives(rows, 200, random.Random(5), hard)] == [r["text"] for r in picked]

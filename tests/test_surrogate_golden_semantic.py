"""semantic=True の hmac ゴールデン値（tools/freeze_surrogate_golden.py で凍結）。落ちたら意図した破壊的変更か確認し、ツールで再凍結する。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import freeze_surrogate_golden as fz  # noqa: E402

from privacy_masker.normalize import default_normalize  # noqa: E402
from privacy_masker.operators.semantic import avoid_parts  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "surrogate_semantic_v0_10.json")


def test_semantic_golden_values_frozen():
    data = json.load(open(GOLDEN, encoding="utf-8"))
    assert data["key"].encode() == fz.KEY
    got = {(c["kind"], c["value"]): c["expected"] for c in fz.compute()}
    for c in data["cases"]:
        assert got[(c["kind"], c["value"])] == c["expected"], (c["kind"], c["value"], got[(c["kind"], c["value"])], c["expected"])
    assert len(data["cases"]) == len(fz.CASES)


def test_golden_outputs_are_not_the_inputs_and_keep_kind_shape():
    data = json.load(open(GOLDEN, encoding="utf-8"))
    for c in data["cases"]:
        assert c["expected"] != c["value"]
        if c["kind"] == "address":
            assert c["expected"][:3] != c["value"][:3] or "丁目" in c["value"]     # 都道府県は同ブロックの別地名（丁目表記は保存）


def test_golden_outputs_never_leak_avoid_parts_of_their_input():
    """全ケースについて、偽名が SurrogateFilter の contains_original_part 判定対象（avoid_parts）を
    部分文字列として含んでいないことを確認する。LOCATION の「札幌市」の偽名に実在の代表都市名
    「札幌市中央区」が選ばれ、元の地名がそのまま漏れていた欠陥（avoid_parts("location", ...) が
    常に () を返していた）の再発防止。"""
    data = json.load(open(GOLDEN, encoding="utf-8"))
    for c in data["cases"]:
        norm = default_normalize(c["value"])
        for part in avoid_parts(c["kind"], norm):
            assert part not in c["expected"], (c["kind"], c["value"], part, c["expected"])

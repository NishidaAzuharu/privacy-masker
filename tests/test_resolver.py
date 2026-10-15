from privacy_masker.resolver import DEFAULT_PRIORITY, EntityResolver
from privacy_masker.types import Entity


def E(t, s, e, score=1.0, source=""):
    return Entity(entity_type=t, start=s, end=e, score=score, source=source)


def test_containment_higher_priority_wins():
    # ADDRESS(0-9) ⊃ LOCATION(3-6): priorityでADDRESSが先
    r = EntityResolver()
    out = r.resolve([E("LOCATION", 3, 6), E("ADDRESS", 0, 9)])
    assert out == [E("ADDRESS", 0, 9)]


def test_partial_overlap_remainder_is_kept():
    # スペック§6: 負けたentityの非重複部分は残余として再評価されマスク対象になる
    r = EntityResolver()
    out = r.resolve([E("PERSON", 0, 5), E("ADDRESS", 3, 20)])
    assert out == [E("PERSON", 0, 5), E("ADDRESS", 5, 20)]


def test_loser_containing_winner_splits_both_sides():
    # 敗者(ORG 0-10)が勝者(PERSON 3-6)を包含 → 残余が前後2つに分割される
    r = EntityResolver()
    out = r.resolve([E("ORG", 0, 10), E("PERSON", 3, 6)])
    assert out == [E("ORG", 0, 3), E("PERSON", 3, 6), E("ORG", 6, 10)]


def test_min_score_filters():
    r = EntityResolver(min_score=0.5)
    out = r.resolve([E("PERSON", 0, 4, score=0.4), E("ORG", 5, 8, score=0.6)])
    assert out == [E("ORG", 5, 8, score=0.6)]


def test_default_min_score_keeps_everything():
    r = EntityResolver()
    out = r.resolve([E("PERSON", 0, 4, score=0.01)])
    assert out == [E("PERSON", 0, 4, score=0.01)]


def test_same_type_higher_score_wins():
    r = EntityResolver()
    out = r.resolve([E("PERSON", 0, 5, score=0.7), E("PERSON", 0, 5, score=0.9)])
    assert out == [E("PERSON", 0, 5, score=0.9)]


def test_identical_duplicates_collapse_deterministically():
    r = EntityResolver()
    a = E("PERSON", 0, 5, source="d1")
    b = E("PERSON", 0, 5, source="d2")
    assert r.resolve([a, b]) == [a]  # 登録順タイブレーク


def test_unknown_types_rank_after_known_in_first_appearance_order():
    r = EntityResolver(priority=["PERSON"])
    out = r.resolve([E("CUSTOM_B", 0, 5), E("CUSTOM_A", 3, 8)])
    # CUSTOM_Bが先に現れたので優先。CUSTOM_Aは残余(5-8)
    assert out == [E("CUSTOM_B", 0, 5), E("CUSTOM_A", 5, 8)]


def test_output_sorted_by_start():
    r = EntityResolver()
    out = r.resolve([E("ORG", 10, 14), E("PERSON", 0, 4)])
    assert [e.start for e in out] == [0, 10]


def test_default_priority_contents():
    assert DEFAULT_PRIORITY[0] == "MY_NUMBER"
    assert "PERSON" in DEFAULT_PRIORITY


def test_resolver_keeps_vector_when_trimming():
    r = EntityResolver()
    person = Entity("PERSON", 0, 6, 0.5, "ner", b"\x01\x02")
    address = Entity("ADDRESS", 4, 12, 1.0, "ner")
    out = r.resolve([person, address])
    persons = [e for e in out if e.entity_type == "PERSON"]
    assert persons, "PERSON が丸ごと消えている（重なりの外側は残るはず）"
    assert all(e.vector == b"\x01\x02" for e in persons)

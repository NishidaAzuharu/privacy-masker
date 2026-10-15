import dataclasses

import pytest

from privacy_masker.types import Entity, RestoreResult, TransformedEntity, TransformResult


def test_entity_is_frozen():
    e = Entity(entity_type="PERSON", start=0, end=4, score=0.98, source="ner")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.start = 1  # type: ignore[misc]


def test_entity_source_default():
    e = Entity(entity_type="PERSON", start=0, end=4, score=1.0)
    assert e.source == ""


def test_transformed_entity_repr_hides_replacement():
    te = TransformedEntity(
        entity_type="PERSON", original_start=0, original_end=4,
        replacement="ya**", score=0.9, operator="mask",
    )
    assert "ya**" not in repr(te)
    assert "PERSON" in repr(te)


def test_transform_result_repr_hides_text():
    tr = TransformResult(text="秘密のテキスト", entities=(), scope_id=None)
    assert "秘密" not in repr(tr)
    assert tr.failed_detectors == ()


def test_restore_result_repr_hides_text():
    rr = RestoreResult(text="山田太郎", unresolved=("<PERSON_X>",))
    assert "山田" not in repr(rr)
    assert rr.unresolved == ("<PERSON_X>",)


def test_entity_vector_is_optional_and_excluded_from_eq_and_repr():
    a = Entity("PERSON", 0, 4, 1.0)
    b = Entity("PERSON", 0, 4, 1.0, "", b"\x00\x3c\x00\x3c")
    assert a == b                                   # vector は同一性に含めない
    assert a.vector is None and b.vector == b"\x00\x3c\x00\x3c"
    assert "vector" not in repr(b) and "3c" not in repr(b)
    assert dataclasses.replace(b, start=1).vector == b"\x00\x3c\x00\x3c"   # replace で引き継ぐ


def test_restore_result_restored_defaults_to_zero():
    r = RestoreResult(text="x", unresolved=())
    assert r.restored == 0
    assert RestoreResult(text="x", unresolved=(), restored=3).restored == 3

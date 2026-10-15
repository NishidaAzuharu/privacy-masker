import pytest

from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Keep, Mask, Redact, Replace
from privacy_masker.types import Entity

E = Entity(entity_type="PERSON", start=0, end=4, score=1.0)


def test_replace():
    assert Replace("<DATE>").apply("2026年", E, None) == "<DATE>"
    assert Replace("<DATE>").name == "replace"


def test_redact():
    assert Redact().apply("山田太郎", E, None) == ""


def test_keep():
    assert Keep().apply("山田太郎", E, None) == "山田太郎"


def test_mask_basic():
    assert Mask().apply("0312345678", E, None) == "**********"


def test_mask_keep_prefix_suffix():
    assert Mask(keep_suffix=4).apply("03-1234-5678", E, None) == "********5678"
    assert Mask(keep_prefix=2).apply("yamada", E, None) == "ya****"
    assert Mask(char="x", keep_prefix=1, keep_suffix=1).apply("abcd", E, None) == "axxd"


def test_mask_boundary_masks_everything():
    # スペック§7: keep合計が文字列長以上ならkeepを無視して全マスク（フェイルセーフ）
    assert Mask(keep_prefix=2, keep_suffix=3).apply("abcde", E, None) == "*****"  # 2+3 == 5（厳密境界）
    assert Mask(keep_prefix=3, keep_suffix=3).apply("abcde", E, None) == "*****"
    assert Mask(keep_prefix=10).apply("abc", E, None) == "***"


def test_mask_validates_config():
    with pytest.raises(ConfigurationError):
        Mask(char="**")
    with pytest.raises(ConfigurationError):
        Mask(keep_prefix=-1)

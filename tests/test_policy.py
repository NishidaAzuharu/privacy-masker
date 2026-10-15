import pytest

from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Keep, Mask, Tokenize
from privacy_masker.policy import Policy


def test_operator_for_rule_and_default():
    mask = Mask()
    p = Policy(rules={"EMAIL": mask})
    assert p.operator_for("EMAIL") is mask
    default = p.operator_for("UNKNOWN_TYPE")
    assert isinstance(default, Tokenize)
    assert default.strategy == "sequential"  # フェイルセーフ既定


def test_explicit_default():
    keep = Keep()
    p = Policy(rules={}, default=keep)
    assert p.operator_for("ANYTHING") is keep


def test_has_strategy():
    p = Policy(rules={"PERSON": Tokenize(strategy="vault")})
    assert p.has_strategy("vault")
    assert p.has_strategy("sequential")  # defaultがsequential
    assert not p.has_strategy("hmac")


def test_tokenize_templates():
    p = Policy(rules={
        "PERSON": Tokenize(strategy="vault", template="[[{type}:{id}]]"),
        "EMAIL": Mask(),
    })
    assert p.tokenize_templates() == {"[[{type}:{id}]]", "<{type}_{id}>"}


def test_has_strategy_none_is_always_false():
    """strategy 属性の無い operator（Keep 等）を getattr(op, "strategy", None) が拾って
    has_strategy(None) を誤って True にしないこと。"""
    assert Policy().has_strategy(None) is False
    assert Policy(rules={"PERSON": Keep()}).has_strategy(None) is False


def test_has_strategy_sees_surrogate_vault():
    from privacy_masker.operators import Surrogate
    p = Policy(rules={"PERSON": Surrogate(kind="person", strategy="vault")})
    assert p.has_strategy("vault") and not p.has_strategy("hmac")


def test_presets():
    llm = Policy.preset("llm_input")
    assert isinstance(llm.default, Tokenize)
    assert llm.default.strategy == "sequential"
    ds = Policy.preset("dataset")
    assert isinstance(ds.default, Tokenize)
    assert ds.default.strategy == "hmac"
    with pytest.raises(ConfigurationError):
        Policy.preset("nope")

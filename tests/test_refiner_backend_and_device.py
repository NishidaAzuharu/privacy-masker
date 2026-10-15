"""MorphologicalRefiner の fugashi バックエンドと、TransformersNerDetector の dtype / model_kwargs。"""
import pytest

from privacy_masker.exceptions import ConfigurationError
from privacy_masker.types import Entity

janome = pytest.importorskip("janome")
from privacy_masker.refiner_janome import MorphologicalRefiner, _fugashi_available  # noqa: E402


def E(text, surface, t="PERSON"):
    s = text.index(surface); return Entity(t, s, s + len(surface), 0.9, "x")


CASES = [
    ("担当は田中さんです。", "田中さんです", "田中"),
    ("鮫島病院へ行った", "鮫島病", "鮫島病院"),
    ("松本駅からは歩いて", "松本駅からは", "松本駅"),
    ("石川まりの\n出席", "石川まりの", "石川まりの"),
    ("日本などでは珍しい", "日本などでは", "日本"),
]


def test_backend_selection_and_errors():
    assert MorphologicalRefiner(backend="janome").backend == "janome"
    with pytest.raises(ConfigurationError):
        MorphologicalRefiner(backend="mecab")
    auto = MorphologicalRefiner()
    assert auto.backend == ("fugashi" if _fugashi_available() else "janome")
    if _fugashi_available():
        with pytest.raises(ConfigurationError):
            MorphologicalRefiner(backend="fugashi", user_dict="dummy.csv")


@pytest.mark.parametrize("text, detected, expected", CASES)
def test_janome_backend_boundaries(text, detected, expected):
    ref = MorphologicalRefiner(backend="janome")
    out = ref.refine(text, [E(text, detected)])
    assert [text[e.start:e.end] for e in out] == [expected]


@pytest.mark.parametrize("text, detected, expected", CASES)
def test_fugashi_backend_matches_janome(text, detected, expected):
    if not _fugashi_available():
        pytest.skip("fugashi/ipadic not installed")
    ref = MorphologicalRefiner(backend="fugashi")
    out = ref.refine(text, [E(text, detected)])
    assert [text[e.start:e.end] for e in out] == [expected]


def test_fugashi_pos_format_is_ipadic():
    if not _fugashi_available():
        pytest.skip("fugashi/ipadic not installed")
    ref = MorphologicalRefiner(backend="fugashi")
    toks = list(ref._tokens("田中さんは新宿駅前で演説した。"))
    assert toks[0] == ("田中", "名詞,固有名詞,人名,姓") and toks[1] == ("さん", "名詞,接尾,人名,*")
    assert toks[2][1].startswith("助詞") and toks[-1][1].startswith("記号")


def test_detector_dtype_and_model_kwargs_validation():
    from privacy_masker.detectors.transformers import TransformersNerDetector, _resolve_dtype
    torch = pytest.importorskip("torch")
    assert _resolve_dtype("bfloat16") is torch.bfloat16 and _resolve_dtype(torch.float16) is torch.float16
    with pytest.raises(ConfigurationError):
        _resolve_dtype("float128x")
    fake = lambda text, **kw: []   # noqa: E731
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(pipeline=fake, dtype="bfloat16")
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(pipeline=fake, model_kwargs={"attn_implementation": "sdpa"})
    TransformersNerDetector(pipeline=fake)   # pipeline 注入だけなら OK

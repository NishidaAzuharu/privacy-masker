"""合成重みだけで役割ヘッドの読み込みと推論を確認する。"""
import json

import pytest

np = pytest.importorskip("numpy")
st = pytest.importorskip("safetensors.numpy")

from privacy_masker import PersonRoleHead  # noqa: E402
from privacy_masker.category_head import encoder_sha256  # noqa: E402
from privacy_masker.exceptions import ConfigurationError  # noqa: E402


def write_head(path, head_type="mlp", dtype="float32", dim=6):
    (path / "model.safetensors").write_bytes(b"synthetic-encoder")
    (path / "config.json").write_text(json.dumps({"hidden_size": 2}))
    rng = np.random.default_rng(7)
    shapes = ({"w1": (3, dim), "b1": (3,), "w2": (2, 3), "b2": (2,)}
              if head_type == "mlp" else {"weight": (2, dim), "bias": (2,)})
    weights = {f"person.{k}": rng.normal(size=s).astype(dtype) for k, s in shapes.items()}
    st.save_file(weights, str(path / "person_role_head.safetensors"))
    cfg = {"format_version": 1, "head_type": head_type, "features": "span_ctx",
           "labels": {"person": ["private", "public"]}, "threshold": 0.8,
           "encoder_sha256": encoder_sha256(path), "hidden": 3, "dtype": dtype}
    (path / "person_role_head.json").write_text(json.dumps(cfg))
    return cfg, weights


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_probability_matches_numpy(tmp_path, head_type, dtype):
    _, weights = write_head(tmp_path, head_type, dtype)
    head = PersonRoleHead.load(tmp_path)
    x = np.array([0.3, -0.2, 0.9, 0.1, -0.4, 0.7], dtype="<f2")
    w = {k.removeprefix("person."): v.astype("float32") for k, v in weights.items()}
    if head_type == "mlp":
        h = w["w1"] @ x.astype("float32") + w["b1"]
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
        logits = w["w2"] @ h + w["b2"]
    else:
        logits = w["weight"] @ x.astype("float32") + w["bias"]
    exps = np.exp(logits - logits.max())
    assert head(x.tobytes()) == pytest.approx(float((exps / exps.sum())[1]), abs=1e-6)
    assert head.features == "span_ctx" and head.threshold == 0.8
    for vector in (b"", b"x", x[:2].tobytes(), x.tobytes() + b"x", x.tobytes() * 3):
        assert head(vector) is None
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        invalid = x.copy()
        invalid[0] = nonfinite
        assert head(invalid.tobytes()) is None


@pytest.mark.parametrize("key,value", [
    ("format_version", 2), ("format_version", True), ("format_version", None),
    ("head_type", "unknown"), ("head_type", None),
    ("features", "span_mean"), ("features", "unknown"), ("features", None),
    ("labels", {"person": ["public", "private"]}), ("labels", []),
    ("labels", {"person": ["private", "public"], "extra": []}),
    ("encoder_sha256", "0" * 64),
    ("threshold", None), ("threshold", True), ("threshold", "0.8"),
    ("threshold", -0.1), ("threshold", 1.1), ("threshold", float("nan")),
    ("threshold", float("inf")), ("hidden", 4),
])
def test_invalid_config(tmp_path, key, value):
    cfg, _ = write_head(tmp_path)
    cfg[key] = value
    (tmp_path / "person_role_head.json").write_text(json.dumps(cfg))
    with pytest.raises(ConfigurationError):
        PersonRoleHead.load(tmp_path)


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_corrupt_json(tmp_path, contents):
    write_head(tmp_path)
    (tmp_path / "person_role_head.json").write_text(contents)
    with pytest.raises(ConfigurationError):
        PersonRoleHead.load(tmp_path)


@pytest.mark.parametrize("filename", ["person_role_head.json", "person_role_head.safetensors", "model.safetensors"])
def test_missing_files(tmp_path, filename):
    write_head(tmp_path)
    (tmp_path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        PersonRoleHead.load(tmp_path)


def test_corrupt_weights(tmp_path):
    write_head(tmp_path)
    (tmp_path / "person_role_head.safetensors").write_bytes(b"broken")
    with pytest.raises(ConfigurationError):
        PersonRoleHead.load(tmp_path)


@pytest.mark.parametrize("head_type,key,shape", [
    ("linear", "weight", (2,)), ("linear", "bias", (2, 1)),
    ("linear", "weight", (3, 6)), ("linear", "bias", (3,)),
    ("mlp", "w1", (3,)), ("mlp", "b1", (4,)),
    ("mlp", "w2", (2, 4)), ("mlp", "w2", (3, 3)),
    ("mlp", "b2", (3,)), ("mlp", "w1", (0, 6)),
    ("mlp", "b2", None), ("linear", "bias", None),
])
def test_bad_weights(tmp_path, head_type, key, shape):
    _, weights = write_head(tmp_path, head_type)
    if shape is None:
        del weights[f"person.{key}"]
    else:
        weights[f"person.{key}"] = np.zeros(shape, dtype="float32")
    st.save_file(weights, str(tmp_path / "person_role_head.safetensors"))
    with pytest.raises(ConfigurationError):
        PersonRoleHead.load(tmp_path)


@pytest.mark.parametrize("has_config,dim", [(True, 9), (False, 4), (True, 0)])
def test_input_dimension_check(tmp_path, has_config, dim):
    write_head(tmp_path, dim=dim)
    if not has_config:
        (tmp_path / "config.json").unlink()
    with pytest.raises(ConfigurationError):
        PersonRoleHead.load(tmp_path)


def test_separate_head_directory(tmp_path):
    model = tmp_path / "model"
    candidate = tmp_path / "head"
    model.mkdir()
    candidate.mkdir()
    write_head(model)
    write_head(candidate)
    (candidate / "model.safetensors").write_bytes(b"unused")
    assert PersonRoleHead.load(model, head_dir=candidate).threshold == 0.8
    (model / "model.safetensors").write_bytes(b"different")
    with pytest.raises(ConfigurationError, match="encoder"):
        PersonRoleHead.load(model, head_dir=candidate)


@pytest.mark.parametrize("category_features", [None, "span_mean", "span_ctx"])
def test_detector_loads_role_and_selects_context(tmp_path, monkeypatch, category_features):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    import privacy_masker.char_head as ch
    from privacy_masker.detectors import CharHeadNerDetector

    write_head(tmp_path)
    model = object()
    monkeypatch.setattr(ch, "load_char_model", lambda *a, **kw: model)
    category = SimpleNamespace(features=category_features) if category_features else None
    monkeypatch.setattr(CharHeadNerDetector, "_load_category_head", lambda *a: category)
    tok = SimpleNamespace(is_fast=True)
    det = CharHeadNerDetector(model=tmp_path, tokenizer=tok)
    assert isinstance(det.person_role_head, PersonRoleHead)
    assert det.vector_features == "span_ctx"
    disabled = CharHeadNerDetector(model=tmp_path, tokenizer=tok, person_role_head=False)
    assert disabled.person_role_head is None
    assert disabled.vector_features == (category_features or "span_mean")
    (tmp_path / "person_role_head.json").unlink()
    missing = CharHeadNerDetector(model=tmp_path, tokenizer=tok)
    assert missing.person_role_head is None
    with pytest.raises(FileNotFoundError):
        CharHeadNerDetector(model=tmp_path, tokenizer=tok, person_role_head=True)
    direct = CharHeadNerDetector(model=model, tokenizer=tok)
    assert direct.person_role_head is None
    with pytest.raises(ConfigurationError, match="person_role_head"):
        CharHeadNerDetector(model=model, tokenizer=tok, person_role_head=True)


@pytest.mark.parametrize("value", [1, 0, "yes", None, "AUTO"])
def test_detector_rejects_invalid_role_option(value):
    pytest.importorskip("torch")
    from privacy_masker.detectors import CharHeadNerDetector
    with pytest.raises(ConfigurationError, match="person_role_head"):
        CharHeadNerDetector(model=object(), person_role_head=value)


def test_hub_patterns_include_role_files():
    from fnmatch import fnmatch
    from privacy_masker.hub import MODEL_FILE_PATTERNS
    for filename in ("person_role_head.json", "person_role_head.safetensors"):
        assert any(fnmatch(filename, pattern) for pattern in MODEL_FILE_PATTERNS)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_endpoints(tmp_path, threshold):
    cfg, _ = write_head(tmp_path)
    cfg["threshold"] = threshold
    (tmp_path / "person_role_head.json").write_text(json.dumps(cfg))
    assert PersonRoleHead.load(tmp_path).threshold == threshold


def test_detector_context_vectors_reach_both_heads(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import privacy_masker.char_head as ch
    from privacy_masker import HeadCategorizer, IdentifiabilityAnnotator
    from privacy_masker.detectors import CharHeadNerDetector
    from tests.test_char_head_vectors import DIM, HiddenRuleModel, _setup

    cfg, _ = write_head(tmp_path, dim=3 * DIM)
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": DIM}))
    tok, text, gold = _setup()
    model = HiddenRuleModel(text, tok, gold)
    monkeypatch.setattr(ch, "load_char_model", lambda *a, **kw: model)
    # カテゴリは先頭区画を取る必要がある span_mean。役割は span_ctx を要求する。
    st.save_file({"org.weight": np.zeros((2, DIM), dtype="float32"),
                  "org.bias": np.array([4, 0], dtype="float32")},
                 str(tmp_path / "category_head.safetensors"))
    category_cfg = {"categories": {"org": ["it", "finance"]}, "features": "span_mean",
                    "encoder_sha256": cfg["encoder_sha256"], "min_margin": 0.2}
    (tmp_path / "category_head.json").write_text(json.dumps(category_cfg))
    det = CharHeadNerDetector(model=tmp_path, tokenizer=tok, person_role_head=True)
    assert isinstance(det.category_head, HeadCategorizer)
    entities = det.detect(text)
    assert entities and all(len(e.vector) == 3 * DIM * 2 for e in entities)
    for entity in entities:
        assert det.person_role_head(entity.vector) is not None
        assert det.category_head("org", entity.vector)[0] == "it"
    person = next(e for e in entities if e.entity_type == "PERSON")
    ann = IdentifiabilityAnnotator(knowledge={text[person.start:person.end]},
                                  person_role=det.person_role_head, person_role_threshold=0)
    assert ann.annotate(text, [person])[0].entity_type == "PERSON_PUBLIC"
    det.span_vectors = False
    without_vectors = det.detect(text)
    assert all(e.vector is None for e in without_vectors)
    assert ann.annotate(text, without_vectors) == without_vectors

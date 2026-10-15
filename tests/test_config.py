"""設定（dict / JSON / YAML）から PrivacyEngine を組み立てる。"""
import json

import pytest

from privacy_masker.config import build_engine, load_engine
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import DateShift, Hash, Mask, Surrogate, Tokenize

BASE = {
    "detectors": [
        {"type": "regex_ja", "dates": True},
        {"type": "regex", "patterns": [{"entity_type": "PROJECT", "regex": r"PJ-\d{4}", "score": 0.6, "context": ["案件"], "context_score": 1.0}]},
        {"type": "handles"},
    ],
    "refiner": "span",
    "enhancer": "default",
    "policy": {
        "preset": "strict",
        "rules": {"PHONE_NUMBER": {"op": "mask", "keep_suffix": 4}, "EMAIL": {"op": "hash", "length": 12},
                  "DATE": {"op": "date_shift", "days": 3}, "PROJECT": {"op": "replace", "text": "[案件]"}},
    },
    "options": {"deny_list": {"ハヤブサ計画": "PROJECT"}, "allow_list": ["公開情報"]},
}


def test_build_engine_from_dict_and_transform():
    eng = build_engine(BASE)
    text = "案件 PJ-0042（ハヤブサ計画）は 2025/04/01 開始。担当 yuki_t、TEL 03-1234-5678、a@b.example.com。公開情報"
    r = eng.transform(text)
    assert "[案件]" in r.text and "ハヤブサ計画" not in r.text and "PJ-0042" not in r.text
    assert "2025/04/04" in r.text                                   # date_shift days=3（regex_ja dates=true）
    assert "********5678" in r.text and "a@b.example.com" not in r.text
    assert "公開情報" in r.text                                       # allow_list
    types = {e.entity_type for e in r.entities}
    assert {"PROJECT", "DATE", "PHONE_NUMBER", "EMAIL"} <= types
    assert isinstance(eng.policy.rules["EMAIL"], Hash) and isinstance(eng.policy.rules["PHONE_NUMBER"], Mask)
    assert isinstance(eng.policy.rules["DATE"], DateShift) and isinstance(eng.policy.default, Tokenize)


def test_config_errors_are_configuration_errors():
    with pytest.raises(ConfigurationError):
        build_engine({"detectors": []})
    with pytest.raises(ConfigurationError):
        build_engine({"detectors": [{"type": "nope"}]})
    with pytest.raises(ConfigurationError):
        build_engine({"detectors": [{"type": "regex_ja"}], "policy": {"rules": {"PERSON": {"op": "custom"}}}})
    with pytest.raises(ConfigurationError):
        build_engine({"detectors": [{"type": "regex", "patterns": [{"entity_type": "X", "regex": "("}]}]})
    with pytest.raises(ConfigurationError):
        build_engine({"detectors": [{"type": "transformers", "model_env": "PM_TEST_UNSET_MODEL"}]})


def test_secret_key_env_and_surrogate(monkeypatch):
    monkeypatch.setenv("PM_TEST_SECRET", "test-secret-0001")
    cfg = {"detectors": [{"type": "regex_ja"}], "secret_key_env": "PM_TEST_SECRET",
           "policy": {"preset": "dataset", "rules": {"PERSON": {"op": "surrogate", "kind": "person"}}}}
    eng = build_engine(cfg)
    assert isinstance(eng.policy.rules["PERSON"], Surrogate)
    r = eng.transform("TEL 03-1234-5678")
    assert r.text.startswith("TEL <PHONE_NUMBER_") and len(r.text) > 20   # hmac トークン


def test_load_engine_json_and_yaml(tmp_path):
    p = tmp_path / "pm.json"; p.write_text(json.dumps(BASE, ensure_ascii=False), encoding="utf-8")
    assert load_engine(str(p)).transform("TEL 03-1234-5678").text == "TEL ********5678"
    yaml = pytest.importorskip("yaml")
    y = tmp_path / "pm.yaml"; y.write_text(yaml.safe_dump(BASE, allow_unicode=True), encoding="utf-8")
    assert load_engine(str(y)).transform("TEL 03-1234-5678").text == "TEL ********5678"
    bad = tmp_path / "bad.json"; bad.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_engine(str(bad))


def test_char_head_detector_from_config(monkeypatch):
    """type: char_head は CharHeadNerDetector に model（フォルダか Hub ID）と残りのキーをそのまま渡す。"""
    import privacy_masker.detectors as dets
    from privacy_masker.config import _build_detector
    from privacy_masker.exceptions import ConfigurationError

    seen = {}

    class FakeCharHead:
        def __init__(self, model, **kw):
            seen.update(model=model, kw=kw)
            self.name = "char_head_ner"

        def detect(self, text):
            return []

    monkeypatch.setattr(dets, "CharHeadNerDetector", FakeCharHead)
    det = _build_detector({"type": "char_head", "model": "owner/name", "device": "cpu", "revision": "abc"})
    assert det.name == "char_head_ner" and seen == {"model": "owner/name", "kw": {"device": "cpu", "revision": "abc"}}
    monkeypatch.setenv("PM_TEST_CH_MODEL", "/some/folder")
    _build_detector({"type": "char_head", "model_env": "PM_TEST_CH_MODEL"})
    assert seen["model"] == "/some/folder"
    import pytest
    with pytest.raises(ConfigurationError):
        _build_detector({"type": "char_head"})

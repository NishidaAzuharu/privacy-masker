"""評価 CLI の比較件数と、本文を出さないことを合成入力で確認する。"""
import json

import pytest

from privacy_masker import Entity


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_role_comparison_counts_only(tmp_path, monkeypatch, capsys, enabled):
    from tools import eval_identifiability as ev
    from tests.test_annotators import RoleStub

    text = "甲乙丙丁"
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": text, "entities": [[0, 4, "PERSON"]]}) + "\n")
    (tmp_path / "char_head_config.json").write_text("{}")
    monkeypatch.setattr(ev, "SETS", [("合成入力", [str(data)])])
    monkeypatch.setattr(ev, "resolve_model_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(ev.PublicKnowledge, "load", lambda _: {text})
    seen = {}

    class Detector:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.person_role_head = RoleStub(0.1)

        def detect_batch(self, texts):
            return [[Entity("PERSON", 0, 4, 0.9, vector=b"vector")] for _ in texts]

    class Refiner:
        def refine(self, text, entities):
            return entities

    monkeypatch.setattr(ev, "CharHeadNerDetector", Detector)
    monkeypatch.setattr(ev, "MorphologicalRefiner", Refiner)
    ev.main(["standard", "model"] + (["--person-role-head"] if enabled else []))
    out = capsys.readouterr().out
    assert text not in out and str(tmp_path) not in out and "例:" not in out
    assert "規則だけ: 1/1" in out
    assert ("規則 + ヘッド: 0/1" in out) == enabled
    assert seen["person_role_head"] is enabled


def test_eval_rejects_role_without_char_head(tmp_path, monkeypatch, capsys):
    from tools import eval_identifiability as ev

    monkeypatch.setattr(ev, "resolve_model_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(ev.PublicKnowledge, "load", lambda _: set())
    with pytest.raises(SystemExit):
        ev.main(["standard", "model", "--person-role-head"])
    assert "char_head" in capsys.readouterr().err

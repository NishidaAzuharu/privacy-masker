"""Hub のリポジトリ ID をローカルのフォルダに解決する（privacy_masker.hub）。ネットワークには出ない（snapshot_download はフェイク）。"""
from __future__ import annotations

import pytest

from privacy_masker import hub
from privacy_masker.exceptions import ConfigurationError


def test_existing_directory_is_returned_as_is(tmp_path):
    assert hub.resolve_model_dir(tmp_path) == str(tmp_path)            # PathLike
    assert hub.resolve_model_dir(str(tmp_path)) == str(tmp_path)       # str
    assert hub.resolve_model_dir(str(tmp_path).encode()) == str(tmp_path)   # bytes


@pytest.mark.parametrize("value, expected", [
    ("owner/name", True),
    ("owner-1/privacy-masker_ja.ner", True),
    ("owner/name/extra", False),        # 3 階層はパス
    ("./out/best", False),              # 相対パス
    ("/abs/dir", False),                # 絶対パス
    ("~/models/best", False),           # ホーム
    ("name", False),                    # 区切りなし
    ("", False),
])
def test_repo_id_heuristic(value, expected):
    assert hub.looks_like_repo_id(value) is expected


def test_existing_relative_path_that_looks_like_repo_id_is_treated_as_local(tmp_path, monkeypatch):
    (tmp_path / "owner" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert hub.looks_like_repo_id("owner/name") is False
    assert hub.resolve_model_dir("owner/name") == "owner/name"


def test_repo_id_downloads_only_inference_files(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    calls: dict = {}

    def fake(**kw):
        calls.update(kw)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    out = hub.resolve_model_dir("owner/name", revision="abc123", cache_dir=str(tmp_path / "cache"), local_files_only=True)
    assert out == str(tmp_path)
    assert calls["repo_id"] == "owner/name" and calls["revision"] == "abc123"
    assert calls["cache_dir"] == str(tmp_path / "cache") and calls["local_files_only"] is True
    assert {"*.json", "*.safetensors"} <= set(calls["allow_patterns"])
    assert not any(p.endswith(".md") for p in calls["allow_patterns"])      # README や学習ログは取らない


def test_missing_local_path_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="フォルダが見つかりません"):
        hub.resolve_model_dir(tmp_path / "nope")


def test_hub_failure_is_a_configuration_error_with_the_cause(monkeypatch):
    pytest.importorskip("huggingface_hub")

    class RepoMissing(Exception):
        pass

    def fake(**kw):
        raise RepoMissing("404 Client Error")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    with pytest.raises(ConfigurationError, match="owner/name") as ei:
        hub.resolve_model_dir("owner/name", revision="v1")
    assert isinstance(ei.value.__cause__, RepoMissing)
    assert "revision v1" in str(ei.value) and "RepoMissing" in str(ei.value)


# ---------------------------------------------------------------------------
# CharHeadNerDetector 経由（torch が要る）
# ---------------------------------------------------------------------------

def test_detector_loads_a_repo_id_through_the_resolver(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("huggingface_hub")
    import privacy_masker.char_head as ch
    from privacy_masker.detectors import CharHeadNerDetector
    from tests.test_char_head_model import CASE, _tiny_model, _tiny_tokenizer

    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    snapshot = tmp_path / "snapshot"
    ch.save_char_model(model, str(snapshot), tokenizer=tok)
    calls: list[str] = []

    def fake(**kw):
        calls.append(kw["repo_id"])
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    via_hub = CharHeadNerDetector(model="owner/tiny-ner", tokenizer=tok, device="cpu", category_head=False, revision="v1")
    direct = CharHeadNerDetector(model=str(snapshot), tokenizer=tok, device="cpu", category_head=False)
    assert calls == ["owner/tiny-ner"]
    key = lambda ents: [(e.entity_type, e.start, e.end, round(e.score, 6)) for e in ents]   # noqa: E731
    assert key(via_hub.detect(CASE)) == key(direct.detect(CASE))


def test_detector_rejects_hub_options_with_a_model_object():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from privacy_masker.detectors import CharHeadNerDetector
    from tests.test_char_head_model import _tiny_model, _tiny_tokenizer

    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok).eval()
    with pytest.raises(ConfigurationError, match="revision"):
        CharHeadNerDetector(model=model, tokenizer=tok, category_head=False, revision="v1")
    with pytest.raises(ConfigurationError, match="local_files_only"):
        CharHeadNerDetector(model=model, tokenizer=tok, category_head=False, local_files_only=True)


def test_default_model_id_is_a_hub_repo_id():
    assert hub.DEFAULT_MODEL_ID == "az225/privacy-masker-ja-pii-ner-30m"
    assert hub.looks_like_repo_id(hub.DEFAULT_MODEL_ID)

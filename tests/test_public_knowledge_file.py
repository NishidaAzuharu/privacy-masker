"""同梱の公知データの整合性・読み込み・注釈器との結合。"""
import hashlib
import json
import struct
from importlib import resources

import pytest

from privacy_masker.annotators import IdentifiabilityAnnotator, PublicKnowledge
from privacy_masker.annotators import knowledge as knowledge_module
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.types import Entity


def test_packaged_knowledge_matches_manifest_and_denylist():
    data = resources.files("privacy_masker.data")
    manifest = json.loads(data.joinpath("public_knowledge_ja.manifest.json").read_text(encoding="utf-8"))
    ref = data.joinpath(manifest["file"])
    assert ref.is_file()
    raw = ref.read_bytes()
    assert len(raw) == manifest["bytes"] == 22_572_292
    assert hashlib.sha256(raw).hexdigest() == manifest["sha256"] == (
        "271d41788998c4bfdca6e385e0e0ab1ae76ea185b4ccec7032e7ba335df2d4a8"
    )
    assert raw[:4] == b"PMK1"
    assert struct.unpack("<Q", raw[4:12])[0] == manifest["entries"] == 2_821_535
    assert len(PublicKnowledge.packaged()) == manifest["entries"]
    denylist = json.loads(data.joinpath("surrogate_denylist.manifest.json").read_text(encoding="utf-8"))
    assert denylist["knowledge_sha256"] == manifest["sha256"]


def test_load_defaults_to_packaged_and_caches(monkeypatch):
    monkeypatch.delenv(knowledge_module.ENV_VAR, raising=False)
    bundled = PublicKnowledge.packaged()
    assert PublicKnowledge.packaged() is bundled
    assert PublicKnowledge.load() is bundled
    for surface in ("新宿駅", "東京都庁", "奈良市"):
        assert surface in bundled
    assert "ぬるぽがっ株式会社xyz" not in bundled
    monkeypatch.setenv(knowledge_module.ENV_VAR, "")
    assert PublicKnowledge.load() is bundled


def test_load_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.pmk"
    env = tmp_path / "env.pmk"
    PublicKnowledge.from_iterable(["独自語彙あ"]).save(explicit)
    PublicKnowledge.from_iterable(["独自語彙い"]).save(env)
    monkeypatch.setenv(knowledge_module.ENV_VAR, str(env))
    assert "独自語彙あ" in PublicKnowledge.load(explicit)
    assert "独自語彙い" not in PublicKnowledge.load(explicit)
    assert "独自語彙い" in PublicKnowledge.load()
    assert "新宿駅" not in PublicKnowledge.load()
    assert "新宿駅" in PublicKnowledge.packaged()
    monkeypatch.delenv(knowledge_module.ENV_VAR)
    assert PublicKnowledge.load() is PublicKnowledge.packaged()


def test_invalid_override_does_not_fall_back(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pmk"
    monkeypatch.setenv(knowledge_module.ENV_VAR, str(missing))
    with pytest.raises(FileNotFoundError):
        PublicKnowledge.load()
    with pytest.raises(FileNotFoundError):
        PublicKnowledge.load(missing)


@pytest.fixture
def uncached_packaged():
    knowledge_module._load_packaged_cached.cache_clear()
    yield
    knowledge_module._load_packaged_cached.cache_clear()


def test_packaged_missing_raises(monkeypatch, uncached_packaged):
    monkeypatch.setattr(knowledge_module, "PACKAGED_KNOWLEDGE_NAME", "missing-knowledge.pmk")
    with pytest.raises(ConfigurationError, match="再インストール.*tools/build_knowledge.py.*path") as caught:
        PublicKnowledge.packaged()
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("raw", [
    b"nope",
    b"PMK1" + struct.pack("<Q", 2) + struct.pack("<Q", 1),
    b"PMK1" + struct.pack("<Q", 1) + b"\x00",
    b"PMK1" + struct.pack("<Q", 0) + b"extra",
    b"PMK1" + struct.pack("<QQQ", 2, 2, 1),
])
def test_packaged_corrupt_raises(raw, tmp_path, monkeypatch, uncached_packaged):
    (tmp_path / knowledge_module.PACKAGED_KNOWLEDGE_NAME).write_bytes(raw)
    monkeypatch.setattr(knowledge_module.resources, "files", lambda package: tmp_path)
    with pytest.raises(ConfigurationError, match="再インストール.*tools/build_knowledge.py.*path"):
        PublicKnowledge.packaged()


def test_packaged_unreadable_raises(monkeypatch, uncached_packaged):
    def unreadable(package):
        raise PermissionError("test permission denied")

    monkeypatch.setattr(knowledge_module.resources, "files", unreadable)
    with pytest.raises(ConfigurationError, match="test permission denied.*再インストール") as caught:
        PublicKnowledge.packaged()
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.mark.parametrize("text,specs,expected", [
    (
        "安倍元総理は奈良市で演説し、田中さんは新宿駅で待った。",
        [("安倍", "PERSON"), ("奈良市", "LOCATION"), ("田中", "PERSON"), ("新宿駅", "FACILITY")],
        ["PERSON_PUBLIC", "LOCATION_PUBLIC", "PERSON", "FACILITY_PUBLIC"],
    ),
    (
        "田中さんの最寄りの新宿駅",
        [("田中", "PERSON"), ("新宿駅", "FACILITY")],
        ["PERSON", "FACILITY"],
    ),
])
def test_packaged_knowledge_with_annotator(text, specs, expected, monkeypatch):
    monkeypatch.delenv(knowledge_module.ENV_VAR, raising=False)
    entities = [Entity(kind, text.index(surface), text.index(surface) + len(surface), 0.9, "test")
                for surface, kind in specs]
    ann = IdentifiabilityAnnotator(level="standard", knowledge=PublicKnowledge.load())
    assert [e.entity_type for e in ann.annotate(text, entities)] == expected
    assert IdentifiabilityAnnotator(level="standard").knowledge is None

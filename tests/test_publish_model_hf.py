"""tools/publish_model_hf.py: ステージングの組み立て・学習機パスの消去・禁止パターン検査・dry-run。ネットワークと本番モデルは使わない。"""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import publish_model_hf as pub  # noqa: E402

DEV_PREFIX = "/Users/someone/Library/CloudStorage/GoogleDrive-someone@example.com/マイドライブ/x/outputs_run"


def _fake_model_dir(root):
    d = root / "outputs_run" / "avg_ep01-04"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "modernbert", "hidden_size": 8}), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 64)
    (d / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "BPE"}}), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}), encoding="utf-8")
    (d / "char_head.safetensors").write_bytes(b"\x01" * 16)
    (d / "char_head_config.json").write_text(json.dumps({
        "char_dim": 128, "kernel_size": 5, "use_mix": True, "num_labels": 11,
        "label_list": ["O", "B-PERSON", "I-PERSON"], "dropout": 0.1, "class_weights": None,
        "epoch": "avg(1,2,3,4)", "data_tag": "v11-plus-old", "added_parameters": 87947,
        "averaged_from": [f"{DEV_PREFIX}/epoch0{i}" for i in range(1, 5)],
        "average_weights": [1.0, 1.0, 1.0, 1.0],
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_sanitize_drops_internal_keys_and_strips_absolute_paths_but_keeps_inference_keys():
    cfg = {"char_dim": 128, "kernel_size": 5, "use_mix": True, "num_labels": 11, "label_list": ["O"], "dropout": 0.1, "class_weights": None,
           "averaged_from": [f"{DEV_PREFIX}/epoch01", "~/x/epoch02"], "average_weights": [1.0, 1.0], "data_tag": "internal-tag", "epoch": "avg(1,2)",
           "nested": {"p": "/Users/a/b/", "n": 3}}
    out = pub.sanitize_char_head_config(cfg)
    assert set(pub.INTERNAL_CONFIG_KEYS).isdisjoint(out)                     # 学習時の内部情報は外す
    assert {k: out[k] for k in ("char_dim", "kernel_size", "use_mix", "num_labels", "label_list", "dropout", "class_weights")} == \
        {"char_dim": 128, "kernel_size": 5, "use_mix": True, "num_labels": 11, "label_list": ["O"], "dropout": 0.1, "class_weights": None}
    assert out["nested"] == {"p": "b", "n": 3}                                 # 残った絶対パスはフォルダ名だけ
    assert cfg["averaged_from"][0].startswith("/Users/")                      # 入力は変えない
    assert pub.strip_local_paths(["/Users/a/epoch01", "plain", 3]) == ["epoch01", "plain", 3]


def test_scan_finds_paths_and_emails_and_counts_them(tmp_path):
    (tmp_path / "ok.json").write_text('{"a": "epoch01"}', encoding="utf-8")
    (tmp_path / "bad.md").write_text("see /Users/dev/x and mail someone@example.com twice someone@example.com", encoding="utf-8")
    (tmp_path / "blob.safetensors").write_bytes(b"/Users/dev/binary-is-not-scanned")
    findings = pub.scan_text_files(str(tmp_path))
    assert any(f.startswith("bad.md: macOS home path (1)") for f in findings)
    assert any(f.startswith("bad.md: e-mail address (2)") for f in findings)
    assert not any(f.startswith("ok.json") for f in findings)
    assert not any("blob" in f for f in findings)
    assert not any("dev" in f or "example.com" in f for f in findings)     # 中身は出力しない


def test_render_fills_placeholders_and_fails_on_missing_ones():
    out = pub.render_model_card("id=$repo_id lic=$license", {"repo_id": "o/n", "license": "mit"})
    assert out == "id=o/n lic=mit"
    with pytest.raises(KeyError):
        pub.render_model_card("$repo_id $missing", {"repo_id": "o/n"})


def test_stage_builds_a_clean_folder_with_card_and_manifest(tmp_path):
    model_dir = _fake_model_dir(tmp_path)
    out = tmp_path / "stage"
    manifest = pub.stage(str(model_dir), str(out), repo_id="owner/ja-pii-ner", license="apache-2.0", smoke=False, date="2026-10-04")
    names = sorted(os.listdir(out))
    assert names == sorted(list(pub.REQUIRED_FILES) + ["README.md", pub.MANIFEST_NAME])

    cfg = json.loads((out / "char_head_config.json").read_text(encoding="utf-8"))
    assert set(pub.INTERNAL_CONFIG_KEYS).isdisjoint(cfg) and "v11" not in json.dumps(cfg)
    assert cfg["char_dim"] == 128 and cfg["label_list"][0] == "O"

    assert manifest["files"]["char_head_config.json"]["sha256"] == pub.sha256_file(str(out / "char_head_config.json"))
    assert manifest["files"]["model.safetensors"]["bytes"] == 64
    dumped = json.dumps(manifest, ensure_ascii=False)
    assert "/Users" not in dumped and "v11" not in dumped and "avg_ep" not in dumped and "outputs_run" not in dumped
    assert manifest["smoke_test"] is None and manifest["label_list"][0] == "O"
    assert set(manifest) == {
        "repo_id", "license", "library", "library_version", "detector", "base_model", "base_model_revision",
        "label_list", "added_parameters", "files", "smoke_test", "staged_at",
    }
    assert list(manifest["files"]) == list(pub.REQUIRED_FILES)
    assert manifest["base_model"] == "cl-nagoya/ruri-v3-30m" and len(manifest["base_model_revision"]) == 40

    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n") and "license: apache-2.0" in card and "base_model: cl-nagoya/ruri-v3-30m" in card
    assert 'CharHeadNerDetector(model="owner/ja-pii-ner"' in card
    assert "$" not in card                                              # 置き換え漏れなし
    assert "人名の役割ヘッド" not in card and "person_role" not in card
    assert "v11" not in card and "avg_ep" not in card and "epoch0" not in card   # 第三者向け: 内部の版名を出さない
    assert "github.com/NishidaAzuharu/privacy-masker" in card            # ライブラリへの導線
    assert manifest["files"]["model.safetensors"]["sha256"] in card     # SHA 表が埋まっている
    assert pub.scan_text_files(str(out)) == []


def test_stage_refuses_missing_files_nonempty_out_and_unknown_license(tmp_path):
    model_dir = _fake_model_dir(tmp_path)
    (model_dir / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="tokenizer.json"):
        pub.stage(str(model_dir), str(tmp_path / "s1"), repo_id="o/n", license="mit", smoke=False)
    model_dir2 = _fake_model_dir(tmp_path / "second")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(SystemExit, match="空ではありません"):
        pub.stage(str(model_dir2), str(busy), repo_id="o/n", license="mit", smoke=False)
    with pytest.raises(SystemExit, match="license"):
        pub.stage(str(model_dir2), str(tmp_path / "s3"), repo_id="o/n", license="gpl-3.0", smoke=False)


def test_stage_stops_when_a_local_path_or_email_survives(tmp_path):
    model_dir = _fake_model_dir(tmp_path)
    template = tmp_path / "t.md"
    template.write_text("---\nlicense: $license\n---\n$repo_id $version $date $base_revision\n$sha_table\ncontact: someone@example.com\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="e-mail address"):
        pub.stage(str(model_dir), str(tmp_path / "s"), repo_id="o/n", license="mit", template_path=str(template), smoke=False)


def test_main_dry_run_does_not_upload(tmp_path, monkeypatch):
    model_dir = _fake_model_dir(tmp_path)
    monkeypatch.setattr(pub, "upload", lambda *a, **k: pytest.fail("dry-run must not upload"))
    rc = pub.main(["--repo-id", "owner/ja-pii-ner", "--model", str(model_dir), "--out", str(tmp_path / "stage"),
                   "--dry-run", "--no-smoke", "--license", "mit"])
    assert rc == 0 and (tmp_path / "stage" / "README.md").exists()


def test_main_uploads_after_staging(tmp_path, monkeypatch):
    model_dir = _fake_model_dir(tmp_path)
    seen = {}

    def fake_upload(out_dir, repo_id, *, private, commit_message):
        seen.update(out_dir=out_dir, repo_id=repo_id, private=private, commit_message=commit_message)
        return f"https://huggingface.co/{repo_id}"

    monkeypatch.setattr(pub, "upload", fake_upload)
    rc = pub.main(["--repo-id", "owner/ja-pii-ner", "--model", str(model_dir), "--out", str(tmp_path / "stage"), "--no-smoke", "--private"])
    assert rc == 0 and seen["repo_id"] == "owner/ja-pii-ner" and seen["private"] is True
    assert seen["out_dir"] == str(tmp_path / "stage") and "privacy-masker" in seen["commit_message"] and "v11" not in seen["commit_message"]


def test_help_runs_without_drive(tmp_path):
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root, PRIVACY_MASKER_DRIVE_DIR="/nonexistent")
    r = subprocess.run([sys.executable, os.path.join(root, "tools", "publish_model_hf.py"), "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "--repo-id" in r.stdout and "--dry-run" in r.stdout
    assert "--person-role-head DIR" in r.stdout


def _write_role_head(model_dir, head_dir, *, head_type="linear", metadata=None):
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors.numpy")
    head_dir.mkdir(exist_ok=True)
    dim = 3 * json.loads((model_dir / "config.json").read_text())["hidden_size"]
    shapes = ({"weight": (2, dim), "bias": (2,)} if head_type == "linear" else
              {"w1": (3, dim), "b1": (3,), "w2": (2, 3), "b2": (2,)})
    weights = {f"person.{key}": np.zeros(shape, dtype="float32") for key, shape in shapes.items()}
    st.save_file(weights, str(head_dir / pub.PERSON_ROLE_FILES[1]), metadata=metadata)
    cfg = {
        "format_version": 1, "head_type": head_type, "features": "span_ctx", "threshold": 0.75,
        "labels": {"person": ["private", "public"]},
        "encoder_sha256": pub.sha256_file(str(model_dir / "model.safetensors")),
    }
    (head_dir / pub.PERSON_ROLE_FILES[0]).write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def test_card_without_role_head_is_byte_identical_to_original_template():
    template = Path(pub.DEFAULT_TEMPLATE).read_text(encoding="utf-8")
    keys = ("repo_id", "license", "version", "date", "base_revision", "sha_table")
    unchanged = pub.render_model_card(template, {key: "$" + key for key in keys})
    # 役割ヘッド節の追加前のテンプレート全文。空行も含めて互換性を検査する。
    assert hashlib.sha256(unchanged.encode()).hexdigest() == "ccef1040206b2af13b28b9118e3c6f5f518edeb396063e939b5400c2ace000f6"


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_stage_includes_role_head_files_manifest_and_card(tmp_path, monkeypatch, head_type):
    from privacy_masker import PersonRoleHead

    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    cfg = _write_role_head(model, head, head_type=head_type)
    out = tmp_path / "stage"
    load = PersonRoleHead.load
    loaded = []

    def checked_load(path):
        assert Path(path) == out
        loaded.append(path)
        return load(path)

    monkeypatch.setattr(PersonRoleHead, "load", checked_load)
    manifest = pub.stage(str(model), str(out), repo_id="o/n", license="mit", smoke=False, person_role_head=str(head))
    assert len(loaded) == 1
    assert set(p.name for p in out.iterdir()) == set(pub.REQUIRED_FILES + pub.PERSON_ROLE_FILES + ("README.md", pub.MANIFEST_NAME))
    for name in pub.PERSON_ROLE_FILES:
        assert (out / name).read_bytes() == (head / name).read_bytes()
        assert manifest["files"][name] == {"bytes": (out / name).stat().st_size, "sha256": pub.sha256_file(str(out / name))}
    assert manifest["person_role_head"] == {key: cfg[key] for key in ("features", "threshold", "head_type", "format_version")}
    assert json.loads((out / pub.MANIFEST_NAME).read_text()) == manifest
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "## 人名の役割ヘッド（任意）" in card
    assert "**0.75**" in card and "person_role_threshold=0.8" in card
    assert 'CharHeadNerDetector(model="o/n")' in card
    assert "knowledge=PublicKnowledge.load(), person_role=ner.person_role_head" in card
    assert 'annotators=[annotator], policy=Policy.preset("standard")' in card
    assert "gpt-6.1-terra" in card and "エンコーダを凍結" in card and "弱いラベル" in card
    assert "既定では使いません" in card and "公人を隠す" in card
    assert "$" not in card and "person_role_eval" not in card and "<!-- person_role_head:" not in card
    assert pub.scan_text_files(str(out)) == []


@pytest.mark.parametrize("filename", pub.PERSON_ROLE_FILES)
def test_stage_requires_both_role_files(tmp_path, filename):
    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    _write_role_head(model, head)
    (head / filename).unlink()
    with pytest.raises(SystemExit, match=filename):
        pub.stage(str(model), str(tmp_path / "stage"), repo_id="o/n", license="mit", smoke=False, person_role_head=str(head))


@pytest.mark.parametrize("smoke", [False, True])
def test_stage_rejects_role_encoder_mismatch_before_smoke(tmp_path, monkeypatch, smoke):
    from privacy_masker.exceptions import ConfigurationError

    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    _write_role_head(model, head)
    (model / "model.safetensors").write_bytes(b"different-encoder")
    monkeypatch.setattr(pub, "smoke_test", lambda *a, **k: pytest.fail("encoder must be checked first"))
    with pytest.raises(ConfigurationError, match="encoder SHA-256 mismatch"):
        pub.stage(str(model), str(tmp_path / "stage"), repo_id="o/n", license="mit", smoke=smoke, person_role_head=str(head))


@pytest.mark.parametrize("filename", pub.PERSON_ROLE_FILES)
def test_stage_scans_role_config_and_weights_metadata(tmp_path, filename):
    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    cfg = _write_role_head(model, head, metadata={"source": DEV_PREFIX})
    if filename.endswith(".json"):
        _write_role_head(model, head)
        cfg["source"] = DEV_PREFIX
        (head / filename).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match=filename) as exc:
        pub.stage(str(model), str(tmp_path / "stage"), repo_id="o/n", license="mit", smoke=False, person_role_head=str(head))
    assert "macOS home path" in str(exc.value) and "e-mail address" in str(exc.value)
    assert DEV_PREFIX not in str(exc.value)


def test_role_files_in_source_are_not_published_without_option(tmp_path):
    model = _fake_model_dir(tmp_path)
    out = tmp_path / "before"
    args = dict(repo_id="o/n", license="mit", smoke=False, date="2026-10-04")
    pub.stage(str(model), str(out), **args)
    _write_role_head(model, model)
    after = tmp_path / "after"
    pub.stage(str(model), str(after), **args)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {p.name: p.read_bytes() for p in after.iterdir()}


def test_main_role_dry_run_does_not_upload(tmp_path, monkeypatch):
    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    _write_role_head(model, head)
    monkeypatch.setattr(pub, "upload", lambda *a, **k: pytest.fail("dry-run must not upload"))
    out = tmp_path / "stage"
    assert pub.main(["--repo-id", "o/n", "--model", str(model), "--person-role-head", str(head),
                     "--out", str(out), "--dry-run", "--no-smoke"]) == 0
    assert "person_role_head" in json.loads((out / pub.MANIFEST_NAME).read_text())


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_role_smoke_runs_detector_on_staged_model_with_context_vectors(tmp_path, monkeypatch, head_type):
    pytest.importorskip("torch")
    import privacy_masker.char_head as ch
    import privacy_masker.detectors as detectors
    from tests.test_char_head_vectors import HiddenRuleModel
    from tests.test_char_head_predict import TinyTok

    model = _fake_model_dir(tmp_path)
    head = tmp_path / "head"
    _write_role_head(model, head, head_type=head_type)
    out = tmp_path / "stage"
    text = "あいうえおかきくけこ。"
    monkeypatch.setattr(pub, "SMOKE_TEXT", text)
    tok = TinyTok(text)
    synthetic = HiddenRuleModel(text, tok, [(0, 2, "PERSON"), (3, 5, "ADDRESS")])
    real_detector = detectors.CharHeadNerDetector
    seen = []

    def detector(**kwargs):
        assert kwargs == {"model": str(out), "device": "cpu", "person_role_head": True}
        seen.append(kwargs)
        return real_detector(tokenizer=tok, **kwargs)

    monkeypatch.setattr(ch, "load_char_model", lambda *a, **k: synthetic)
    monkeypatch.setattr(detectors, "CharHeadNerDetector", detector)
    manifest = pub.stage(str(model), str(out), repo_id="o/n", license="mit", person_role_head=str(head))
    assert len(seen) == 1
    assert manifest["smoke_test"]["person_role_probabilities"] == [0.5]
    assert {e[0] for e in manifest["smoke_test"]["entities"]} == pub.SMOKE_REQUIRED_LABELS


@pytest.mark.parametrize("probability", [None, -0.1, 1.1, float("nan"), float("inf"), 0.0, 1.0])
def test_role_smoke_checks_probability_range(monkeypatch, probability):
    import privacy_masker.detectors as detectors

    entities = [SimpleNamespace(entity_type=label, start=i, end=i + 1, score=1, vector=b"vector")
                for i, label in enumerate(sorted(pub.SMOKE_REQUIRED_LABELS))]
    det = SimpleNamespace(detect=lambda _: entities, person_role_head=lambda vector: probability)
    monkeypatch.setattr(detectors, "CharHeadNerDetector", lambda **kwargs: det)
    if probability in (0.0, 1.0):
        assert pub.smoke_test("stage", person_role_head=True)["person_role_probabilities"] == [probability]
    else:
        with pytest.raises(SystemExit, match="役割確率"):
            pub.smoke_test("stage", person_role_head=True)


def test_smoke_without_option_keeps_original_result(monkeypatch):
    import privacy_masker.detectors as detectors

    entities = [SimpleNamespace(entity_type=label, start=i, end=i + 1, score=1)
                for i, label in enumerate(sorted(pub.SMOKE_REQUIRED_LABELS))]

    def detector(**kwargs):
        assert kwargs == {"model": "model", "device": "cpu"}
        return SimpleNamespace(detect=lambda _: entities)

    monkeypatch.setattr(detectors, "CharHeadNerDetector", detector)
    assert pub.smoke_test("model") == {
        "text": pub.SMOKE_TEXT, "entities": [[e.entity_type, e.start, e.end, 1] for e in entities],
    }

"""tools/publish_model_hf.py: ステージングの組み立て・学習機パスの消去・禁止パターン検査・dry-run。ネットワークと本番モデルは使わない。"""
from __future__ import annotations

import json
import os
import sys

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
    assert manifest["base_model"] == "cl-nagoya/ruri-v3-30m" and len(manifest["base_model_revision"]) == 40

    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n") and "license: apache-2.0" in card and "base_model: cl-nagoya/ruri-v3-30m" in card
    assert 'CharHeadNerDetector(model="owner/ja-pii-ner"' in card
    assert "$" not in card                                              # 置き換え漏れなし
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

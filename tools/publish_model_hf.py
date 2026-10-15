"""本番の文字ヘッド NER モデルを Hugging Face Hub に公開する（ステージング → 検査 → 任意でアップロード）。

  PYTHONPATH=. .venv/bin/python tools/publish_model_hf.py --repo-id OWNER/NAME --dry-run --out /tmp/hf_stage   # 置く内容を作って確認するだけ
  PYTHONPATH=. .venv/bin/python tools/publish_model_hf.py --repo-id OWNER/NAME --private                        # 非公開リポジトリとして作成してアップロード
  PYTHONPATH=. .venv/bin/python tools/publish_model_hf.py --repo-id OWNER/NAME                                  # 公開リポジトリとして作成 / 更新

手順（止まる条件は fail-closed）:
  1. モデルフォルダ（既定: データフォルダ内の本番モデル。--model で指定）に推論に必要な 6 ファイルが揃っていることを確認する
  2. char_head_config.json から学習時の内部情報（averaged_from = 学習機の絶対パス（メールアドレス入り）、average_weights、data_tag、epoch）を外して書き出す
     （推論が読むキーではない。残った絶対パスがあればフォルダ名だけにする）
  3. 読み込んで 1 文を検出するスモークテスト（--no-smoke で省略。壊れたフォルダを上げない）
  4. README.md（モデルカード。docs/hub/model_card_template.md の $placeholder を埋める）と privacy_masker_model.json（由来と SHA-256）を生成する
  5. ステージング内の全テキストファイルに学習機のパス・メールアドレスが残っていないことを検査する（残っていれば止める）
  6. --dry-run でなければ create_repo(exist_ok=True) → upload_folder。認証は `hf auth login` か環境変数 HF_TOKEN
     （このスクリプトはトークンを引数で受け取らない）
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _paths import drive_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_SUBDIR = "outputs_ruri-v3-30m-charhead-v11-plus-old-frompre/avg_ep01-04"
BASE_MODEL = "cl-nagoya/ruri-v3-30m"
BASE_MODEL_REVISION = "24899e5de370b56d179604a007c0d727bf144504"   # 学習時に固定したベースモデルのリビジョン（学習ノートの ENCODER_REVISION）
REQUIRED_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "char_head.safetensors", "char_head_config.json",
)
DEFAULT_TEMPLATE = os.path.join(ROOT, "docs", "hub", "model_card_template.md")
MANIFEST_NAME = "privacy_masker_model.json"
# 学習時の内部情報。第三者向けの配布物には入れない（推論は読まない: load_char_model が使うのは char_dim / kernel_size / use_mix / num_labels / label_list / dropout / class_weights）
INTERNAL_CONFIG_KEYS = ("averaged_from", "average_weights", "data_tag", "epoch")
LICENSES = ("apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-4.0", "other")
SMOKE_TEXT = "山田太郎さんは東京都新宿区西新宿2-8-1に住んでいます。"      # 架空
SMOKE_REQUIRED_LABELS = {"PERSON", "ADDRESS"}

# ステージングに残ってはいけないもの（学習機の絶対パス・クラウドストレージのミラーのパス・メールアドレス）
FORBIDDEN_PATTERNS = (
    ("macOS home path", re.compile(r"/Users/[^/\s\"']+")),
    ("Linux home path", re.compile(r"/home/[^/\s\"']+")),
    ("cloud storage mirror path", re.compile(r"CloudStorage")),
    ("e-mail address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")),
)
TEXT_SUFFIXES = (".json", ".md", ".txt", ".yaml", ".yml", ".py", ".cfg", ".toml")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_local_paths(obj):
    """文字列の絶対パス（/… や ~/…）を末尾のフォルダ名だけにする。dict / list は再帰。他はそのまま。"""
    if isinstance(obj, str):
        if obj.startswith(("/", "~")):
            return os.path.basename(obj.rstrip("/")) or obj
        return obj
    if isinstance(obj, dict):
        return {k: strip_local_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_local_paths(v) for v in obj]
    return obj


def sanitize_char_head_config(cfg: dict) -> dict:
    """学習時の内部情報（``INTERNAL_CONFIG_KEYS``）を外し、残った絶対パスはフォルダ名だけにする。推論が読むキーは変えない。"""
    return strip_local_paths({k: v for k, v in cfg.items() if k not in INTERNAL_CONFIG_KEYS})


def scan_text_files(root: str) -> list[str]:
    """テキストファイルに禁止パターンが残っていないか。見つかったら「ファイル: 種類（件数）」を返す（中身は出力しない）。"""
    findings: list[str] = []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(TEXT_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for label, pat in FORBIDDEN_PATTERNS:
                n = len(pat.findall(text))
                if n:
                    findings.append(f"{os.path.relpath(path, root)}: {label} ({n})")
    return findings


def render_model_card(template: str, mapping: dict) -> str:
    """``$name`` を埋める。不足していれば KeyError（黙って空にしない）。"""
    return string.Template(template).substitute(mapping)


def sha_table(files: dict) -> str:
    return "\n".join(f"| `{name}` | {info['bytes']:,} | `{info['sha256']}` |" for name, info in files.items())


def library_version() -> str:
    try:
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:  # pragma: no cover
        from importlib.metadata import version

        return version("privacy-masker")


def smoke_test(model_dir: str) -> dict:
    """読み込んで 1 文を検出し、PERSON と ADDRESS が出ることを確かめる。出なければ止める。"""
    from privacy_masker.detectors import CharHeadNerDetector

    det = CharHeadNerDetector(model=model_dir, device="cpu")
    ents = det.detect(SMOKE_TEXT)
    labels = {e.entity_type for e in ents}
    missing = SMOKE_REQUIRED_LABELS - labels
    if missing:
        raise SystemExit(f"スモークテスト失敗: {sorted(missing)} が検出されませんでした（検出 {len(ents)} 件: {sorted(labels)}）")
    return {"text": SMOKE_TEXT, "entities": [[e.entity_type, e.start, e.end, round(e.score, 3)] for e in ents]}


def stage(model_dir: str, out_dir: str, *, repo_id: str, license: str, template_path: str = DEFAULT_TEMPLATE,
          smoke: bool = True, date: str | None = None) -> dict:
    """アップロードする内容を ``out_dir`` に組み立て、manifest を返す。``out_dir`` は無いか空であること。"""
    model_dir = os.path.abspath(model_dir)
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise SystemExit(f"モデルフォルダに必要なファイルがありません: {missing}（{model_dir}）")
    if license not in LICENSES:
        raise SystemExit(f"--license は {LICENSES} のどれかにしてください: {license}")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"出力先が空ではありません: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    smoke_result = smoke_test(model_dir) if smoke else None

    with open(os.path.join(model_dir, "char_head_config.json"), encoding="utf-8") as f:
        raw_cfg = json.load(f)
    cfg = sanitize_char_head_config(raw_cfg)
    for name in REQUIRED_FILES:
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        if name == "char_head_config.json":
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
                f.write("\n")
        else:
            shutil.copyfile(src, dst)

    files = {name: {"bytes": os.path.getsize(os.path.join(out_dir, name)), "sha256": sha256_file(os.path.join(out_dir, name))}
             for name in REQUIRED_FILES}
    version = library_version()
    date = date or dt.date.today().isoformat()
    manifest = {
        "repo_id": repo_id,
        "license": license,
        "library": "privacy-masker",
        "library_version": version,
        "detector": "privacy_masker.detectors.CharHeadNerDetector",
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "label_list": cfg.get("label_list"),
        "added_parameters": cfg.get("added_parameters"),
        "files": files,
        "smoke_test": smoke_result,
        "staged_at": date,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    card = render_model_card(template, {
        "repo_id": repo_id, "license": license, "version": version, "date": date,
        "base_revision": BASE_MODEL_REVISION, "sha_table": sha_table(files),
    })
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    findings = scan_text_files(out_dir)
    if findings:
        raise SystemExit("ステージングに学習機のパスやメールアドレスが残っています（アップロードしません）:\n  " + "\n  ".join(findings))
    return manifest


def upload(out_dir: str, repo_id: str, *, private: bool, commit_message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=out_dir, commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="Hub のリポジトリ ID（owner/name）")
    ap.add_argument("--model", default=None, help="モデルフォルダ（既定: データフォルダ PRIVACY_MASKER_DRIVE_DIR 内の本番モデル）")
    ap.add_argument("--license", default="apache-2.0", choices=LICENSES, help="重みのライセンス（既定 apache-2.0 = encoder の元モデルと同じ）")
    ap.add_argument("--out", default=None, help="ステージング先（既定: 一時フォルダ）。無いか空のフォルダ")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--private", action="store_true", help="非公開リポジトリとして作成する（後から Hub の設定で公開にできる）")
    ap.add_argument("--dry-run", action="store_true", help="ステージングと検査だけ行い、アップロードしない")
    ap.add_argument("--no-smoke", action="store_true", help="読み込み・検出のスモークテストを省略する")
    ap.add_argument("--commit-message", default=None)
    args = ap.parse_args(argv)

    model_dir = args.model or os.path.join(drive_dir(), PROD_SUBDIR)
    out_dir = args.out or tempfile.mkdtemp(prefix="privacy-masker-hf-")
    manifest = stage(model_dir, out_dir, repo_id=args.repo_id, license=args.license, template_path=args.template,
                     smoke=not args.no_smoke)
    print(f"staged: {out_dir}")
    for name, info in manifest["files"].items():
        print(f"  {name:<24} {info['bytes']:>12,}  {info['sha256'][:16]}…")
    print(f"  README.md / {MANIFEST_NAME}  (license {args.license}, library {manifest['library_version']})")
    if manifest["smoke_test"]:
        print(f"  smoke: {len(manifest['smoke_test']['entities'])} entities, labels {sorted({e[0] for e in manifest['smoke_test']['entities']})}")
    if args.dry_run:
        print("dry-run: アップロードしません")
        return 0
    msg = args.commit_message or f"privacy-masker {manifest['library_version']}: ruri-v3-30m char-head NER for Japanese PII detection"
    url = upload(out_dir, args.repo_id, private=args.private, commit_message=msg)
    print(f"uploaded: {url}" + ("  (private)" if args.private else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

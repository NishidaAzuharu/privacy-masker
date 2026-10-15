"""Hugging Face Hub 上のモデルをローカルのフォルダに解決する。

``CharHeadNerDetector(model=...)`` の ``model`` は、ローカルのフォルダか Hub のリポジトリ ID（``owner/name``）を受け取る。
このモジュールはその判定と取得だけを担う（``huggingface_hub.snapshot_download``。取得済みならキャッシュから返す）。

- 取得するのは推論に必要なファイルだけ（``*.json``・``*.safetensors``・``*.txt``・``*.model``）。
- 取れないとき（リポジトリや revision が無い、ネットワーク不通で未キャッシュ、認証が要る）は ``ConfigurationError`` で止める。
  空のフォルダや別のモデルで黙って続けない（検出器が空結果を返すと PII が丸ごと落ちる）。
- トークンはここでは扱わない。非公開リポジトリは ``hf auth login`` か環境変数 ``HF_TOKEN`` で huggingface_hub 側に持たせる。
"""
from __future__ import annotations

import os
import re
from typing import Any, Sequence

from .exceptions import ConfigurationError

# owner/name。先頭が英数字なので ./x/y・/a/b・~/x のようなパスは一致しない
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")

# 当方が Hub で配布している本番モデル（ruri-v3-30m + 文字ヘッド）。examples の既定に使う
DEFAULT_MODEL_ID = "az225/privacy-masker-ja-pii-ner-30m"

# 推論に要るものだけ取る（README や .gitattributes、学習ログは取らない）
MODEL_FILE_PATTERNS: tuple[str, ...] = ("*.json", "*.safetensors", "*.txt", "*.model")


def looks_like_repo_id(model: str) -> bool:
    """``owner/name`` の形で、ローカルにその名前のファイル・フォルダが無いもの。"""
    return bool(REPO_ID_RE.match(model)) and not os.path.exists(model)


def resolve_model_dir(
    model: Any,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    local_files_only: bool = False,
    allow_patterns: Sequence[str] = MODEL_FILE_PATTERNS,
) -> str:
    """``model`` をローカルのフォルダのパスにする。

    - 既存のフォルダ → そのまま返す
    - Hub のリポジトリ ID → ``snapshot_download`` で取得（キャッシュ済みなら再取得しない）し、そのフォルダを返す
    - どちらでもない → ``ConfigurationError``

    ``revision`` はブランチ・タグ・コミット（本番では固定を推奨）。``local_files_only=True`` はネットワークに出ない。
    """
    path = os.fsdecode(model) if isinstance(model, bytes) else os.fspath(model)
    if os.path.isdir(path):
        return path
    if not looks_like_repo_id(path):
        raise ConfigurationError(
            f"モデルのフォルダが見つかりません: {path}\n"
            "  → ローカルのフォルダか、Hugging Face Hub のリポジトリ ID（owner/name）を指定してください"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:   # pragma: no cover - transformers extra なしで Hub ID を渡した場合
        raise ConfigurationError(
            "Hugging Face Hub からの取得には huggingface_hub が必要です: pip install 'privacy-masker[transformers]'"
        ) from e
    try:
        return snapshot_download(
            repo_id=path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=list(allow_patterns),
        )
    except Exception as e:
        # RepositoryNotFoundError / RevisionNotFoundError / GatedRepoError / HfHubHTTPError /
        # LocalEntryNotFoundError（オフラインで未キャッシュ）/ OfflineModeIsEnabled など。原因は from で保つ
        rev = f"（revision {revision}）" if revision else ""
        raise ConfigurationError(
            f"Hugging Face Hub からモデル {path!r}{rev} を取得できませんでした: {type(e).__name__}: {e}\n"
            "  → リポジトリ ID・revision・ネットワークを確認してください。非公開リポジトリなら hf auth login か HF_TOKEN が要ります"
        ) from e

"""ツール共通のパス解決。環境変数で差し替える。

  PRIVACY_MASKER_DRIVE_DIR   データフォルダ（学習データ・モデル・評価セット・公知データの置き場。ローカルのフォルダでも、クラウドストレージのミラーでもよい）
  PRIVACY_MASKER_NOTEBOOKS   Colab ノートブック置き場（学習・生成ノートブックの生成先、ローカル実行の読み込み元）

未設定なら ~/privacy-masker-data（ノートブックは ~/privacy-masker-data/notebooks）を使う。データフォルダが存在しなければ分かりやすく止める。
"""
from __future__ import annotations

import os

_DEFAULT_DRIVE = "~/privacy-masker-data"
_DEFAULT_NOTEBOOKS = "~/privacy-masker-data/notebooks"


def drive_dir(require: bool = True) -> str:
    path = os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", _DEFAULT_DRIVE))
    if require and not os.path.isdir(path):
        raise SystemExit(
            f"データフォルダが見つかりません: {path}\n"
            "  環境変数 PRIVACY_MASKER_DRIVE_DIR にデータフォルダ（モデル・評価セット・公知データの置き場）のパスを設定してください"
        )
    return path


def notebooks_dir() -> str:
    return os.path.expanduser(os.environ.get("PRIVACY_MASKER_NOTEBOOKS", _DEFAULT_NOTEBOOKS))

"""privacy-masker の例外階層。

例外メッセージには入力テキスト・検出値・鍵を一切含めないこと（スペック§12）。
"""


class PrivacyMaskerError(Exception):
    """全例外の基底。"""


class ConfigurationError(PrivacyMaskerError):
    """構成・引数の組み合わせ不整合。"""


class DetectorError(PrivacyMaskerError):
    """検出器の実行時失敗・不正なEntityの返却。"""


class AnnotatorError(PrivacyMaskerError):
    """Annotatorの実行時失敗・不正なEntityの返却。"""


class StoreError(PrivacyMaskerError):
    """Store I/O失敗・一意性制約違反。"""


class RestoreError(PrivacyMaskerError):
    """復元失敗（復号失敗・コンテキスト不一致等）。"""

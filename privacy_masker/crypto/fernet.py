from __future__ import annotations

from ..exceptions import RestoreError


def _require_fernet():
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "FernetProvider requires the 'cryptography' package. "
            "Install with: pip install 'privacy-masker[crypto]'"
        ) from e
    return Fernet, InvalidToken


class FernetProvider:
    """高レベルAEAD（Fernet）による対応表暗号化。低レベルAPIは公開しない（スペック§10）。"""

    def __init__(self, key: bytes | str) -> None:
        Fernet, _ = _require_fernet()
        self._fernet = Fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        Fernet, _ = _require_fernet()
        return Fernet.generate_key()

    def encrypt(self, value: bytes) -> bytes:
        return self._fernet.encrypt(value)

    def decrypt(self, value: bytes) -> bytes:
        _, InvalidToken = _require_fernet()
        try:
            return self._fernet.decrypt(value)
        except InvalidToken as e:
            raise RestoreError("decryption failed") from e

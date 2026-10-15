import pytest

from privacy_masker.exceptions import RestoreError

from .conftest import FakeCrypto


def test_fernet_roundtrip():
    pytest.importorskip("cryptography", reason="requires [crypto] extra")
    from privacy_masker.crypto import FernetProvider

    key = FernetProvider.generate_key()
    p = FernetProvider(key)
    blob = p.encrypt(b"secret-payload")
    assert blob != b"secret-payload"
    assert p.decrypt(blob) == b"secret-payload"


def test_fernet_tampered_raises_restore_error():
    pytest.importorskip("cryptography", reason="requires [crypto] extra")
    from privacy_masker.crypto import FernetProvider

    p = FernetProvider(FernetProvider.generate_key())
    with pytest.raises(RestoreError):
        p.decrypt(b"not-a-valid-token")


def test_fake_crypto_roundtrip():
    # conftestのFakeCryptoがProtocolを満たすことをsanity check
    f = FakeCrypto()
    assert f.decrypt(f.encrypt(b"abc")) == b"abc"
    with pytest.raises(RestoreError):
        f.decrypt(b"garbage")

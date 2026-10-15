import pytest

from privacy_masker.encoding import decode_fields, encode_fields, hmac_token_id, value_hash
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.normalize import default_normalize
from privacy_masker.secrets_types import SecretBytes

KEY = b"test-secret-key"


def test_encode_fields_is_injective():
    # スペック§7: 区切りなし連結による境界の曖昧さを防ぐ
    assert encode_fields("case_1", "PERSON", "x") != encode_fields("case_1P", "ERSON", "x")


def test_encode_decode_roundtrip_multibyte():
    fields = ["scope_あ", "PERSON", "山田　太郎"]
    assert decode_fields(encode_fields(*fields)) == fields


def test_decode_fields_rejects_truncated():
    data = encode_fields("abc")
    with pytest.raises(ValueError):
        decode_fields(data[:-1])


def test_value_hash_deterministic_and_scope_sensitive():
    a = value_hash(KEY, "s1", "PERSON", "山田太郎")
    assert a == value_hash(KEY, "s1", "PERSON", "山田太郎")
    assert a != value_hash(KEY, "s2", "PERSON", "山田太郎")
    assert a != value_hash(b"other-key", "s1", "PERSON", "山田太郎")
    int(a, 16)  # hex文字列


def test_hmac_token_id_charset_and_length():
    tid = hmac_token_id(KEY, "", "PERSON", "山田太郎", length=8)
    assert len(tid) == 8
    assert set(tid) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")  # RFC 4648 base32
    assert tid == hmac_token_id(KEY, "", "PERSON", "山田太郎", length=8)
    assert tid != hmac_token_id(KEY, "s1", "PERSON", "山田太郎", length=8)


def test_secret_bytes_hides_value():
    s = SecretBytes(b"super-secret")
    assert "super-secret" not in repr(s)
    assert "super-secret" not in str(s)
    assert s.reveal() == b"super-secret"


def test_secret_bytes_rejects_empty():
    with pytest.raises(ConfigurationError):
        SecretBytes(b"")


def test_secret_bytes_from_env(monkeypatch):
    monkeypatch.setenv("PM_TEST_KEY", "env-secret")
    assert SecretBytes.from_env("PM_TEST_KEY").reveal() == b"env-secret"
    monkeypatch.delenv("PM_TEST_KEY")
    with pytest.raises(ConfigurationError):
        SecretBytes.from_env("PM_TEST_KEY")


def test_default_normalize():
    # スペック§7の例: 山田太郎 と 山田　太郎 を同一視するため空白は全除去
    assert default_normalize("山田　太郎") == "山田太郎"
    assert default_normalize("  山田 太郎\n") == "山田太郎"
    assert default_normalize("ＡＢＣ１２３") == "ABC123"  # NFKC

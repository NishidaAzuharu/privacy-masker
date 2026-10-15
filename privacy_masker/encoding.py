"""フィールドの単射エンコードとHMAC計算（スペック§7・§9）。"""
from __future__ import annotations

import base64
import hashlib
import hmac


def encode_fields(*fields: str) -> bytes:
    """各フィールドをUTF-8化し8バイトbig-endian長プレフィックスを付けて連結する。"""
    out = bytearray()
    for f in fields:
        b = f.encode("utf-8")
        out += len(b).to_bytes(8, "big")
        out += b
    return bytes(out)


def decode_fields(data: bytes) -> list[str]:
    fields: list[str] = []
    i = 0
    while i < len(data):
        if i + 8 > len(data):
            raise ValueError("truncated field header")
        n = int.from_bytes(data[i : i + 8], "big")
        i += 8
        if i + n > len(data):
            raise ValueError("truncated field body")
        fields.append(data[i : i + n].decode("utf-8"))
        i += n
    return fields


def value_hash(secret_key: bytes, scope_id: str, entity_type: str, normalized_value: str) -> str:
    """Storeのキーに使う秘密鍵付きハッシュ（スペック§9）。"""
    mac = hmac.new(secret_key, encode_fields(scope_id, entity_type, normalized_value), hashlib.sha256)
    return mac.hexdigest()


def hmac_token_id(
    key: bytes, scope_component: str, entity_type: str, normalized_value: str, length: int = 8
) -> str:
    """hmac戦略のトークンID部（スペック§7）。base32（RFC 4648）先頭length文字。"""
    digest = hmac.new(
        key, encode_fields(scope_component, entity_type, normalized_value), hashlib.sha256
    ).digest()
    return base64.b32encode(digest).decode("ascii")[:length]

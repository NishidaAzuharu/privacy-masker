from __future__ import annotations

import threading

from ..exceptions import StoreError


class MemoryStore:
    """プロセス内メモリのStore。スレッドセーフ。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[tuple[str, str, str], str] = {}
        self._values: dict[tuple[str, str], bytes | None] = {}
        self._seq: dict[tuple[str, str], int] = {}

    def get_token(self, scope_id: str, entity_type: str, value_hash: str) -> str | None:
        with self._lock:
            return self._tokens.get((scope_id, entity_type, value_hash))

    def put_token(
        self, scope_id: str, entity_type: str, value_hash: str,
        token: str, encrypted_value: bytes | None,
    ) -> None:
        with self._lock:
            if (scope_id, entity_type, value_hash) in self._tokens:
                raise StoreError("store uniqueness constraint violated")
            if (scope_id, token) in self._values:
                raise StoreError("store uniqueness constraint violated")
            self._tokens[(scope_id, entity_type, value_hash)] = token
            self._values[(scope_id, token)] = encrypted_value

    def get_value(self, scope_id: str, token: str) -> bytes | None:
        with self._lock:
            return self._values.get((scope_id, token))

    def next_seq(self, scope_id: str, entity_type: str) -> int:
        with self._lock:
            n = self._seq.get((scope_id, entity_type), 0) + 1
            self._seq[(scope_id, entity_type)] = n
            return n

    def delete_scope(self, scope_id: str) -> None:
        with self._lock:
            self._tokens = {k: v for k, v in self._tokens.items() if k[0] != scope_id}
            self._values = {k: v for k, v in self._values.items() if k[0] != scope_id}
            self._seq = {k: v for k, v in self._seq.items() if k[0] != scope_id}

    def iter_tokens(self, scope_id: str) -> list[tuple[str, str]]:
        """scope に登録済みの (entity_type, token) を登録順に返す。"""
        with self._lock:
            return [(k[1], v) for k, v in self._tokens.items() if k[0] == scope_id]

"""PseudonymStore Protocol（スペック§9）。"""
from __future__ import annotations

from typing import Iterable, Protocol


class PseudonymStore(Protocol):
    def get_token(self, scope_id: str, entity_type: str, value_hash: str) -> str | None: ...

    def put_token(
        self, scope_id: str, entity_type: str, value_hash: str,
        token: str, encrypted_value: bytes | None,
    ) -> None: ...

    def get_value(self, scope_id: str, token: str) -> bytes | None: ...

    def next_seq(self, scope_id: str, entity_type: str) -> int: ...

    def delete_scope(self, scope_id: str) -> None: ...

    def iter_tokens(self, scope_id: str) -> Iterable[tuple[str, str]]:
        """scope に登録済みの (entity_type, token) を登録順に返す（v0.9。Surrogate(vault) の復元が使う）。"""
        ...

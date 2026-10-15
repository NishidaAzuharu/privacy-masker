from __future__ import annotations

import sqlite3
import threading

from ..exceptions import StoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
    scope_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    token TEXT NOT NULL,
    encrypted_value BLOB,
    PRIMARY KEY (scope_id, entity_type, value_hash),
    UNIQUE (scope_id, token)
);
CREATE TABLE IF NOT EXISTS counters (
    scope_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    n INTEGER NOT NULL,
    PRIMARY KEY (scope_id, entity_type)
);
"""


class SQLiteStore:
    """SQLite永続Store。WAL + プロセス内ロックで同一プロセス多スレッドまで保証（スペック§9）。"""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_token(self, scope_id: str, entity_type: str, value_hash: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT token FROM tokens WHERE scope_id=? AND entity_type=? AND value_hash=?",
                (scope_id, entity_type, value_hash),
            ).fetchone()
        return row[0] if row else None

    def put_token(
        self, scope_id: str, entity_type: str, value_hash: str,
        token: str, encrypted_value: bytes | None,
    ) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO tokens (scope_id, entity_type, value_hash, token, encrypted_value)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (scope_id, entity_type, value_hash, token, encrypted_value),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise StoreError("store uniqueness constraint violated") from e

    def get_value(self, scope_id: str, token: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT encrypted_value FROM tokens WHERE scope_id=? AND token=?",
                (scope_id, token),
            ).fetchone()
        return row[0] if row else None

    def next_seq(self, scope_id: str, entity_type: str) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO counters (scope_id, entity_type, n) VALUES (?, ?, 1)"
                " ON CONFLICT(scope_id, entity_type) DO UPDATE SET n = n + 1",
                (scope_id, entity_type),
            )
            row = self._conn.execute(
                "SELECT n FROM counters WHERE scope_id=? AND entity_type=?",
                (scope_id, entity_type),
            ).fetchone()
            self._conn.commit()
        return int(row[0])

    def delete_scope(self, scope_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tokens WHERE scope_id=?", (scope_id,))
            self._conn.execute("DELETE FROM counters WHERE scope_id=?", (scope_id,))
            self._conn.commit()

    def iter_tokens(self, scope_id: str) -> list[tuple[str, str]]:
        """scope に登録済みの (entity_type, token) を登録順に返す。"""
        with self._lock:
            rows = self._conn.execute(
                # rowid は挿入順（AUTOINCREMENT は使っていないが削除後の再利用は scope 単位の
                # DELETE しか無いので登録順として扱える）
                "SELECT entity_type, token FROM tokens WHERE scope_id=? ORDER BY rowid",
                (scope_id,),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self._conn.close()

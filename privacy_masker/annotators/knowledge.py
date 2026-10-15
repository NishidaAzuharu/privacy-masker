"""PublicKnowledge: 公知の表層集合（Wikipedia 見出し語など）を 64bit ハッシュで保持する。

- 照合は「NFKC 正規化 + 空白を 1 個に圧縮した表層」と「空白を全除去した表層」の 2 通りで行う
  （見出し語「岸田 文雄」と本文「岸田文雄」を同一視する）
- 保存形式（.pmk）: magic ``PMK1`` (4B) + 件数 uint64 LE (8B) + ソート済み uint64 LE 配列
- 偽陽性は「私人を公知と誤る = 漏れ方向」なので Bloom フィルタは使わず、正確なハッシュ集合にする
  （blake2b 64bit の偶然衝突は 246 万件で約 1.6e-7 であり実用上無視できる）
- メモリは 8B/件（246 万件で約 20MB）。照合は二分探索 O(log n)
"""
from __future__ import annotations

import bisect
import gzip
import hashlib
import os
import re
import struct
import sys
import unicodedata
from array import array
from typing import Iterable, Iterator

from ..exceptions import ConfigurationError

_MAGIC = b"PMK1"
_HEADER = struct.Struct("<Q")
_WS = re.compile(r"\s+")
_TITLE_HEADER = "page_title"  # Wikipedia のタイトルダンプ先頭行
ENV_VAR = "PRIVACY_MASKER_KNOWLEDGE"


def normalize_surface(surface: str) -> str:
    """NFKC 正規化し、Wikipedia 見出し語の ``_`` を空白に戻し、空白を 1 個に圧縮する。"""
    s = unicodedata.normalize("NFKC", surface).replace("_", " ")
    return _WS.sub(" ", s).strip()


def _h64(normalized: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest(), "little"
    )


def _iter_titles(path: str | os.PathLike[str]) -> Iterator[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:  # type: ignore[operator]
        first = True
        for line in f:
            line = line.rstrip("\n")
            if first:
                first = False
                if line == _TITLE_HEADER:
                    continue
            if line:
                yield line


class PublicKnowledge:
    """公知表層の集合。``surface in knowledge`` で照合する。"""

    def __init__(self, hashes: array) -> None:
        if hashes.typecode != "Q":
            raise ConfigurationError("PublicKnowledge requires an array('Q') of hashes")
        self._h = hashes

    # ---- 構築 ----

    @classmethod
    def from_iterable(cls, surfaces: Iterable[str]) -> "PublicKnowledge":
        hashes: set[int] = set()
        for s in surfaces:
            n = normalize_surface(s)
            if not n:
                continue
            hashes.add(_h64(n))
            compact = n.replace(" ", "")
            if compact != n:
                hashes.add(_h64(compact))  # 「岸田 文雄」を本文の「岸田文雄」でも引けるようにする
        return cls(array("Q", sorted(hashes)))

    @classmethod
    def from_titles_file(cls, path: str | os.PathLike[str]) -> "PublicKnowledge":
        """見出し語ファイル（1 行 1 見出し。``.gz`` 可。先頭行 ``page_title`` は無視）から構築する。"""
        return cls.from_iterable(_iter_titles(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "PublicKnowledge":
        """``save()`` した .pmk を読む。path 省略時は環境変数 PRIVACY_MASKER_KNOWLEDGE を使う。"""
        if path is None:
            path = os.environ.get(ENV_VAR)
            if not path:
                raise ConfigurationError(
                    f"PublicKnowledge.load() requires a path or the {ENV_VAR} environment variable"
                )
        with open(path, "rb") as f:
            head = f.read(4 + _HEADER.size)
            if len(head) != 4 + _HEADER.size or head[:4] != _MAGIC:
                raise ConfigurationError("not a PublicKnowledge (.pmk) file")
            (count,) = _HEADER.unpack(head[4:])
            arr = array("Q")
            arr.frombytes(f.read(count * arr.itemsize))
        if len(arr) != count:
            raise ConfigurationError("truncated PublicKnowledge (.pmk) file")
        if sys.byteorder != "little":  # pragma: no cover
            arr.byteswap()
        return cls(arr)

    def save(self, path: str | os.PathLike[str]) -> None:
        arr = self._h
        if sys.byteorder != "little":  # pragma: no cover
            arr = array("Q", arr)
            arr.byteswap()
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(_HEADER.pack(len(arr)))
            f.write(arr.tobytes())

    # ---- 照合 ----

    def __len__(self) -> int:
        return len(self._h)

    def _has(self, h: int) -> bool:
        i = bisect.bisect_left(self._h, h)
        return i < len(self._h) and self._h[i] == h

    def __contains__(self, surface: object) -> bool:
        if not isinstance(surface, str):
            return False
        n = normalize_surface(surface)
        if not n:
            return False
        if self._has(_h64(n)):
            return True
        compact = n.replace(" ", "")
        return compact != n and self._has(_h64(compact))

    def __repr__(self) -> str:
        return f"PublicKnowledge(entries={len(self._h)})"

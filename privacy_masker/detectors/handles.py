"""HandleDetector: チャット・ログ・SNS のハンドル名（ユーザー ID）を規則で拾う（v0.8.1）。

NER モデルは ``yuki_t`` ``hamada.k`` ``@haruka_ig`` のような表層を学習していない（盲検評価 v1 では、この型が PERSON の危険見逃しの約 2 割を占めた）。人名と同じく個人を指すので ``USERNAME`` 型で出す。

拾うもの:
  - ``@handle``（メールアドレスの一部は除く。@ 自体はスパンに含めない）
  - 行頭の話者ラベル ``name:`` / ``[09:12] name:`` / ``12:03 name:`` / ``Agent (name):``。
    ヘッダ語（Subject: / To: / Tel: …）と 1 文字ラベル（SOAP の S: など）は除く。ラベルは
    (a) ``_`` ``.`` 数字を含んでハンドルらしいか、(b) 同じ文書で 2 回以上（行頭ラベルか @メンションで）出る（会話の話者）ときだけ採用する
  - ``user=`` ``login:`` ``by=`` ``checked_by=`` などのキーに続く英字の値（ログ・チケット）

漢字・かなの話者名は NER が拾うので対象外。日本語のハンドル（``sora@出品``）も対象外。
"""
from __future__ import annotations

import re
from typing import Iterable

from ..types import Entity

ENTITY_TYPE = "USERNAME"

_AT = re.compile(r"(?<![A-Za-z0-9._%+\-])@([A-Za-z][A-Za-z0-9_.]{1,30})(?![A-Za-z0-9_.@])")
_LABEL = re.compile(
    r"^(?P<indent>[ \t]*)(?P<prefix>\[\d{1,2}:\d{2}(?::\d{2})?\][ \t]*|\d{1,2}:\d{2}[ \t]+|[-*•][ \t]*)?"
    r"(?P<label>[A-Za-z][A-Za-z0-9_.]{1,30})[ \t]*[:：](?=\s)(?P<rest>[^\n]*)", re.M)
_ASCII_TOKEN = re.compile(r"^[ \t]*[A-Za-z0-9_.@/-]*[ \t]*$")
_PAREN_LABEL = re.compile(r"^[ \t]*[A-Za-z][A-Za-z ]{0,20}\(([A-Za-z][A-Za-z0-9_.]{1,30})\)[ \t]*[:：]", re.M)
_KV = re.compile(
    r"(?<![A-Za-z0-9_])(?:user(?:_?name|_?id)?|login|uid|checked_by|by|owner|author|assignee|reporter|"
    r"account|handle|nick(?:name)?|screen_name)[ \t]*[=:][ \t]*\"?([A-Za-z][A-Za-z0-9_.@-]{1,40})", re.I)
_HANDLE_LIKE = re.compile(r"[_.\d]")

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    "subject to cc bcc from date sent re fw fwd tel fax phone mobile email mail url http https note notes attn "
    "name title id user host ip status error warn warning info debug time type key value tags ref cf ps todo fixme "
    "msg message requester organization org agent category priority summary description location address contact "
    "dept department company total sum rp dr mr ms mrs prof sir madam q a".split()
)


class HandleDetector:
    def __init__(self, stopwords: Iterable[str] = DEFAULT_STOPWORDS, name: str = "handle", score: float = 0.8) -> None:
        self.stopwords = frozenset(w.lower() for w in stopwords)
        self.name = name
        self.score = score

    def _ok(self, label: str) -> bool:
        if label.lower() in self.stopwords or label.endswith("."):
            return False
        if len(label) >= 3 and label.isupper():   # TODO: / ERROR: / NOTE: のような大文字語
            return False
        return True

    def detect(self, text: str) -> list[Entity]:
        found: dict[tuple[int, int], str] = {}
        for m in _AT.finditer(text):
            if self._ok(m.group(1)):
                found[(m.start(1), m.end(1))] = m.group(1)
        labels = list(_LABEL.finditer(text))
        counts: dict[str, int] = {}
        for lab in [m.group("label") for m in labels] + [m.group(1) for m in _AT.finditer(text)]:
            counts[lab] = counts.get(lab, 0) + 1   # 行頭ラベルと @メンションを合わせて数える
        for m in labels:
            lab = m.group("label")
            if not self._ok(lab):
                continue
            if _HANDLE_LIKE.search(lab):
                found[(m.start("label"), m.end("label"))] = lab
                continue
            # 普通の単語のラベルは「会話の話者」に見えるときだけ: 2 回以上出て、YAML/箇条書きのキーでなく
            # （インデントや「- 」で始まらない）、コロンの後が英数字 1 トークンの値ではない（channel: mail / owner: x を除く）
            yaml_like = len(m.group("indent")) >= 2 or (m.group("prefix") or "").startswith(("-", "*", "•"))
            if counts[lab] >= 2 and not yaml_like and not _ASCII_TOKEN.match(m.group("rest")):
                found[(m.start("label"), m.end("label"))] = lab
        for m in _PAREN_LABEL.finditer(text):
            if self._ok(m.group(1)):
                found[(m.start(1), m.end(1))] = m.group(1)
        for m in _KV.finditer(text):
            if self._ok(m.group(1)) and "@" not in m.group(1):   # メールアドレスは RegexDetector が拾う
                found[(m.start(1), m.end(1))] = m.group(1)
        return [Entity(ENTITY_TYPE, s, e, self.score, self.name) for (s, e) in sorted(found)]

"""Surrogate operator: 実在風の日本人名・企業名・住所・施設名・地名への仮名化。

v0.3〜v0.8 の挙動（型が合った偽名、表層特徴の保存、姓リンク）は ``semantic=False`` で維持される。
v0.9（設計 docs/superpowers/specs/2026-09-24-semantic-surrogate-design.md）:

- ``semantic=True``（既定）: 元の実体と同じ粗いカテゴリ（病院・小学校・製造業・同じ地方ブロックの市 …）の架空の名前にする。
  カテゴリは種別語の規則で決め（``CategoryResolver``。分類ヘッドは計画 B）、生成器は ``Features``（閉じた語彙の表層特徴）
  だけを受け取る。候補は ``SurrogateFilter``（元との一致・元の一部の混入・同梱の禁止リスト・実行時の ``knowledge``・
  文書内の既存表層・利用者の禁止語）を通ったものだけ採用し、全部落ちたら型レベルの生成器 → ``<TYPE_xxxxxxxx>`` トークンに退避する。
  **元の値を返す経路は無い**
- ``kind="facility" / "location"``、``kind="auto"``（Entity の型で選ぶ。対象外の型はトークンに退避）
- ``strategy="vault"``: 偽名そのものをトークンとして対応表に保存し、``Engine.restore()`` で元に戻せる（往復）。
  ``scope_id``・``CryptoProvider``・鍵が必要（``Tokenize(vault)`` と同じ）
- 同梱の禁止リストは ``semantic=False`` でも掛かる（v0.8 との差分）

逆算不能について: 偽名が元の値に依存するのは、カテゴリと表層特徴（母集団が数千以上の粗い属性）と、鍵付き HMAC のシードを通してだけ。
「元の値の埋め込みに最も近い候補を選ぶ」方式は同じモデルと候補集合を持つ攻撃者が写像を再現できるため採らない。
hmac+姓リンクの相関特性（同じ鍵・scope で同姓が同じ偽姓に収束する）は v0.4 のまま（緩和策: scope_id を分ける / link_surnames=False / random）。
"""
from __future__ import annotations

import dataclasses
import random
import re
from typing import Callable, Mapping

from ..annotators.knowledge import PublicKnowledge
from ..data.ja_pools import (
    BUILDING_A, BUILDING_B, GIVEN_KANJI, GIVEN_KANJI_F, GIVEN_KANJI_M, GIVEN_KATA, GIVEN_KATA_JA, ORG_CORES, ORG_DIRECTIONS,
    ORG_INDUSTRIES, ORG_KATA_PARTS, ORG_KNOWN_SUFFIXES, PREF_CITIES, SURNAMES_KANJI, SURNAMES_KATA, SURNAMES_KATA_JA, TOWNS,
)
from ..exceptions import ConfigurationError, StoreError
from ..normalize import default_normalize
from ..secrets_types import SecretBytes
from ..types import Entity
from .base import TransformContext
from .semantic import (
    OTHER, CategoryResolver, HAS_BUILDING_RE, address_city_and_town, address_granularity, avoid_parts,
    extract_features, gender_hint, is_municipality, split_person,
)
from .surrogate_filter import SurrogateFilter
from .surrogate_generators import generate

_KINDS = ("person", "org", "facility", "location", "address", "auto")
_LEGACY_KINDS = ("person", "org", "address")
_TYPE_TO_KIND = {"PERSON": "person", "ORG": "org", "FACILITY": "facility", "LOCATION": "location", "ADDRESS": "address"}
_STRATEGIES = ("random", "hmac", "vault")
_ADDRESS_KEEP = (None, "city")
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
# 住所に建物・部屋の表記が含まれるか（偽住所に建物名を付けるかの判定。v0.8 以前は長さ>18 で判定）は
# semantic.HAS_BUILDING_RE と同一だったので、そちらを import して使う（二重定義の解消）
_SAVE_RETRIES = 5
_DRAW_RETRIES = 10
_SEED_LENGTH = 32  # hmacシードに使うbase32文字数（160bit相当）
_ALIAS_PREFIX = "\x00sei:"       # 姓リンクの別名エントリ（制御文字で実テキストと衝突しない）
_SURNAME_SET = frozenset(SURNAMES_KANJI)
# legacy(semantic=False) の person 生成器がカタカナ単独名として返しうるプール語。姓だけ・名だけと同じ辞書語なので
# 公知照合の対象外（60 語のうち 58 語が Wikipedia 見出し語と衝突し、照合すると実文書でほぼ全滅する）
_KATA_BARE_SET = frozenset(GIVEN_KATA + SURNAMES_KATA + SURNAMES_KATA_JA + GIVEN_KATA_JA)

_KATA_RE = re.compile(r"[ァ-ヶー・]")
_KANJI_RE = re.compile(r"[一-龥々]")
_SYS_RNG = random.SystemRandom()


def _katakana_dominant(value: str) -> bool:
    if not value:
        return False
    return len(_KATA_RE.findall(value)) / len(value) > 0.5


def _katakana_person(value: str) -> bool:
    """人名をカタカナプールで扱うか。漢字を1文字でも含む名前（漢字姓+カタカナ名の
    混在名など）は漢字名と同じ扱いにする（v0.5.1、姓リンクも有効になる）。"""
    return _katakana_dominant(value) and not _KANJI_RE.search(value)


def generate_person(value: str, rng: random.Random) -> str:
    v = value.strip()
    if _katakana_person(v):
        if "・" in v:
            return f"{rng.choice(GIVEN_KATA)}・{rng.choice(SURNAMES_KATA)}"
        return rng.choice(GIVEN_KATA + SURNAMES_KATA)
    sep = " " if (" " in v or "　" in v) else ""
    if not sep and len(default_normalize(v)) <= 3:
        return rng.choice(SURNAMES_KANJI)  # 姓のみとみなす
    return rng.choice(SURNAMES_KANJI) + sep + rng.choice(GIVEN_KANJI)


_LEGAL_FORMS = ("株式会社", "合同会社", "有限会社")


def generate_org(value: str, rng: random.Random) -> str:
    v = default_normalize(value)  # NFKCで全角（株）は半角(株)になる
    if _katakana_dominant(v):
        core = "".join(rng.sample(ORG_KATA_PARTS, 2))
    else:
        core = rng.choice(ORG_CORES) + rng.choice(("",) + ORG_INDUSTRIES)
    for form in _LEGAL_FORMS:
        if v.startswith(form):
            return form + core
    if v.startswith("(株)"):
        return "（株）" + core
    for form in _LEGAL_FORMS:
        if v.endswith(form):
            return core + form
    for suffix in ORG_KNOWN_SUFFIXES:
        if v.endswith(suffix):
            # 銀行・信金等は実在名の空間が密なため、方角プレフィックスで明確に架空にする
            return rng.choice(ORG_DIRECTIONS) + rng.choice(ORG_CORES) + suffix
    return core


def generate_address(value: str, rng: random.Random) -> str:
    value = default_normalize(value)  # 表層特徴の判定は他kindと同じく正規化後で行う
    pref, city = rng.choice(PREF_CITIES)
    town = rng.choice(TOWNS)
    c, b, g = rng.randint(1, 5), rng.randint(1, 30), rng.randint(1, 20)
    if "丁目" in value:
        num = f"{c}丁目{b}番{g}号"
    else:
        num = f"{c}-{b}-{g}"
    addr = f"{pref}{city}{town}{num}"
    if HAS_BUILDING_RE.search(value):  # 原文に建物・部屋の表記があるときだけ建物名を付ける（表層特徴の保存）
        addr += rng.choice(BUILDING_A) + rng.choice(BUILDING_B) + f"{rng.randint(2, 12)}0{rng.randint(1, 9)}号室"
    return addr


def _keep_city(value: str, fake: str) -> str:
    """keep="city": 原文の都道府県・市区町村を残し、町名・番地・建物・部屋を偽住所のもので置き換える。

    識別性ベース匿名化（docs/design_identifiability.md §3.5）: 市区町村までは公知で個人を
    識別しないが、番地以下は直接識別子。分解は semantic の住所分解
    （``address_city_and_town`` / ``address_granularity`` / ``is_municipality``）を使う。
    次のどれかに当たれば原文を一切残さず偽住所全体に落とす（fail-closed）:

    - 市区町村が特定できない（建物名だけ・複数行など）
    - 前置部分が本当の市区町村ではない（「本町1-2-3」の「本町」のような町名。分解の正規表現
      ``.{1,8}?[市区町村]`` は町名にも一致するため、実在の町名がそのまま残る欠陥があった）
    - 原文が都道府県・市区町村止まりで、置き換える「以下」が無い
    - 偽住所の側が分解できない（市区町村までしか無い等）
    """
    n = value.strip()
    if address_granularity(n) in ("pref", "city"):
        return fake
    city, _town = address_city_and_town(n)
    if city is None or not is_municipality(city):
        return fake
    if not n[len(city):]:
        return fake
    fake_city, _ = address_city_and_town(fake)
    if not fake_city or len(fake) <= len(fake_city):
        return fake
    return city + fake[len(fake_city):]


_LEGACY_GENERATORS: dict[str, Callable[[str, random.Random], str]] = {
    "person": generate_person,
    "org": generate_org,
    "address": generate_address,
}


def _public_check_applies(kind: str, structure: str | None = None) -> bool:
    """公知照合を掛けるか。姓だけ・名だけ（Wikipedia は姓・名の記事を持つ）と実在地名を使う LOCATION / ADDRESS には掛けない。

    person は候補の文字数ではなく構造で判定する（設計 §11 B：文字数を代理指標にすると、姓2字+名1字のような
    短いフルネームが「姓だけ」免除に紛れ込み、公知照合をすり抜けてしまうため）。``structure`` は
    ``Features.structure``（semantic 経路）と同じ語彙（sei/mono＝姓だけ・名だけ、full＝姓+名）か、
    legacy 経路が渡す ``"sei"``（候補が姓プールそのもの）／``None``（それ以外＝フルネーム相当）のどちらか。
    """
    if kind in ("org", "facility"):
        return True
    if kind == "person":
        return structure not in ("sei", "mono")
    return False


class Surrogate:
    name = "surrogate"

    def __init__(
        self,
        kind: str,
        strategy: str = "hmac",
        key: bytes | None = None,
        key_env: str | None = None,
        generator: Callable[[str, random.Random], str] | None = None,
        normalizer: Callable[[str], str] | None = None,
        link_surnames: bool = True,
        keep: str | None = None,
        *,
        semantic: bool = True,
        knowledge: PublicKnowledge | str | None = None,
        denylist: tuple[str, ...] | list[str] = (),
        min_margin: float | None = None,
        category_head: Callable[[str, bytes], tuple[str, float]] | None = None,
    ) -> None:
        if kind not in _KINDS:
            raise ConfigurationError(f"unknown surrogate kind: {kind!r}")
        if keep not in _ADDRESS_KEEP:
            raise ConfigurationError(f"unknown surrogate keep: {keep!r} (use None or 'city')")
        if keep is not None and kind not in ("address", "auto"):
            raise ConfigurationError("keep is only supported for kind='address' (or 'auto')")
        if strategy not in _STRATEGIES:
            raise ConfigurationError(f"unknown surrogate strategy: {strategy!r}")
        if key is not None and key_env is not None:
            raise ConfigurationError("key and key_env are mutually exclusive")
        if not semantic and kind in ("facility", "location"):
            raise ConfigurationError("kind='facility' / 'location' requires semantic=True (no v0.8 generator exists)")
        if category_head is not None and not callable(category_head):
            raise ConfigurationError("category_head must be callable")
        self.kind = kind
        self.strategy = strategy
        self.key: SecretBytes | None
        if key is not None:
            self.key = SecretBytes(key)
        elif key_env is not None:
            self.key = SecretBytes.from_env(key_env)
        else:
            self.key = None
        self.generator = generator
        self.normalizer = normalizer or default_normalize
        self.link_surnames = link_surnames
        self.keep = keep
        self.semantic = bool(semantic)
        if min_margin is None:
            head_min_margin = getattr(category_head, "min_margin", None)
            resolved_min_margin: float | Mapping[str, float] = 0.15 if head_min_margin is None else head_min_margin
        else:
            resolved_min_margin = min_margin
        self._resolver = CategoryResolver(head=category_head, min_margin=resolved_min_margin)
        self._filter = SurrogateFilter(knowledge=knowledge, denylist=tuple(denylist), normalizer=self.normalizer)

    def forget_document(self) -> None:
        """文書の処理が終わったら SurrogateFilter が持つ全文のメモを捨てる（Engine が文書ごとに呼ぶ）。

        _filter はテストでフィルタのふり（reject_reason だけを持つ簡易実装）に差し替えられることが
        あるため、forget_document を持たない場合は何もしない（engine._forget_documents と同じ流儀）。
        """
        forget = getattr(self._filter, "forget_document", None)
        if callable(forget):
            forget()

    # ------------------------------------------------------------------ 入口
    def _kind_for(self, entity_type: str) -> str | None:
        if self.kind != "auto":
            return self.kind
        return _TYPE_TO_KIND.get(entity_type)

    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str:
        norm = self.normalizer(value)
        if not norm:                                        # 正規化後に空 → fail-closed でトークン
            # extract_features は短い値を「姓のみ」等として扱うため、空文字はそのまま生成器に
            # 渡すと（PERSON の場合）普通の姓に化けてしまい、退避すべき退化値が退避されない
            # （2026-09-28 保留 Minor レビュー: 実測で "" -> 通常の2字姓を返していた不具合）。
            return self._non_target_token(value, norm, entity.entity_type, ctx)
        kind = self._kind_for(entity.entity_type)
        if kind is None:                                   # auto で対象外の型 → fail-closed でトークン
            return self._non_target_token(value, norm, entity.entity_type, ctx)
        fake = self._apply_full(value, norm, entity, ctx, kind)
        if self.keep == "city" and kind == "address":
            return _keep_city(value, fake)
        return fake

    def _apply_full(self, value: str, norm: str, entity: Entity, ctx: TransformContext, kind: str) -> str:
        etype = entity.entity_type
        if kind == "person" and self._linkable(value, norm):
            linked = self._apply_person_linked(value, norm, etype, ctx)
            if linked is not None:
                return linked
        if self.strategy == "hmac":
            return self._draw(value, norm, kind, entity, ctx, self._hmac_rng(ctx, etype, norm))
        existing = ctx.get_token(etype, norm)
        if existing is not None:
            return existing
        last_error: StoreError | None = None
        for _ in range(_SAVE_RETRIES):
            candidate = self._draw(value, norm, kind, entity, ctx, _SYS_RNG)
            try:
                ctx.save_token(etype, norm, candidate, original_value=value if self.strategy == "vault" else None)
                return candidate
            except StoreError as e:
                last_error = e
        raise StoreError("could not generate a unique surrogate") from last_error

    # ------------------------------------------------------------------ 生成
    def _restorable(self, cand: str) -> bool:
        """vault専用ガード: 2字未満の偽名はEngine.restore()の完全一致復元の対象外
        （設計§9、偽姓の復元は2字以上）になり、往復が片道になってしまう。
        strategy="vault"のときだけ候補を2字未満で拒否して再抽選する
        （random/hmacの抽選順序・受理条件はここでは変えない。常にTrueを返す）。"""
        return self.strategy != "vault" or len(cand) >= 2

    def _draw(self, value: str, norm: str, kind: str, entity: Entity, ctx: TransformContext, rng: random.Random) -> str:
        text = getattr(ctx, "text", None)
        if self.generator is not None or (not self.semantic and kind in _LEGACY_KINDS):
            gen = self.generator or _LEGACY_GENERATORS[kind]
            # distinctness（原文と同一）以外の理由で1回でも却下されたか。生成器が原文と
            # 区別できる値を一度も作れなかった（設定ミス）だけなら vault でも握りつぶさず送出する
            rejected_beyond_distinctness = False
            for _ in range(_DRAW_RETRIES):
                candidate = gen(value, rng)
                if not candidate or self.normalizer(candidate) == norm:
                    continue
                if not self._restorable(candidate):
                    rejected_beyond_distinctness = True
                    continue
                # 同梱の禁止リスト・利用者の denylist は semantic=False でも掛ける（設計 §10）。
                # 自作 generator は利用者の責任で対象外。public_entity 以外の理由（denylist 等）も再抽選する。
                # legacy の person 生成器は、姓のみのとき候補が姓プールそのもの、カタカナ単独名のときカタカナ単独
                # プールそのもの（generate_person 参照）になる。どちらも辞書語なので公知照合は掛けない。
                # それ以外（フルネーム）はフルネーム相当として公知照合を掛ける
                legacy_structure = ("sei" if kind == "person" and (candidate in _SURNAME_SET or candidate in _KATA_BARE_SET)
                                    else None)
                if self.generator is None and self._filter.reject_reason(
                    candidate, original_norm=norm, check_public=_public_check_applies(kind, legacy_structure),
                ) is not None:
                    rejected_beyond_distinctness = True
                    continue
                return candidate
            if rejected_beyond_distinctness:
                # 2字未満・公知照合・denylist 等が原因で尽きたときは、戦略に関わらず元の値には絶対に落とさず
                # トークンへ退避する（設計 §7「フィルタ全滅 → トークン」。semantic 経路と同じ）
                return self._fallback_token(entity.entity_type, norm, ctx)
            raise ConfigurationError("surrogate generator failed to produce a value distinct from the input")
        if not self.semantic and kind in ("facility", "location"):
            # semantic=False 用の v0.8 generator が無い型。auto がここに解決したときは元の値を
            # 絶対に返さずトークンへ退避する（コンストラクタのエラーメッセージと整合）
            return self._fallback_token(entity.entity_type, norm, ctx)
        category = self._resolver.resolve(kind, norm, entity.vector)
        feats = extract_features(kind, value, norm, category, keep=self.keep if kind == "address" else None)
        avoid = avoid_parts(kind, norm)
        # カテゴリ別 → 型レベルに退避。既に型レベル（OTHER）なら 2 周目は 1 周目と同じ特徴の引き直しにすぎないので
        # 1 周にまとめるが、試行回数は 2 周分を保つ（同じ特徴・同じ rng から同じ順に引くので、hmac で従来 11〜20 回目に
        # 通っていた値もそのまま通る。1 周 10 回に減らすとそれらがトークンに落ちる。最終レビュー M2）
        if feats.category == OTHER[kind]:
            rounds = ((feats, 2 * _DRAW_RETRIES),)
        else:
            rounds = ((feats, _DRAW_RETRIES), (dataclasses.replace(feats, category=OTHER[kind]), _DRAW_RETRIES))
        for f, budget in rounds:
            for _ in range(budget):
                candidate = generate(f, rng)
                if candidate and self._restorable(candidate) and self._filter.reject_reason(
                    candidate, original_norm=norm, avoid_parts=avoid, text=text,
                    check_public=_public_check_applies(kind, f.structure),
                ) is None:
                    return candidate
        return self._fallback_token(entity.entity_type, norm, ctx)

    def _non_target_token(self, value: str, norm: str, etype: str, ctx: TransformContext) -> str:
        """auto で対象外の型（EMAIL・PHONE 等）の退避トークン。hmac は決定的なので無状態のまま返す。
        random / vault は store に保存して同一 scope の同じ値を同じトークンにし、vault では原文も保存して
        ``Engine.restore()`` で戻せるようにする（保存しないと往復が無言で片道になる）。"""
        if self.strategy == "hmac":
            return self._fallback_token(etype, norm, ctx)
        existing = ctx.get_token(etype, norm)
        if existing is not None:
            return existing
        last_error: StoreError | None = None
        for attempt in range(_SAVE_RETRIES):
            token = self._fallback_token(etype, norm, ctx) if attempt == 0 else \
                f"<{etype}_{''.join(_SYS_RNG.choice(_B32) for _ in range(8))}>"
            try:
                ctx.save_token(etype, norm, token, original_value=value if self.strategy == "vault" else None)
                return token
            except StoreError as e:                   # 同一 scope で別実体に同じトークン → 乱数で引き直す
                last_error = e
        raise StoreError("could not generate a unique surrogate token") from last_error

    def _fallback_token(self, etype: str, norm: str, ctx: TransformContext) -> str:
        """偽名を作れなかったときの退避先。元の値は絶対に返さない。鍵があれば決定的、無ければ乱数。"""
        try:
            tid = ctx.hmac_token(etype, norm, key=self.key.reveal() if self.key is not None else None, length=8)
        except ConfigurationError:
            if self.strategy != "random":    # hmac / vault は鍵が要る。握りつぶすと決定性が黙って失われる
                raise
            tid = "".join(_SYS_RNG.choice(_B32) for _ in range(8))
        return f"<{etype}_{tid}>"

    # ------------------------------------------------------------------ 姓リンク（漢字名）
    def _linkable(self, value: str, norm: str) -> bool:
        """漢字の姓リンク経路（``_apply_person_linked``。偽姓は SURNAMES_KANJI から引く）に入れる値か。

        Important I4: 判定が「カタカナ優勢でなければ通す」だったため、平仮名・ローマ字・半角カナの
        裸姓（正規化後 3 字以下）まで漢字の偽姓になり、表層の文字種が保存されていなかった
        （たなか → 武田）。semantic=True では漢字を含む値だけを通し、それ以外は semantic の
        生成器（script に応じたプール）に回す。semantic=False の出力は変えないため、
        この絞り込みは ``self.semantic`` のときだけ掛ける。
        """
        if not self.link_surnames or self.generator is not None:
            return False
        if _katakana_person(value.strip()):
            return False
        if self.semantic:
            compact = norm.replace(" ", "")
            return bool(_KANJI_RE.search(compact)) or split_person(compact) is not None
        return True

    def _hmac_rng(self, ctx: TransformContext, etype: str, key_norm: str) -> random.Random:
        seed = ctx.hmac_token(
            etype, key_norm,
            key=self.key.reveal() if self.key is not None else None,
            length=_SEED_LENGTH,
        )
        return random.Random(seed)

    def _given_pool(self, norm: str) -> tuple[str, ...]:
        if not self.semantic:
            return GIVEN_KANJI
        split = split_person(norm.replace(" ", ""))
        # split_person は姓プールの前方一致で切るだけなので、「山田・太郎」のように中黒区切りだと
        # 名の側に先頭の「・」が残る（"・太郎"）。gender_hint に渡す前に落とす
        given = split[1].strip("・") if split else None
        hint = gender_hint(given) if given else "unknown"
        return GIVEN_KANJI_M if hint == "m" else GIVEN_KANJI_F if hint == "f" else GIVEN_KANJI

    def _ok_linked(self, cand: str, norm: str, ctx: TransformContext) -> bool:
        if self.normalizer(cand) == norm:
            return False
        if not self.semantic:
            # Important #3 と同じ理由: public_entity 以外（denylist・same_as_original 等）も再抽選する
            return self._filter.reject_reason(cand, original_norm=norm) is None
        split = split_person(norm.replace(" ", ""))
        # 同上: 「・太郎」のまま avoid に入れると、区切りを持たずに連結される候補（_draw_linked の
        # sep は半角/全角スペースの有無でしか判定しない）に絶対に一致せず、名の再利用を防げない
        # （最終レビュー残件1 と同じクラスの不具合。姓リンク経路は avoid_parts() を経由しないため
        # 別途ここでも「・」を落とす）
        given = split[1].strip("・") if split else ""
        avoid = (given,) if split and len(given) >= 2 else ()
        return self._filter.reject_reason(cand, original_norm=norm, avoid_parts=avoid,
                                          text=getattr(ctx, "text", None)) is None

    def _sei_ok(self, cand: str, avoid: str, text: str | None) -> bool:
        if not self._restorable(cand):
            return False
        return self._filter.reject_reason(
            cand, original_norm=avoid,
            avoid_parts=(avoid,) if len(avoid) >= 2 else (),
            text=text, check_public=False,
        ) is None

    def _draw_linked(self, fake_sei: str, sep: str, pool: tuple[str, ...], norm: str,
                     ctx: TransformContext, rng: random.Random) -> str | None:
        """姓リンクのフルネーム候補「偽姓 + 区切り + 名」を 1 つ選ぶ。見つからなければ None。

        Important I5 の残り: ランダム 10 回で諦めていたため、名の候補がフィルタ（公知照合・
        denylist・文書内表層）に落ち続けるだけで姓リンクを捨ててトークンへ退避していた。
        ``_draw_sei`` と同じく、ランダム試行の**後**にプール全体を ``rng.sample`` の順で走査する
        （走査は既存の抽選の後ろなので、従来成功していた場合の抽選順序＝hmac のゴールデン値は
        変わらない）。``_restorable`` は「偽姓 + 名」が常に 2 字以上になるため実質常に真だが、
        vault の往復条件を明示するために合わせて掛ける。
        """
        def ok(cand: str) -> bool:
            return self._restorable(cand) and self._ok_linked(cand, norm, ctx)

        for _ in range(_DRAW_RETRIES):
            cand = fake_sei + sep + rng.choice(pool)
            if ok(cand):
                return cand
        for given in rng.sample(pool, len(pool)):
            cand = fake_sei + sep + given
            if ok(cand):
                return cand
        return None

    def _draw_sei(self, rng: random.Random, avoid: str, text: str | None = None) -> str:
        """偽姓を1つ引く。姓単独は Wikipedia が姓の記事を持つため公知照合（check_public）は掛けない
        （掛けると常用姓のほとんどが弾かれてしまう）が、denylist と文書内表層はここでも掛ける。
        vaultではSURNAMES_KANJIの1字姓（林・森・堀）も拒否して再抽選する（呼び出し元が
        枯渇時に型レベルのトークンへ退避する。random/hmacはこのガードの影響を受けない）。

        ランダム試行で見つからなかったときは、プール全体を ``rng.sample`` の順で走査して
        最初に通る候補を返す（Important I5: 文書に常用姓が大量に出ていて present_in_document で
        弾かれ続けるだけで、まだ使える姓が残っているのに諦めていた）。走査は既存の抽選の**後**に
        置くので、従来成功していた場合の抽選順序＝hmac のゴールデン値は変わらない。
        本当にプールが尽きたときだけ ConfigurationError（呼び出し元が通常経路へ落とす）。"""
        for _ in range(_DRAW_RETRIES):
            cand = rng.choice(SURNAMES_KANJI)
            if self._sei_ok(cand, avoid, text):
                return cand
        for cand in rng.sample(SURNAMES_KANJI, len(SURNAMES_KANJI)):
            if self._sei_ok(cand, avoid, text):
                return cand
        raise ConfigurationError("surrogate surname pool exhausted")

    def _apply_person_linked(self, value: str, norm: str, etype: str, ctx: TransformContext) -> str | None:
        """姓リンク経路。対象外の形（分割不能など）や候補が尽きたときは None を返し通常経路へ。"""
        v = value.strip()
        text = getattr(ctx, "text", None)
        sep = " " if (" " in v or "　" in v) else ""
        bare = not sep and len(norm) <= 3
        vault_orig = self.strategy == "vault"
        if bare:
            if self.strategy == "hmac":
                try:
                    return self._draw_sei(self._hmac_rng(ctx, etype, norm), avoid=norm, text=text)
                except ConfigurationError:
                    return None              # 枯渇 → 通常経路（_draw → フィルタ → トークン）
            alias = ctx.get_token(etype, _ALIAS_PREFIX + norm)
            if alias is not None:
                return alias
            existing = ctx.get_token(etype, norm)
            if existing is not None:
                return existing
            for _ in range(_SAVE_RETRIES):
                try:
                    cand = self._draw_sei(_SYS_RNG, avoid=norm, text=text)
                except ConfigurationError:
                    # 候補が尽きた（denylist で全滅・vault で2字以上が残らない等）。姓リンク
                    # 経路をあきらめて通常経路（_apply_full の _draw + 保存）へフォールスルー
                    # する。vault の保存はそちらに任せる（ここで自前にフォールバックトークンを
                    # 作ると _apply_full の save_token を経由せず、保存されない片道になる）。
                    # Important I5: 設計は「フィルタ全滅 → トークン」なので、hmac / random でも
                    # 例外を上げずに通常経路へ落とす（transform() はここで止まってはいけない）
                    return None
                try:
                    ctx.save_token(etype, norm, cand, original_value=value if vault_orig else None)
                    return cand
                except StoreError:
                    continue
            raise StoreError("could not generate a unique surrogate")
        split = split_person(norm)
        if split is None:
            return None                      # 姓プール外 → 通常経路（リンクなし）
        sei_o, _ = split
        sei_norm = self.normalizer(sei_o)
        pool = self._given_pool(norm)
        if self.strategy == "hmac":
            try:
                fake_sei = self._draw_sei(self._hmac_rng(ctx, etype, sei_norm), avoid=sei_norm, text=text)
            except ConfigurationError:
                return None                  # 枯渇 → 通常経路（_draw → フィルタ → トークン）
            rng = self._hmac_rng(ctx, etype, norm)
            # 候補が尽きたら None → 通常経路（フィルタ付き）
            return self._draw_linked(fake_sei, sep, pool, norm, ctx, rng)
        existing = ctx.get_token(etype, norm)
        if existing is not None:
            return existing
        adopted = (ctx.get_token(etype, _ALIAS_PREFIX + sei_norm)
                   or ctx.get_token(etype, sei_norm))   # 後方リンク: 裸姓が先に独立割当済みの場合
        if adopted is not None and adopted.startswith("<"):
            # Important I2: 裸姓が偽名を作れず <TYPE_xxxxxxxx> トークンに退避されていた場合。
            # これを偽姓として採ると「<PERSON_xxxxxxxx>名」という偽名ができ、restore() の
            # 型トークン書式と衝突して往復が壊れる（トークン部分だけが先に一致して切られる）。
            # 姓リンクはあきらめ、新しい偽姓を引く（引けなければ通常経路へ）
            adopted = None
        if adopted:
            fake_sei = adopted
        else:
            try:
                fake_sei = self._draw_sei(_SYS_RNG, avoid=sei_norm, text=text)
            except ConfigurationError:
                # 偽姓が尽きた。姓リンクをあきらめて通常経路（_draw + 保存）へフォールスルーする
                # （保存はそちらに任せる。ここで作ると保存されず片道になる）。Important I5:
                # vault だけでなく random でも例外を上げずに通常経路へ落とす
                return None
        last_error: StoreError | None = None
        cand = None
        for _ in range(_SAVE_RETRIES):
            # フィルタ通過の探索（_DRAW_RETRIES）と保存衝突の再試行（_SAVE_RETRIES）は別予算にする。
            # 同じ予算を共有すると（v0.9 で _ok_linked が同梱禁止リスト等のフィルタを掛けるようになったため）
            # フィルタ却下と保存衝突が重なったときに再試行回数が尽きやすい
            found = self._draw_linked(fake_sei, sep, pool, norm, ctx, _SYS_RNG)
            if found is None:
                return None               # フィルタ候補が尽きた → 通常経路（フィルタ付き）
            cand = found
            try:
                ctx.save_token(etype, norm, cand, original_value=value if vault_orig else None)
                break
            except StoreError as e:
                last_error = e
                cand = None
        if cand is None:
            if last_error is not None:
                raise StoreError("could not generate a unique surrogate") from last_error
            return None
        try:
            ctx.save_token(etype, _ALIAS_PREFIX + sei_norm, fake_sei, original_value=sei_o if vault_orig else None)
        except StoreError:
            pass                              # 別名の重複は無害（既存を優先）
        return cand

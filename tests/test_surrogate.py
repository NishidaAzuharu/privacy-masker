import pytest

from privacy_masker.data.ja_pools import (
    GIVEN_KATA, ORG_CORES, PREF_CITIES, SURNAMES_KANJI, SURNAMES_KATA,
)
from privacy_masker.encoding import hmac_token_id
from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError, StoreError
from privacy_masker.normalize import default_normalize
from privacy_masker.operators import Surrogate
from privacy_masker.policy import Policy
from privacy_masker.types import Entity

from .conftest import FakeCrypto, FakeDetector

E = Entity(entity_type="PERSON", start=0, end=4, score=1.0)
KEY = b"surrogate-test-key"


class FakeContext:
    def __init__(self, scope_id="s1", fail_saves=0):
        self.scope_id = scope_id
        self.tokens = {}
        self._fail_saves = fail_saves

    def next_seq(self, entity_type):
        raise AssertionError("surrogate must not use next_seq")

    def get_token(self, entity_type, normalized_value):
        return self.tokens.get((entity_type, normalized_value))

    def save_token(self, entity_type, normalized_value, token, original_value=None):
        if self._fail_saves > 0:
            self._fail_saves -= 1
            raise StoreError("token already exists in scope")
        self.tokens[(entity_type, normalized_value)] = token

    def hmac_token(self, entity_type, normalized_value, key=None, length=8):
        return hmac_token_id(key or KEY, self.scope_id or "", entity_type,
                             normalized_value, length)


# ---- 決定性・一貫性 ----

def test_hmac_same_value_same_surrogate_across_contexts():
    op = Surrogate(kind="person", strategy="hmac")
    s1 = op.apply("山田太郎", E, FakeContext())
    s2 = op.apply("山田太郎", E, FakeContext())
    assert s1 == s2


def test_hmac_normalized_variants_share_identity():
    # 同一人物（正規化同値）は同じ偽名になる。表記（空白）は各出現の原文に追従するため
    # 文字列は完全一致でなく正規化後一致を仕様とする
    op = Surrogate(kind="person", strategy="hmac")
    s1 = op.apply("山田太郎", E, FakeContext())
    s2 = op.apply("山田　太郎", E, FakeContext())
    assert default_normalize(s1) == default_normalize(s2)


def test_hmac_different_values_differ():
    op = Surrogate(kind="person", strategy="hmac")
    ctx = FakeContext()
    assert op.apply("山田太郎", E, ctx) != op.apply("佐藤花子", E, ctx)


def test_random_consistent_within_context():
    op = Surrogate(kind="person", strategy="random")
    ctx = FakeContext()
    assert op.apply("山田太郎", E, ctx) == op.apply("山田太郎", E, ctx)


def test_random_collision_retries_then_succeeds():
    op = Surrogate(kind="person", strategy="random")
    ctx = FakeContext(fail_saves=2)
    out = op.apply("山田太郎", E, ctx)
    assert out and out != "山田太郎"


def test_never_returns_original():
    op = Surrogate(kind="person", strategy="hmac")
    for name in ["山田太郎", "佐藤", "田中健一", "ミゲル・サントス"]:
        assert op.apply(name, E, FakeContext()) != name


# ---- 文字種・形式の保存 ----

def test_person_full_name_uses_pools():
    out = Surrogate(kind="person", strategy="hmac").apply("山田太郎", E, FakeContext())
    assert any(out.startswith(s) for s in SURNAMES_KANJI)
    assert len(out) >= 4


def test_person_short_input_gives_surname_only():
    out = Surrogate(kind="person", strategy="hmac", semantic=False).apply("佐藤", E, FakeContext())
    assert out in SURNAMES_KANJI


def test_person_katakana_preserved():
    out = Surrogate(kind="person", strategy="hmac").apply("ミゲル・サントス", E, FakeContext())
    assert "・" in out
    given, sur = out.split("・")
    assert given in GIVEN_KATA and sur in SURNAMES_KATA


def test_person_space_format_preserved():
    out = Surrogate(kind="person", strategy="hmac").apply("山田 太郎", E, FakeContext())
    assert " " in out


def test_org_keeps_corporate_prefix():
    out = Surrogate(kind="org", strategy="hmac").apply("株式会社大和技研", E, FakeContext())
    assert out.startswith("株式会社") and out != "株式会社大和技研"


def test_org_keeps_corporate_suffix():
    out = Surrogate(kind="org", strategy="hmac", semantic=False).apply("大和技研株式会社", E, FakeContext())
    assert out.endswith("株式会社") and out != "大和技研株式会社"


def test_org_keeps_known_suffix():
    out = Surrogate(kind="org", strategy="hmac", semantic=False).apply("西日本あおば信用金庫", E, FakeContext())
    assert out.endswith("信用金庫")


def test_address_realistic_pref_city():
    out = Surrogate(kind="address", strategy="hmac").apply(
        "東京都新宿区西新宿1-2-3", E, FakeContext())
    assert any(out.startswith(p + c) for p, c in PREF_CITIES)
    assert out != "東京都新宿区西新宿1-2-3"


def test_address_mirrors_chome_style():
    out = Surrogate(kind="address", strategy="hmac").apply(
        "大阪府堺市北区中百舌鳥町2丁目39番1号", E, FakeContext())
    assert "丁目" in out


def test_custom_generator_overrides():
    op = Surrogate(kind="person", strategy="hmac",
                   generator=lambda value, rng: "固定偽名")
    assert op.apply("山田太郎", E, FakeContext()) == "固定偽名"


# ---- 構成検証 ----

def test_vault_strategy_allowed_but_needs_crypto_in_engine():
    # Policy.has_strategy が Surrogate(strategy="vault") も見るようになった（Task 10）ため、
    # crypto無しの構成はengine構築時点で一括拒否される（Tokenize(vault)と同じ経路、engine.py:_validate_construction）
    op = Surrogate(kind="person", strategy="vault")
    assert op.strategy == "vault"  # 構成時点では拒否されない（v0.8 は vault を拒否していた）
    with pytest.raises(ConfigurationError):
        PrivacyEngine(
            detectors=[FakeDetector([])],
            policy=Policy(rules={"PERSON": op}),
            secret_key=b"k" * 16,
        )


def test_vault_strategy_hits_scope_id_guard_without_scope_id():
    # has_strategy の一般化により、_check_call の scope_id ガードが Surrogate(vault) にも掛かるようになった
    # （engine.py:_check_call）。scope_id 無しの transform() は DocumentContext まで進む前にここで拒否される
    op = Surrogate(kind="person", strategy="vault")
    engine = PrivacyEngine(
        detectors=[FakeDetector([Entity("PERSON", 0, 4, 1.0)])],
        policy=Policy(rules={"PERSON": op}),
        secret_key=b"k" * 16,
        crypto=FakeCrypto(),
    )
    with pytest.raises(ConfigurationError):
        engine.transform("山田太郎")  # scope_id 無し → _check_call の scope_id ガード


def test_unknown_kind_rejected():
    with pytest.raises(ConfigurationError):
        Surrogate(kind="phone")


def test_key_and_key_env_exclusive():
    with pytest.raises(ConfigurationError):
        Surrogate(kind="person", key=b"k", key_env="X")


# ---- エンジン統合 ----

def test_engine_hmac_surrogate_requires_key():
    policy = Policy(rules={"PERSON": Surrogate(kind="person", strategy="hmac")})
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], policy=policy)  # 鍵なし


def test_engine_transform_produces_natural_text():
    text = "山田太郎さんは株式会社大和技研に勤務。"
    ents = [Entity("PERSON", 0, 4, 1.0), Entity("ORG", 7, 15, 1.0)]
    policy = Policy(rules={
        "PERSON": Surrogate(kind="person", strategy="hmac"),
        "ORG": Surrogate(kind="org", strategy="hmac"),
    })
    engine = PrivacyEngine(detectors=[FakeDetector(ents)], policy=policy,
                           secret_key=b"k" * 16)
    result = engine.transform(text)
    assert "山田太郎" not in result.text and "大和技研" not in result.text
    assert "<" not in result.text                      # トークン書式が混ざらない
    assert result.text.endswith("に勤務。")


def test_document_context_rejects_duplicate_token():
    from privacy_masker.scopes import DocumentContext
    ctx = DocumentContext(None)
    ctx.save_token("PERSON", "a", "佐藤健一")
    with pytest.raises(StoreError):
        ctx.save_token("PERSON", "b", "佐藤健一")


# ---- レビュー指摘対応 ----

# 実在の著名企業・金融機関（完全一致で生成してはならない）
_FAMOUS_ORGS = {
    "大和証券", "大和銀行", "三和銀行", "瑞穂銀行", "みずほ銀行", "東洋銀行",
    "太陽銀行", "千代田銀行", "旭銀行", "さくら銀行", "富士銀行", "三井銀行",
    "住友銀行", "北都銀行", "野村證券", "野村証券", "日興証券", "山一證券",
    "太陽生命", "千代田生命", "第一生命", "日本生命", "住友生命", "明治生命",
    "共栄火災", "大同生命", "朝日生命", "東京海上", "三井住友海上",
    "旭化成", "曙ブレーキ工業", "東洋紡", "東洋製罐", "双葉社", "高砂熱学工業",
    "高砂香料工業", "常磐興産", "栄光ゼミナール", "三和シヤッター工業",
    "京浜急行電鉄", "千代田化工建設", "大和ハウス工業", "東洋水産", "太陽誘電",
    "大和総研", "三和化学研究所",
}


def test_generated_orgs_never_match_famous_companies():
    """生成可能な全組織名を列挙し、著名企業名と完全一致しないことを保証する"""
    from privacy_masker.data.ja_pools import (
        ORG_CORES, ORG_DIRECTIONS, ORG_INDUSTRIES, ORG_KNOWN_SUFFIXES,
    )
    generated = set()
    for core in ORG_CORES:
        generated.add(core)
        generated.add("株式会社" + core)
        generated.add(core + "株式会社")
        for ind in ORG_INDUSTRIES:
            generated.add(core + ind)
            generated.add("株式会社" + core + ind)
            generated.add(core + ind + "株式会社")
        for d in ORG_DIRECTIONS:
            for suf in ORG_KNOWN_SUFFIXES:
                generated.add(d + core + suf)
    hits = generated & _FAMOUS_ORGS
    assert not hits, f"実在企業名と一致: {sorted(hits)}"


def test_known_suffix_org_uses_direction_prefix():
    # 金融機関等の種別語保存では方角プレフィックス付きの明確に架空な名前を使う
    from privacy_masker.data.ja_pools import ORG_DIRECTIONS
    out = Surrogate(kind="org", strategy="hmac", semantic=False).apply("西日本あおば信用金庫", E, FakeContext())
    assert out.endswith("信用金庫")
    assert out[0] in "".join(ORG_DIRECTIONS)


def test_org_keeps_godo_kaisha():
    out = Surrogate(kind="org", strategy="hmac", semantic=False).apply("合同会社テトラワークス", E, FakeContext())
    assert out.startswith("合同会社") and out != "合同会社テトラワークス"


def test_custom_normalizer_used_for_distinctness_check():
    # カスタムnormalizer基準で原文と等価な生成値は弾かれる（レビューImportant 2）
    def upper_gen(value, rng):
        return value.upper()

    op = Surrogate(kind="person", strategy="hmac", semantic=False,
                   generator=upper_gen,
                   normalizer=lambda v: v.replace(" ", "").lower())
    with pytest.raises(ConfigurationError):
        op.apply("sato ichiro", E, FakeContext())


def test_hmac_golden_values_frozen():
    """ゴールデン値回帰テスト（レビューImportant 3）。

    ja_pools.pyのタプル内容・順序の変更、およびPythonのrandomシード仕様変更は
    hmac戦略の既存出力をすべて変える破壊的変更である。このテストが落ちた場合、
    意図的な破壊的変更（メジャーバージョンアップ）でない限り変更を巻き戻すこと。

    v0.9: semantic=False（v0.8の挙動）のゴールデン値。同梱禁止リストとの衝突で
    再抽選が入った値はv0.9で更新（CHANGELOG参照）。
    """
    cases = [
        # v0.4: 姓リンク導入で人名の生成規則が変更（偽姓=姓部分のHMAC決定値）
        ("person", "山田太郎", "杉本智子"),
        ("person", "佐藤", "矢野"),
        ("person", "ミゲル・サントス", "ダニエル・スミス"),
        ("org", "株式会社大和技研", "株式会社富士見薬品"),
        ("org", "西日本あおば信用金庫", "新丸昌信用金庫"),
        ("address", "東京都新宿区西新宿1-2-3", "滋賀県大津市大手町4-25-6"),
        ("person", "山田", "杉本"),   # 姓リンク: 山田太郎(→杉本智子)と同じ偽姓になる
    ]
    for kind, value, expected in cases:
        got = Surrogate(kind=kind, strategy="hmac", semantic=False).apply(value, E, FakeContext())
        assert got == expected, f"{kind}/{value}: {got!r} != {expected!r}"


# ---- 姓リンク（v0.4）: フルネームと姓のみ言及の同一人物連携 ----

class LinkCtx(FakeContext):
    """トークン一意性も検査するcontext（DocumentContext相当の挙動）"""

    def save_token(self, entity_type, normalized_value, token, original_value=None):
        if self._fail_saves > 0:
            self._fail_saves -= 1
            raise StoreError("token already exists in scope")
        if token in self.tokens.values():
            raise StoreError("token already exists in scope")
        self.tokens[(entity_type, normalized_value)] = token


def test_link_forward_fullname_then_bare():
    # 山田太郎→偽名を割当後、姓のみ「山田」は偽名の姓にリンクされる
    #
    # len(bare) >= 2 を要求していたが、SURNAMES_KANJI には 1 文字の姓が
    # 170 件中 3 件（林・森・堀）あるため約 2% の確率で落ちていた（実測 3000 回で 61 回）。
    # 落ちていたのは assert の前提で、リンク自体は全ケースで正しく動いていた
    # （失敗例でも full='堀結衣' bare='堀' で startswith は成立）。
    op = Surrogate(kind="person", strategy="random")
    ctx = LinkCtx()
    full = op.apply("山田太郎", E, ctx)
    bare = op.apply("山田", E, ctx)
    assert bare and full.startswith(bare) and len(bare) < len(full)


def test_link_reverse_bare_then_fullname():
    # 姓のみ「山田」が先（文末署名が先に処理されるケース）でも、
    # 後から来たフルネームが既存の偽姓を引き継ぐ
    op = Surrogate(kind="person", strategy="random")
    ctx = LinkCtx()
    bare = op.apply("山田", E, ctx)
    full = op.apply("山田太郎", E, ctx)
    assert full.startswith(bare)


def test_link_works_with_hmac_strategy():
    op = Surrogate(kind="person", strategy="hmac")
    ctx = LinkCtx()
    full = op.apply("山田太郎", E, ctx)
    bare = op.apply("山田", E, ctx)
    assert full.startswith(bare)


def test_link_ambiguous_surname_falls_back_to_independent():
    # 同一scopeに山田太郎と山田花子 → 裸の「山田」はどちらにもリンクしない
    op = Surrogate(kind="person", strategy="hmac")
    ctx = LinkCtx()
    p1 = op.apply("山田太郎", E, ctx)
    p2 = op.apply("山田花子", E, ctx)
    bare = op.apply("山田", E, ctx)
    # 独立割当（リンク無効時のhmac決定値）と一致することで「リンクしていない」ことを検証
    independent = Surrogate(kind="person", strategy="hmac",
                            link_surnames=False).apply("山田", E, LinkCtx())
    assert bare == independent


def test_link_disabled_by_flag():
    op = Surrogate(kind="person", strategy="hmac", link_surnames=False)
    ctx = LinkCtx()
    op.apply("山田太郎", E, ctx)
    bare = op.apply("山田", E, ctx)
    independent = Surrogate(kind="person", strategy="hmac",
                            link_surnames=False).apply("山田", E, LinkCtx())
    assert bare == independent


def test_link_random_shared_sei_forward():
    # random戦略: 同一原姓の2人が偽姓を共有し（alias採用）、裸姓もそれにリンクする
    op = Surrogate(kind="person", strategy="random")
    ctx = LinkCtx()
    p1 = op.apply("山田太郎", E, ctx)
    p2 = op.apply("山田花子", E, ctx)
    bare = op.apply("山田", E, ctx)
    assert p1[:2] == p2[:2] == bare[:2] or p1.startswith(bare)


def test_link_same_person_consistency_preserved():
    # リンク導入後も、同値の一貫性（山田太郎2回→同じ偽名）は維持される
    op = Surrogate(kind="person", strategy="hmac")
    ctx = LinkCtx()
    assert op.apply("山田太郎", E, ctx) == op.apply("山田太郎", E, ctx)


def test_link_engine_end_to_end_descending_order():
    # 1文書内で本文にフルネーム・署名に姓のみ。エンジンは文末側から処理するため
    # 姓のみが先に割当てられるが、後方リンクでフルネームが偽姓を引き継ぐ
    text = "山田太郎です。よろしくお願いします。担当:山田"
    ents = [Entity("PERSON", 0, 4, 1.0), Entity("PERSON", 21, 23, 1.0)]
    engine = PrivacyEngine(
        detectors=[FakeDetector(ents)],
        policy=Policy(rules={"PERSON": Surrogate(kind="person", strategy="random")}),
        secret_key=b"k" * 16)
    r = engine.transform(text)
    full_rep = next(e.replacement for e in r.entities if e.original_start == 0)
    bare_rep = next(e.replacement for e in r.entities if e.original_start == 21)
    assert full_rep.startswith(bare_rep)


_KATA_CHARS = set("ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノ"
                  "ハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー・")


def test_mixed_kanji_katakana_treated_as_kanji_name():
    # 漢字姓+カタカナ名の混在名は漢字名と同じ扱い（カタカナプールに落ちない）
    op = Surrogate(kind="person", key=KEY)
    out = op.apply("高橋マリアエレナ", E, FakeContext())
    assert not any(c in _KATA_CHARS for c in out), out
    assert out != "高橋マリアエレナ"


def test_mixed_name_links_surname_with_bare_mention():
    # 混在名でも姓リンクが効く: 高橋マリアエレナ と裸の「高橋」が同じ偽姓になる
    op = Surrogate(kind="person", key=KEY)
    ctx = FakeContext()
    full = op.apply("高橋マリアエレナ", E, ctx)
    bare = op.apply("高橋", E, ctx)
    assert full.startswith(bare), (full, bare)


def test_katakana_only_name_still_uses_katakana_pool():
    # 漢字を含まないカタカナ名は従来どおりカタカナプール（ゴールデン値でも固定済み）
    op = Surrogate(kind="person", key=KEY)
    out = op.apply("ミゲル・サントス", E, FakeContext())
    assert any(c in _KATA_CHARS for c in out), out

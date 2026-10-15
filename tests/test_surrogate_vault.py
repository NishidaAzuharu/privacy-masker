"""可逆 Surrogate（strategy="vault"）の往復（設計 §9）。外部 LLM が敬称を変え・姓だけで呼び・文を足した応答を戻す。"""
import pytest

from privacy_masker.data.ja_pools import SURNAMES_KANJI
from privacy_masker.encoding import encode_fields
from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Keep, Surrogate, Tokenize
from privacy_masker.policy import Policy
from privacy_masker.stores import MemoryStore, SQLiteStore
from privacy_masker.types import Entity

from .conftest import FakeCrypto, FakeDetector

TEXT = "山田太郎さんは株式会社青葉電機に勤務している。山田さんの上司は佐藤花子部長です。"


def _ent(t, surface, start=None):
    s = TEXT.index(surface) if start is None else start
    return Entity(t, s, s + len(surface), 0.95)


ENTS = [_ent("PERSON", "山田太郎"), _ent("ORG", "株式会社青葉電機"),
        _ent("PERSON", "山田", start=TEXT.index("山田さん")), _ent("PERSON", "佐藤花子")]


def engine(**kw):
    kw.setdefault("policy", Policy(rules={
        "PERSON": Surrogate(kind="person", strategy="vault"),
        "ORG": Surrogate(kind="org", strategy="vault")}, default=Keep()))
    kw.setdefault("crypto", FakeCrypto())
    kw.setdefault("secret_key", b"k" * 16)
    kw.setdefault("detectors", [FakeDetector(ENTS)])
    kw.setdefault("refiner", None)
    return PrivacyEngine(**kw)


def test_roundtrip_exact():
    eng = engine()
    r = eng.transform(TEXT, scope_id="case_1")
    assert "山田太郎" not in r.text and "青葉電機" not in r.text and "佐藤花子" not in r.text
    back = eng.restore(r.text, scope_id="case_1")
    assert back.text == TEXT and back.unresolved == () and back.restored == 4


def test_roundtrip_llm_style_response():
    eng = engine()
    r = eng.transform(TEXT, scope_id="case_2")
    fake_full = r.entities[0].replacement            # 山田太郎 の偽名
    fake_sei = r.entities[2].replacement             # 山田 の偽姓（姓リンク）
    fake_org = r.entities[1].replacement
    assert fake_full.startswith(fake_sei)
    response = f"ご質問の件、{fake_sei}氏（{fake_full}様）は{fake_org}の所属で間違いありません。追加の確認事項は特にありません。"
    back = eng.restore(response, scope_id="case_2")
    assert back.text == "ご質問の件、山田氏（山田太郎様）は株式会社青葉電機の所属で間違いありません。追加の確認事項は特にありません。"
    assert back.restored == 3


def test_restore_is_exact_match_only_and_scope_isolated():
    eng = engine()
    r = eng.transform(TEXT, scope_id="case_3")
    fake_full = r.entities[0].replacement
    # 姓リンクの偽姓（fake_full の先頭部分）は、それ自体が 2 字以上の値付きトークンとして
    # 別途登録されている（姓リンク設計）。空白をどこに挿んでも姓の全体が文字列として無傷で
    # 残る位置（姓の後ろ）だと、その部分だけが正しく（設計どおり）exact match で復元されて
    # しまう。姓の「内部」に空白を挿んで分断すれば、登録済みのどの値ともexact matchしない
    # 表記ゆれになる
    spaced = fake_full[:1] + " " + fake_full[1:]                        # 表記ゆれは戻さない
    back = eng.restore(f"{spaced}へ連絡。", scope_id="case_3")
    assert back.text == f"{spaced}へ連絡。" and back.restored == 0
    other = eng.restore(fake_full, scope_id="another_scope")
    assert other.text == fake_full and other.restored == 0
    none = eng.restore("関係のない文です。", scope_id="case_3")
    assert none.text == "関係のない文です。" and none.restored == 0


def test_restore_replaces_longest_first():
    eng = engine()
    r = eng.transform(TEXT, scope_id="case_4")
    fake_full = r.entities[0].replacement
    fake_sei = r.entities[2].replacement
    back = eng.restore(f"{fake_full}と{fake_sei}", scope_id="case_4")
    assert back.text == "山田太郎と山田"


def test_store_without_iter_tokens_is_configuration_error():
    class OldStore(MemoryStore):
        iter_tokens = None                     # 旧 Store のふり

    eng = engine(store=OldStore())
    eng.transform(TEXT, scope_id="case_5")
    with pytest.raises(ConfigurationError):
        eng.restore("x", scope_id="case_5")


def test_vault_surrogate_requires_scope_and_crypto():
    eng = engine()
    with pytest.raises(ConfigurationError):
        eng.transform(TEXT)                      # scope_id 必須
    with pytest.raises(ConfigurationError):
        engine(crypto=None)


class _RejectAll:
    """全候補を拒否するフィルタのふり（tests/test_surrogate_semantic_operator.py の RejectAll と同型）。"""

    def reject_reason(self, candidate, **kw):
        return "denylist"


def test_fallback_token_saved_under_vault_round_trips():
    # フィルタが全候補を拒否すると Surrogate は型レベルの <PERSON_xxxxxxxx> トークンへ退避する
    # （surrogate.py: _draw → _fallback_token）。strategy="vault" ではこのトークンも
    # ctx.save_token(..., original_value=value) で対応表に保存される（Task 9）。
    # _surrogate_tokens は「<」始まりを除外していたが、それではこの唯一の正規復元経路が
    # 塞がれてしまうため、フィルタ条件を「値付きトークンであること」だけに絞った（本タスクでの修正）。
    text = "山田太郎に連絡してください。"
    ent = Entity("PERSON", 0, 4, 0.95)
    op = Surrogate(kind="person", strategy="vault", link_surnames=False)
    op._filter = _RejectAll()
    eng = engine(
        policy=Policy(rules={"PERSON": op}, default=Keep()),
        detectors=[FakeDetector([ent])],
    )
    r = eng.transform(text, scope_id="case_fallback")
    fake = r.entities[0].replacement
    assert fake.startswith("<PERSON_") and fake.endswith(">") and "山田太郎" not in fake

    back = eng.restore(f"{fake}に連絡してください。", scope_id="case_fallback")
    assert back.text == text
    assert back.restored == 1


def test_whitespace_only_value_is_saved_and_round_trips_under_vault():
    """Fix round 1（2026-09-28 保留 Minor レビュー）: Surrogate.apply は正規化後に空になる値
    （空白のみ含む）を _non_target_token に渡して fail-closed でトークン化する（修正前は
    extract_features が『姓のみ』扱いし、普通の姓に化けて退避されていなかった）。vault では
    このトークンも original_value=value で保存されるため、iter_tokens に載り、restore() で
    元の空白そのものに正確に戻る。"""
    text = "宛先：  様、よろしくお願いします。"
    start = text.index("  ")
    ent = Entity("PERSON", start, start + 2, 0.95)
    eng = engine(detectors=[FakeDetector([ent])])
    r = eng.transform(text, scope_id="ws-vault")
    fake = r.entities[0].replacement
    assert fake.startswith("<PERSON_") and fake.endswith(">")
    assert "  " not in r.text

    toks = [t for _, t in eng.store.iter_tokens("ws-vault")]
    assert toks == [fake]                                          # vault に保存済み

    back = eng.restore(r.text, scope_id="ws-vault")
    assert back.text == text and back.restored == 1 and back.unresolved == ()


def test_roundtrip_with_sqlite_store(tmp_path):
    # MemoryStore だけでなく永続 Store (SQLiteStore) でも iter_tokens 経由の復元が動くことを確認
    store = SQLiteStore(str(tmp_path / "vault.db"))
    try:
        eng = engine(store=store)
        r = eng.transform(TEXT, scope_id="case_sqlite")
        assert "山田太郎" not in r.text and "佐藤花子" not in r.text
        back = eng.restore(r.text, scope_id="case_sqlite")
        assert back.text == TEXT and back.unresolved == () and back.restored == 4
    finally:
        store.close()


def test_vault_person_surrogate_never_saves_a_token_shorter_than_2_chars():
    """レビュー指摘の再発防止（surrogate.py._restorable）: SURNAMES_KANJI には1字姓が
    3件（林・森・堀、140件中）あり、姓リンクの偽姓としてこれが引かれると以前は2字未満の
    トークンがvaultに保存され、restore()が「2字以上」ルールにより静かに片道になっていた。
    フルネーム（山田太郎）と裸姓のエイリアス（山田）の両方を含む TEXT を多数の scope で
    再抽選させ、iter_tokens に2字未満のトークンが一件も現れないことを直接検査する。"""
    eng = engine()
    for i in range(150):
        scope_id = f"case_len_{i}"
        eng.transform(TEXT, scope_id=scope_id)
        toks = list(eng.store.iter_tokens(scope_id))
        assert toks, "iter_tokens が空になってはいけない"
        assert all(len(t) >= 2 for _, t in toks), toks


def test_full_name_and_bare_surname_alias_roundtrip_without_denylist():
    """surrogate.py._restorable 修正後は、denylist で1字姓を明示的に避けなくても、
    フルネーム + 裸姓のエイリアス（姓リンク）の往復が常に成立する（1字姓を実際に引かせる
    ため多数回試行）。修正前はこの denylist 無しの往復が確率的に失敗していた。"""
    for i in range(150):
        eng = engine()
        scope_id = f"case_nodeny_{i}"
        r = eng.transform(TEXT, scope_id=scope_id)
        back = eng.restore(r.text, scope_id=scope_id)
        assert back.text == TEXT
        assert back.restored == 4
        assert back.unresolved == ()


def test_one_char_token_written_directly_is_left_untouched_by_restore():
    """restore()の「2字以上」ルール（設計§9）自体はSurrogate側の修正で変えていないことの
    ドキュメント。Surrogateを経由せず、store（scope API）に1字トークンを直接書き込んでも、
    restore()はそれを完全一致の対象にせず、原文のまま・restored=0・unresolved=()を返す。"""
    eng = engine()
    scope_id = "case_one_char_manual"
    token = "堀"                       # SURNAMES_KANJI に実在する1字姓（架空人物の代用表記）
    blob = eng.crypto.encrypt(encode_fields(scope_id, token, "山田"))
    eng.store.put_token(scope_id, "PERSON", "manual-value-hash", token, blob)

    text = f"{token}さんへ連絡してください。"
    back = eng.restore(text, scope_id=scope_id)
    assert back.text == text            # 1字トークンは戻らない（原文のまま）
    assert back.restored == 0
    assert back.unresolved == ()        # マッチ自体を試みないので unresolved にも出ない


def test_restorable_guard_is_vault_only_unit():
    """_restorable() 自体の単体テスト（決定的。round 1 のテストは1字姓を実際に引かせて
    確率的に検査していたため、ガードが壊れても見逃す確率が非ゼロだった）。"""
    vault_op = Surrogate(kind="person", strategy="vault")
    assert vault_op._restorable("林") is False           # 1字（SURNAMES_KANJIの1字姓）は拒否
    assert vault_op._restorable("林田") is True           # 2字以上は許可
    for strategy in ("hmac", "random"):
        op = Surrogate(kind="person", strategy=strategy)
        assert op._restorable("林") is True               # vault専用ガード。hmac/randomはノーオペ


def test_vault_surname_pool_exhaustion_falls_back_and_saves_both():
    """レビュー再指摘の再発防止: 偽姓プールが尽きたとき（フィルタで却下され続ける等）に
    _apply_person_linked が自前でフォールバックトークンを作って返すと、それは
    _apply_full の save_token（surrogate.py:249）を経由しないため vault に保存されず、
    restore() が無言で片道になる（Task 10 fix round 1 で新設した不具合）。
    _apply_person_linked は None を返して通常経路（_draw + 保存）へフォールスルーしなければ
    ならない。denylist で2字以上の姓を全滅させ、フルネームと裸姓の両方を同一文書に含めて
    往復させる。"""
    all_long_surnames = tuple(s for s in SURNAMES_KANJI if len(s) >= 2)
    text = "山田太郎さんと山田さんへ連絡。"
    bare_start = text.index("山田さん")
    ents = [Entity("PERSON", 0, 4, 0.95), Entity("PERSON", bare_start, bare_start + 2, 0.95)]
    op = Surrogate(kind="person", strategy="vault", denylist=all_long_surnames)
    eng = engine(
        policy=Policy(rules={"PERSON": op}, default=Keep()),
        detectors=[FakeDetector(ents)],
    )
    r = eng.transform(text, scope_id="exh")

    assert "山田" not in r.text  # フルネーム・裸姓とも原文には絶対に落ちない（"山田太郎"も含む）
    reps = {e.original_start: e.replacement for e in r.entities}
    # 裸姓は候補が全滅して型トークンへ退避する。フルネームは姓リンクをあきらめて通常経路で
    # 偽名を得る（denylist は候補全体の完全一致なので「偽姓+名」は通る）。Important I2 の修正で
    # フルネームがこのトークンを偽姓として採る（"<PERSON_xxxxxxxx>名"）ことは無くなった
    assert reps[bare_start].startswith("<PERSON_") and reps[bare_start].endswith(">")
    assert "<" not in reps[0] and "太郎" not in reps[0]

    toks = list(eng.store.iter_tokens("exh"))
    assert len(toks) == 2                                          # 2つの異なる入力 → 2トークン
    assert all(eng.store.get_value("exh", t) is not None for _, t in toks)  # 両方ともvault保存済み

    back = eng.restore(r.text, scope_id="exh")
    assert back.text == text
    assert back.restored == 2
    assert back.unresolved == ()


def test_legacy_generator_misconfiguration_still_raises_under_vault():
    """レビュー再指摘のMinor対応: vaultのフォールバックは「2字未満」「denylist等」が原因で
    候補が尽きたときだけ働く。生成器が原文と区別できる値を一度も作れない（利用者側の設定
    ミス）だけで尽きた場合は、vaultでも握りつぶさず ConfigurationError を送出し続ける。"""
    op = Surrogate(kind="person", strategy="vault", semantic=False, generator=lambda v, rng: v)
    eng = engine(
        policy=Policy(rules={"PERSON": op}, default=Keep()),
        detectors=[FakeDetector([Entity("PERSON", 0, 4, 0.95)])],
    )
    with pytest.raises(ConfigurationError):
        eng.transform("山田太郎です。", scope_id="misconfig")


def test_roundtrip_when_the_bare_surname_is_tokenized_before_the_full_name():
    """Important I2: 裸姓が先に処理されて（エンジンは後ろのスパンから置換する）姓プールが
    枯渇していると、裸姓は <PERSON_xxxxxxxx> トークンに退避される。その後に処理される
    フルネームの後方リンクがこの保存済みトークンを「偽姓」として採用してしまい、
    「<PERSON_xxxxxxxx>名」という偽名ができていた。

    restore() の alternation は Tokenize の書式が先だったため、この偽名の内側の
    <PERSON_xxxxxxxx> にだけ一致して偽名が途中で切られ、往復が壊れる（保存されている
    トークンは偽名全体なので、切られた側は復号できず unresolved になる）。既存の
    枯渇テストは policy に Tokenize が無く書式が空だったため、この破綻を踏まなかった。

    (a) 偽名リテラルを Tokenize の書式より前に置く、(b) 後方リンクは "<" で始まる
    保存済みトークンを偽姓として採らない、の両方を確認する。"""
    all_long_surnames = tuple(s for s in SURNAMES_KANJI if len(s) >= 2)
    text = "山田太郎さんと山田さんへ連絡。"          # 裸姓が後ろ = 裸姓が先に処理される
    bare_start = text.index("山田さん")
    ents = [Entity("PERSON", 0, 4, 0.95), Entity("PERSON", bare_start, bare_start + 2, 0.95)]
    op = Surrogate(kind="person", strategy="vault", denylist=all_long_surnames)
    eng = engine(
        policy=Policy(rules={"PERSON": op}, default=Tokenize(strategy="vault")),
        detectors=[FakeDetector(ents)],
    )
    r = eng.transform(text, scope_id="exh_bare_first")

    assert "山田" not in r.text
    reps = {e.original_start: e.replacement for e in r.entities}
    bare_fake, full_fake = reps[bare_start], reps[0]
    assert bare_fake.startswith("<PERSON_") and bare_fake.endswith(">")
    assert len(bare_fake) == len("<PERSON_>") + 8              # トークンに偽名の断片が続かない
    assert "<" not in full_fake and ">" not in full_fake       # 偽姓として型トークンを採らない
    assert "山田" not in full_fake and "太郎" not in full_fake

    back = eng.restore(r.text, scope_id="exh_bare_first")
    assert back.text == text
    assert back.unresolved == ()
    assert back.restored == 2


def test_restore_matches_a_saved_fake_before_a_shorter_tokenize_pattern_inside_it():
    """Important I2 (a): restore() の alternation は先に書いた枝が勝つ。保存された偽名が
    Tokenize の書式で始まる形だと、書式を先に置いていたため偽名の内側のトークンにだけ
    一致し、偽名が途中で切られて復号できない断片が残っていた（保存されているのは偽名全体）。

    (b) の修正でこの形の偽名は姓リンクからは作られなくなったが、restore 側の多重防御として、
    利用者の generator がその形を返した場合でも往復することを確認する。"""
    fake = "<PERSON_ABCD1234>某"
    op = Surrogate(kind="person", strategy="vault", semantic=False, generator=lambda v, rng: fake)
    eng = engine(
        policy=Policy(rules={"PERSON": op}, default=Tokenize(strategy="vault")),
        detectors=[FakeDetector([Entity("PERSON", 0, 4, 0.95)])],
    )
    text = "山田太郎さんへ連絡。"
    r = eng.transform(text, scope_id="alt_order")
    assert r.text == fake + "さんへ連絡。"

    back = eng.restore(r.text, scope_id="alt_order")
    assert back.text == text
    assert back.unresolved == ()
    assert back.restored == 1



def test_auto_non_target_type_token_is_saved_and_round_trips_under_vault():
    """性能計測（2026-09-27）で発見: ``kind="auto"`` が対象外の型（EMAIL・PHONE 等）を ``<TYPE_xxxxxxxx>`` に
    退避する経路は ``apply`` の冒頭で store に保存せず返していたため、vault で復元されず（unresolved にも
    出ない無言の片道）、同じ値が 2 回出ても別トークンになり得た。保存して往復させる。"""
    text = "電話は 052-000-0000、再掲 052-000-0000。"
    p1, p2 = text.index("052"), text.rindex("052")
    eng = engine(
        policy=Policy(default=Surrogate(kind="auto", strategy="vault")),
        detectors=[FakeDetector([Entity("PHONE_NUMBER", p1, p1 + 12, 0.99), Entity("PHONE_NUMBER", p2, p2 + 12, 0.99)])],
    )
    r = eng.transform(text, scope_id="auto-vault")
    toks = [t for _, t in eng.store.iter_tokens("auto-vault")]
    assert "052-000-0000" not in r.text
    assert len(toks) == 1 and toks[0].startswith("<PHONE_NUMBER_") and r.text.count(toks[0]) == 2
    back = eng.restore(r.text, scope_id="auto-vault")
    assert back.text == text and back.restored == 2 and back.unresolved == ()


def test_restore_batch_builds_the_scope_pattern_once(monkeypatch):
    """restore_batch は scope の対応表（偽名トークンの列挙）を文書ごとではなく 1 回だけ読む。

    ENTS は TEXT の位置に固定されているため、2 つ目の文書は別の Detector で検出する
    （FakeDetector(ENTS) をそのまま短い文書に使うと範囲外エンティティで DetectorError になる）。
    """
    text2 = "佐藤花子も来た。"
    ent2 = Entity("PERSON", text2.index("佐藤花子"), text2.index("佐藤花子") + len("佐藤花子"), 0.95)

    class TwoDocDetector:
        name = "fake"

        def detect(self, t):
            return ENTS if t == TEXT else [ent2]

    eng = engine(detectors=[TwoDocDetector()])
    r = [eng.transform(TEXT, scope_id="batch1"), eng.transform(text2, scope_id="batch1")]
    calls = []
    orig = eng._surrogate_tokens
    monkeypatch.setattr(eng, "_surrogate_tokens", lambda s: calls.append(s) or orig(s))
    backs = eng.restore_batch([x.text for x in r], scope_id="batch1")
    assert [b.text for b in backs] == [TEXT, text2] and calls == ["batch1"]


def test_restore_empty_text():
    eng = engine()
    eng.transform(TEXT, scope_id="e")
    assert eng.restore("", scope_id="e").text == "" and eng.restore("", scope_id="e").restored == 0


def test_auto_non_target_type_token_is_consistent_under_random_strategy():
    """random 戦略でも同一 scope の同じ値は同じトークン（store に保存して再利用する）。"""
    text = "電話は 052-000-0000、再掲 052-000-0000。"
    p1, p2 = text.index("052"), text.rindex("052")
    eng = engine(
        policy=Policy(default=Surrogate(kind="auto", strategy="random")),
        detectors=[FakeDetector([Entity("PHONE_NUMBER", p1, p1 + 12, 0.99), Entity("PHONE_NUMBER", p2, p2 + 12, 0.99)])],
    )
    r = eng.transform(text, scope_id="auto-random")
    toks = [t for _, t in eng.store.iter_tokens("auto-random")]
    assert len(toks) == 1 and r.text.count(toks[0]) == 2 and "052-000-0000" not in r.text

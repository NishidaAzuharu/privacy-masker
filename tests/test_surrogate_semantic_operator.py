"""Surrogate(semantic=True) の operator としての振る舞い（設計 §2・§7・§10）。"""
import pytest

import random

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.data import ja_pools as jp
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Surrogate
from privacy_masker.operators import surrogate as sur
from privacy_masker.operators.semantic import to_halfwidth_kana
from privacy_masker.types import Entity

from .conftest import SurrogateCtx

KEY = b"semantic-surrogate-test-key"


class Ctx(SurrogateCtx):
    """このファイルの既定鍵 KEY を使う SurrogateCtx。呼び出し側は従来どおり Ctx(scope_id=..., text=...)。
    NoKeyCtx（下方）がこのクラスを継承するため、関数（functools.partial）ではなくサブクラスにする。"""

    def __init__(self, scope_id="s1", text=None):
        super().__init__(KEY, scope_id=scope_id, text=text)


def E(t, n):
    return Entity(t, 0, n, 1.0, "fake")


def test_org_keeps_category_and_legal_form_position():
    op = Surrogate(kind="org", key=KEY)
    out = op.apply("株式会社青葉電機", E("ORG", 8), Ctx())
    assert out.startswith("株式会社") and out.endswith(jp.ORG_INDUSTRY_WORDS["manufacturing"]) and "青葉" not in out
    out = op.apply("東雲信用金庫", E("ORG", 6), Ctx())
    assert out.endswith(jp.ORG_INDUSTRY_WORDS["finance"]) and "東雲" not in out
    out = op.apply("営業二課", E("ORG", 4), Ctx())
    assert out.endswith(jp.DEPT_UNITS) and out != "営業二課"


def test_facility_and_location_kinds():
    fac = Surrogate(kind="facility", key=KEY).apply("汐見クリニック", E("FACILITY", 7), Ctx())
    assert fac.endswith(jp.FACILITY_WORDS["medical"]) and "汐見" not in fac
    loc = Surrogate(kind="location", key=KEY)
    assert loc.apply("大分県", E("LOCATION", 3), Ctx()) in set(jp.REGION_BLOCKS["九州・沖縄"]) - {"大分県"}
    city = loc.apply("札幌市", E("LOCATION", 3), Ctx())
    # PREF_CITIES_EXTRA（北海道の追加市。semantic 経路専用）も選択肢に入る
    assert city != "札幌市" and any(city == c and p == "北海道" for p, c in jp.PREF_CITIES + jp.PREF_CITIES_EXTRA)
    assert loc.apply("パリ", E("LOCATION", 2), Ctx()) in set(jp.FOREIGN_CITIES_KATA) - {"パリ"}
    assert loc.apply("日本", E("LOCATION", 2), Ctx()) in jp.COUNTRIES


def test_address_kind_respects_granularity_and_never_leaks_city_or_town():
    # Critical #1: gen_address が f.granularity を無視して番地まで捏造し、同一ブロック内の元の市区町村を
    # そのまま echo してしまっていた不具合の再発防止。scope_id を変えて hmac シードを 25 通り試す
    op = Surrogate(kind="address", key=KEY)
    for i in range(25):
        ctx = Ctx(scope_id=f"scope-{i}")
        city_out = op.apply("東京都新宿区", E("ADDRESS", 6), ctx)
        assert not city_out.startswith("東京都新宿区")
        assert "新宿区" not in city_out
        assert not any(ch.isdigit() for ch in city_out)

        pref_out = op.apply("大阪府", E("ADDRESS", 3), ctx)
        assert pref_out != "大阪府" and pref_out in jp.PREFECTURES
        assert not any(ch.isdigit() for ch in pref_out)

        banchi_out = op.apply("青凪市栄町2丁目5番地", E("ADDRESS", 10), ctx)
        assert any(ch.isdigit() for ch in banchi_out)          # 番地レベルは番地のまま
        assert "青凪市" not in banchi_out and "栄町" not in banchi_out

        building_out = op.apply("サンモールビル別館302号室", E("ADDRESS", 13), ctx)
        assert "号室" in building_out                          # 建物レベルは建物語が残る
        assert building_out != "サンモールビル別館302号室"


def test_location_surrogate_never_leaks_original_place_name_or_stem():
    """LOCATION の偽名が元の地名やその語幹（例: 「札幌市」の「札幌」）をそのまま含んでしまっていた
    不具合（avoid_parts("location", ...) が常に () を返していた）の再発防止。鍵を20通り変えて確認する
    （北海道の市区町村プールは「札幌市中央区」「旭川市」の2件しかなく、回避後は「旭川市」しか残らない
    ——それ自体は許容する。フォールバックトークン化した鍵があれば、テストを緩めずに報告する）。"""
    keys = [f"location-avoid-parts-key-{i}".encode() for i in range(20)]
    for key in keys:
        op = Surrogate(kind="location", key=key)
        for value, must_not_leak in (("札幌市", "札幌"), ("東京都", "東京")):
            out = op.apply(value, E("LOCATION", len(value)), Ctx())
            assert out, (key, value)
            assert must_not_leak not in out, (key, value, out)
            is_fallback_token = out.startswith("<LOCATION_") and out.endswith(">")
            assert not is_fallback_token, (key, value, out)  # フォールバックに退避したら緩めず報告する


@pytest.mark.parametrize("value", [
    "米国", "仏国", "大韓民国", "中華民国", "ギリシア", "ソ連", "アメリカ合衆国", "西ドイツ", "スコットランド",
])
def test_location_surrogate_never_reuses_the_canonical_name_of_a_country_alias(value):
    """Task 3 レビュー Important I1: 国名の別名・略称・前身国家（米国・仏国・大韓民国・中華民国・ギリシア・ソ連・
    アメリカ合衆国・西ドイツ・スコットランド 等）が新たに country 判定されるようになったが、avoid_parts が元の
    値そのものしか避けず、COUNTRIES の標準名（アメリカ・フランス・韓国・台湾・ギリシャ・ロシア・イギリス・ドイツ）が
    同じ referent のまま偽名に出ていた（実測 400 件中: 米国→アメリカ 6、仏国→フランス 12、大韓民国→韓国 9、
    中華民国→台湾 10、ギリシア→ギリシャ 7、ソ連→ロシア 8。本文なし・本文ありの両方で同じ割合）。
    レビュー指摘（N4b）: 150 scope では「大韓民国」が修正前のコードでも偶然 0/150 になり検出力が足りなかったため、
    scope を 400 に引き上げた（レビューの実測と同じ規模）。"""
    canon = jp.COUNTRY_CANONICAL[value]
    op = Surrogate(kind="location", key=KEY)
    for i in range(400):
        out = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"alias-{i}"))
        assert out != canon, (i, value, out)
        text = f"{value}に出張した。"
        out2 = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"alias-{i}", text=text))
        assert out2 != canon, (i, value, out2, "with text")


def test_location_surrogate_never_leaks_katakana_state_or_province_stem():
    """Task 3 レビュー Important I2: カタカナ+州・省（例: ニューヨーク州）が foreign 判定され
    FOREIGN_CITIES_KATA から偽名を選ぶが、_LOCATION_SUFFIX（都道府県市区町村）に州・省が無いため語幹
    （ニューヨーク）が avoid_parts で避けられず、本文なし（structured 経路）で 400 件中 9 件が語幹を
    含んでいた。scope を変えて一度も語幹を含まないことを確認する（本文なし・本文ありの両方）。
    レビュー指摘（N4a）: 「ハルカゼ州」は FOREIGN_CITIES_KATA のどの項目にも「ハルカゼ」を含むものが無いため
    漏れようがなく検出力ゼロのテストだった。語幹がプールの実在語（FOREIGN_CITIES_KATA の「ニューヨーク」）と
    一致する「ニューヨーク州」に差し替える。"""
    value, stem = "ニューヨーク州", "ニューヨーク"
    op = Surrogate(kind="location", key=KEY)
    for i in range(400):
        out = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"state-{i}"))
        assert stem not in out, (i, out)
        text = f"{value}へ行った。"
        out2 = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"state-{i}", text=text))
        assert stem not in out2, (i, out2, "with text")


@pytest.mark.parametrize("value, forbidden", [
    ("ロシア連邦", "ロシア"), ("フランス共和国", "フランス"),
])
def test_location_surrogate_never_leaks_the_short_country_name_inside_an_official_long_form(value, forbidden):
    """fix round 2 N1: _LOC_COUNTRY_SUFFIX（共和国|帝国|合衆国|連邦）で country になった正式名は
    COUNTRY_CANONICAL に無いため、同じ国の短い名前（COUNTRIES の項目）が偽名にそのまま出ていた
    （実測 400 件中、本文なし = structured 経路: ロシア連邦→ロシア 6、フランス共和国→フランス 18）。
    scope を変えて一度も短い名前と一致しないことを確認する。"""
    op = Surrogate(kind="location", key=KEY)
    for i in range(400):
        out = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"longform-{i}"))
        assert out != forbidden, (i, value, out)


def test_location_surrogate_never_leaks_the_short_country_name_with_text_present():
    """fix round 2 N1: 「大英帝国」は COUNTRIES の項目「イギリス」を部分文字列に含まないため、本文ありでも
    既存の文書内一致チェックでは救われず、本文なし・本文ありの両方で漏れていた（実測 400 件中 6/400, 6/400）。
    COUNTRY_CANONICAL に「大英帝国」→「イギリス」を追加したことで、本文ありでも一致しないことを確認する。"""
    op = Surrogate(kind="location", key=KEY)
    for i in range(400):
        text = "大英帝国について話した。"
        out = op.apply("大英帝国", E("LOCATION", 4), SurrogateCtx(KEY, scope_id=f"britain-{i}", text=text))
        assert out != "イギリス", (i, out)


@pytest.mark.parametrize("value, core", [("若葉地区", "若葉"), ("青葉1区", "青葉"), ("本町3区", "本町")])
def test_location_surrogate_never_leaks_district_or_electoral_stem(value, core):
    """Task 3 レビュー Minor M1: 地区・選挙区は avoid_parts が末尾 1 文字（都道府県市区町村州省）しか
    外さないため語幹が避けられず、district 生成器（TOWNS ∪ DISTRICT_WORDS）がコア語を再利用しうる
    （本文あり・実測 400 件中: 若葉地区 11、青葉1区 18、本町3区 14）。scope を変えて一度も語幹を含まないこと。"""
    op = Surrogate(kind="location", key=KEY)
    for i in range(150):
        text = f"{value}で会合があった。"
        out = op.apply(value, E("LOCATION", len(value)), SurrogateCtx(KEY, scope_id=f"dist-{i}", text=text))
        assert core not in out, (i, out)


def test_auto_kind_dispatches_by_entity_type_and_falls_back_for_unknown_types():
    op = Surrogate(kind="auto", key=KEY)
    assert op.apply("山田太郎", E("PERSON", 4), Ctx()) != "山田太郎"
    assert op.apply("青葉総合病院", E("FACILITY", 6), Ctx()).endswith(jp.FACILITY_WORDS["medical"])
    tok = op.apply("a@example.com", E("EMAIL", 13), Ctx())
    assert tok.startswith("<EMAIL_") and tok.endswith(">") and len(tok) == len("<EMAIL_>") + 8
    assert tok == op.apply("a@example.com", E("EMAIL", 13), Ctx())        # hmac で決定的


def test_person_preserves_gender_hint_and_scripts_with_hmac():
    op = Surrogate(kind="person", key=KEY)
    ctx = Ctx()
    m = op.apply("山田太郎", E("PERSON", 4), ctx)
    assert any(m.endswith(g) for g in jp.GIVEN_KANJI_M) and not m.startswith("山田")
    f = op.apply("佐藤花子", E("PERSON", 4), ctx)
    assert any(f.endswith(g) for g in jp.GIVEN_KANJI_F)
    assert m.startswith(op.apply("山田", E("PERSON", 2), ctx))          # 姓リンク維持（偽姓は 1〜3 字）
    assert op.apply("みやさか", E("PERSON", 4), ctx) in jp.SURNAMES_HIRA + jp.GIVEN_HIRA
    rom = op.apply("John Smith", E("PERSON", 10), ctx)
    assert rom.split(" ")[1] in jp.SURNAMES_ROMAJI
    hw = op.apply("ｷﾀﾞ ｹﾞﾞﾎ", E("PERSON", 7), ctx)
    assert all(ord(ch) >= 0xFF61 or ch == " " for ch in hw)


def test_present_in_document_and_knowledge_filters():
    text = "佐藤健一と鈴木一郎は同僚だ。"
    op = Surrogate(kind="person", strategy="random")
    ctx = Ctx(text=text)
    out = op.apply("田中太郎", E("PERSON", 4), ctx)
    assert out not in text and "田中" not in out
    k = PublicKnowledge.from_iterable([])
    op2 = Surrogate(kind="org", key=KEY, knowledge=k, denylist=("株式会社曙電機",))
    out2 = op2.apply("株式会社青葉電機", E("ORG", 8), Ctx())
    assert out2 != "株式会社曙電機"


def test_exhausted_filter_returns_token_never_original():
    class RejectAll:
        def reject_reason(self, candidate, **kw):
            return "denylist"

    op = Surrogate(kind="org", key=KEY)
    op._filter = RejectAll()
    out = op.apply("株式会社青葉電機", E("ORG", 8), Ctx())
    assert out.startswith("<ORG_") and "青葉" not in out


def test_vault_saves_original_and_alias():
    op = Surrogate(kind="person", strategy="vault")
    ctx = Ctx()
    fake = op.apply("山田太郎", E("PERSON", 4), ctx)
    assert ctx.saved_originals[fake] == "山田太郎"
    sei_fake = op.apply("山田", E("PERSON", 2), ctx)
    assert fake.startswith(sei_fake) and ctx.saved_originals[sei_fake] == "山田"


def test_semantic_false_is_legacy_behaviour():
    legacy = Surrogate(kind="org", key=KEY, semantic=False).apply("東雲信用金庫", E("ORG", 6), Ctx())
    assert legacy.endswith("信用金庫") and legacy.startswith(jp.ORG_DIRECTIONS)
    with pytest.raises(ConfigurationError):
        Surrogate(kind="facility", keep="city")
    with pytest.raises(ConfigurationError):
        Surrogate(kind="person", strategy="nope")


def test_facility_or_location_with_semantic_false_raises_configuration_error():
    """kind='facility'/'location' には v0.8 の非semantic生成器が無いため、構築時に拒否する
    （コンストラクタのガード。実行時に auto 経由ですり抜けないことは Important #4 の別テストで確認済み）。"""
    with pytest.raises(ConfigurationError):
        Surrogate(kind="facility", semantic=False)
    with pytest.raises(ConfigurationError):
        Surrogate(kind="location", semantic=False)


def test_person_empty_value_falls_back_to_token():
    """正規化後に空になる値（空文字・空白のみ）は fail-closed でトークンに退避する
    （Fix round 1: Surrogate.apply が norm 計算直後に not norm を見て _non_target_token へ渡すよう修正。
    修正前は extract_features が短い値を『姓のみ』扱いするため、_linkable を経ずに semantic の
    生成器が普通の姓を返してしまい、退避すべき退化値が退避されていなかった）。"""
    op = Surrogate(kind="person", key=KEY)
    out = op.apply("", E("PERSON", 0), Ctx())
    assert out.startswith("<PERSON_") and out.endswith(">")

    # 空白のみの値も同じ扱い（default_normalize が空白を全除去するため norm も空になる）。
    # 正規化後の値が同じ ("") なので同一 scope では同一トークンになる。
    ws_out = op.apply("  ", E("PERSON", 2), Ctx())
    assert ws_out == out


def test_person_single_char_surname_uses_surname_pool():
    """"林"（姓プールに実在する 1 字姓）は avoid_parts が 2 字未満を素通しする
    （SurrogateFilter.reject_reason の avoid_parts ガード、original_norm の同一性チェックのみ効く）ため、
    姓プールの別の姓に置き換わる。この鍵では 2 字以上の姓が引かれる。"""
    op = Surrogate(kind="person", key=KEY)
    out = op.apply("林", E("PERSON", 1), Ctx())
    assert out != "林" and out in jp.SURNAMES_KANJI and len(out) >= 2


def test_bare_surname_path_honors_denylist():
    # Important #2: _draw_sei が SurrogateFilter を一切通っておらず、denylist/knowledge が
    # 姓のみの出力に効かなかった不具合の再発防止（check_public=False は意図的、denylist は掛かる）
    banned = jp.SURNAMES_KANJI[10]
    op = Surrogate(kind="person", denylist=(banned,))
    for i in range(200):
        ctx = Ctx(scope_id=f"scope-{i}")
        out = op.apply("山田", E("PERSON", 2), ctx)
        assert out != banned


def test_semantic_false_honors_denylist():
    # Important #3: legacy 経路（semantic=False）は reject_reason が "public_entity" 以外
    # （denylist 等）を返しても再抽選していなかった
    baseline = Surrogate(kind="org", key=KEY, semantic=False).apply("東雲信用金庫", E("ORG", 6), Ctx())
    op = Surrogate(kind="org", key=KEY, semantic=False, denylist=(baseline,))
    out = op.apply("東雲信用金庫", E("ORG", 6), Ctx())
    assert out != baseline


def test_semantic_false_linked_person_honors_denylist():
    # _ok_linked の semantic=False 分岐が reject_reason を "public_entity" としか比較しておらず、
    # denylist/same_as_original を無視していた（Important #3 と同種、姓リンク経路側）
    baseline = Surrogate(kind="person", key=KEY, semantic=False).apply("山田太郎", E("PERSON", 4), Ctx())
    op = Surrogate(kind="person", key=KEY, semantic=False, denylist=(baseline,))
    out = op.apply("山田太郎", E("PERSON", 4), Ctx())
    assert out != baseline


def test_semantic_false_auto_kind_falls_back_for_facility_and_location():
    # Important #4: kind="auto" は実行時に entity_type から解決されるため、コンストラクタの
    # 「kind in ('facility','location') なら semantic=False を拒否する」ガードをすり抜けて
    # semantic の生成器に落ちていた不具合の再発防止
    op = Surrogate(kind="auto", key=KEY, semantic=False)
    fac = op.apply("汐見クリニック", E("FACILITY", 7), Ctx())
    assert fac.startswith("<FACILITY_") and fac.endswith(">") and "汐見" not in fac
    loc = op.apply("大分県", E("LOCATION", 3), Ctx())
    assert loc.startswith("<LOCATION_") and loc.endswith(">") and "大分" not in loc
    assert op.apply("山田太郎", E("PERSON", 4), Ctx()) != "山田太郎"     # 対象の型は従来どおり


def test_surname_pool_exhaustion_never_raises_and_falls_back_to_a_token():
    """Important I5: 偽姓が尽きたとき None を返して通常経路へ落ちるのは vault だけで、
    hmac / random は ConfigurationError を送出していた。設計では「フィルタ全滅 → トークン」
    なので、transform() が例外で止まってはいけない（裸姓・フルネームの両方の経路を通す）。

    姓プール全部を denylist に入れると、裸姓は候補が候補プールそのものなので全滅して
    トークンへ退避する。フルネームは姓リンクをあきらめたあと通常経路で「偽姓+名」を作り、
    denylist（候補全体の完全一致）には掛からないので普通の偽名になる。"""
    op = Surrogate(kind="person", key=KEY, denylist=tuple(jp.SURNAMES_KANJI))
    ctx = Ctx()

    bare = op.apply("山田", E("PERSON", 2), ctx)
    assert bare.startswith("<PERSON_") and bare.endswith(">")
    assert len(bare) == len("<PERSON_>") + 8

    full = op.apply("山田太郎", E("PERSON", 4), ctx)          # 修正前はここで ConfigurationError
    assert full != "山田太郎" and "山田" not in full and "太郎" not in full
    assert "<" not in full


def test_surname_draw_scans_the_whole_pool_before_giving_up():
    """Important I5（続き）: 偽姓の抽選はランダム 10 回で諦めていたため、文書に常用姓が
    大量に出ている（present_in_document で弾かれる）だけで、まだプールに使える姓が
    残っているのにトークンへ退避していた。ランダム試行の後にプール全体を走査する。"""
    crowded = "".join(jp.SURNAMES_KANJI[:-1])          # 1 つを除く全部が文書に出ている
    op = Surrogate(kind="person", key=KEY)
    out = op.apply("山田", E("PERSON", 2), Ctx(text=crowded))
    assert not out.startswith("<PERSON_")
    assert out == jp.SURNAMES_KANJI[-1]                 # 残り 1 つを必ず見つける


def test_bare_surname_keeps_its_script_instead_of_becoming_kanji():
    """Important I4: _linkable が「カタカナ優勢でなければ漢字の姓リンク経路」だったため、
    平仮名・ローマ字・半角カナの裸姓（正規化後 3 字以下）まで漢字の偽姓に置き換わり、
    文字種が保存されていなかった。semantic=True では漢字を含む値だけを姓リンク経路に入れ、
    それ以外は semantic の生成器（script に応じたプール）に回す。"""
    op = Surrogate(kind="person", key=KEY)

    hira = op.apply("たなか", E("PERSON", 3), Ctx())
    assert hira != "たなか"
    assert hira in jp.SURNAMES_HIRA + jp.GIVEN_HIRA
    assert all("ぁ" <= ch <= "ゖ" or ch == "ー" for ch in hira)

    rom = op.apply("Ito", E("PERSON", 3), Ctx())
    assert rom != "Ito" and rom in jp.SURNAMES_ROMAJI

    hw = op.apply("ｻﾄｳ", E("PERSON", 3), Ctx())
    assert hw != "ｻﾄｳ" and all(0xFF61 <= ord(ch) <= 0xFF9F for ch in hw)
    assert hw in {to_halfwidth_kana(x) for x in jp.SURNAMES_KATA_JA}

    kata = op.apply("アキ", E("PERSON", 2), Ctx())              # 既存の挙動（カタカナのまま）
    assert kata != "アキ" and kata in jp.SURNAMES_KATA_JA + jp.NICKNAMES_KATA


def test_kanji_bare_surname_still_uses_the_surname_link():
    """C4 の絞り込みで漢字の姓リンク（フルネームと裸姓の連携）が壊れていないこと。"""
    op = Surrogate(kind="person", key=KEY)
    ctx = Ctx()
    full = op.apply("山田太郎", E("PERSON", 4), ctx)
    assert full.startswith(op.apply("山田", E("PERSON", 2), ctx))
    mixed = op.apply("山田タロウ", E("PERSON", 5), ctx)         # 漢字を含む混在名も姓リンク経路
    assert mixed.startswith(op.apply("山田", E("PERSON", 2), ctx))


def test_semantic_false_bare_surname_scripts_are_unchanged():
    """C4 の絞り込みは semantic=True のときだけ。semantic=False（v0.8 互換）の出力は不変。"""
    op = Surrogate(kind="person", key=KEY, semantic=False)
    for value in ("たなか", "Ito", "ｻﾄｳ"):
        assert op.apply(value, E("PERSON", len(value)), Ctx()) in jp.SURNAMES_KANJI




class _PickyRng:
    """名のプールに対する ``choice`` だけ必ずフィルタに落ちる名を返し、それ以外（姓プール等）と
    ``sample`` は seed 固定の本物の乱数に委ねるスタブ。「ランダム試行だけが全滅する」状況を
    決定的に作るために使う。"""

    def __init__(self, given_pool, bad, seed=7):
        self._r = random.Random(seed)
        self._pool = tuple(given_pool)
        self._bad = bad
        self.given_choices = 0

    def choice(self, seq):
        if tuple(seq) == self._pool:
            self.given_choices += 1
            return self._bad
        return self._r.choice(seq)

    def sample(self, seq, k):
        return self._r.sample(seq, k)


def _picky(strategy):
    """「どの偽姓と組んでも『偽姓+bad』は必ず却下される」operator と、その stub rng を作る。
    姓そのものは denylist に入れないので偽姓は普通に引ける。"""
    kw = {"key": KEY} if strategy == "hmac" else {}
    pool = Surrogate(kind="person", strategy=strategy, **kw)._given_pool("山田太郎")
    bad = pool[0]
    op = Surrogate(kind="person", strategy=strategy,
                   denylist=tuple(sei + bad for sei in jp.SURNAMES_KANJI), **kw)
    return op, pool, bad, _PickyRng(pool, bad)


def test_linked_fullname_scans_the_given_pool_when_every_random_draw_is_rejected(monkeypatch):
    """Important I5 の残り: 姓リンクのフルネーム側（偽姓+名）はランダム 10 回で諦めていたため、
    名の候補がフィルタに落ち続けるだけで姓リンクを捨ててトークンへ退避していた（レビュアー計測で
    約 1/5,000 のフレーク）。_draw_sei と同じくランダム試行の後にプール全体を走査する。"""
    op, pool, bad, stub = _picky("random")
    monkeypatch.setattr(sur, "_SYS_RNG", stub)

    out = op.apply("山田太郎", E("PERSON", 4), Ctx())
    assert stub.given_choices == 10                 # ランダム試行は使い切った（走査が効いている）
    assert not out.startswith("<PERSON_")            # 修正前はここでトークンに退避していた
    assert out.endswith(tuple(g for g in pool if g != bad))
    assert "山田" not in out and "太郎" not in out


def test_linked_fullname_pool_scan_applies_to_hmac_too(monkeypatch):
    """hmac 分岐（_hmac_rng で引く側）にも同じ走査が入っていること。"""
    op, pool, bad, stub = _picky("hmac")
    monkeypatch.setattr(Surrogate, "_hmac_rng", lambda self, ctx, etype, key_norm: stub)

    out = op.apply("山田太郎", E("PERSON", 4), Ctx())
    assert stub.given_choices == 10
    assert not out.startswith("<PERSON_")
    assert out.endswith(tuple(g for g in pool if g != bad))
    assert "山田" not in out and "太郎" not in out


class _RejectAllFilter:
    def reject_reason(self, candidate, **kw):
        return "denylist"


def test_legacy_bare_katakana_person_is_exempt_from_public_check_and_never_raises():
    """semantic=False の v0.8 生成器はカタカナ単独名をカタカナ単独プール（60 語）から引くが、そのうち 58 語が
    Wikipedia 見出し語と衝突する。姓だけ・名だけと同じ辞書語なので公知照合の対象外にしないと、実文書では
    10 回連続で却下されて ConfigurationError になっていた（2026-09-27 の実測: 1,870 文書中 36 文書）。"""
    pool = set(jp.GIVEN_KATA + jp.SURNAMES_KATA)
    for i in range(60):
        op = Surrogate(kind="person", key=b"k%d" % i, semantic=False)
        out = op.apply("リオネ", E("PERSON", 3), Ctx())
        assert out in pool and out != "リオネ"


@pytest.mark.parametrize("strategy", ["hmac", "random"])
def test_legacy_filter_exhaustion_falls_back_to_token_not_exception(strategy):
    """semantic=False でも「フィルタ全滅 → 退避トークン」（設計 §7）。vault 以外で ConfigurationError を送出しない。"""
    op = Surrogate(kind="org", key=KEY if strategy == "hmac" else None, strategy=strategy, semantic=False)
    op._filter = _RejectAllFilter()
    out = op.apply("青凪電機株式会社", E("ORG", 8), Ctx())
    assert out.startswith("<ORG_") and out.endswith(">") and "青凪" not in out


def test_legacy_generator_misconfiguration_still_raises_under_hmac():
    """生成器が原文と区別できる値を一度も作れない設定ミスは、戦略に関わらず送出し続ける。"""
    op = Surrogate(kind="person", key=KEY, semantic=False, generator=lambda v, rng: v)
    with pytest.raises(ConfigurationError):
        op.apply("山田太郎", E("PERSON", 4), Ctx())


def test_draw_skips_redundant_other_round_but_keeps_the_retry_budget(monkeypatch):
    """カテゴリが既に OTHER のとき 2 周目（型レベル退避）は 1 周目と同一なので 1 周にまとめるが、試行回数は
    2 周分（2 * _DRAW_RETRIES）を保つ（最終レビュー M2: 10 回に半減して、従来 11〜20 回目に通っていた値が
    トークンに落ちるようになっていた）。

    surrogate.py は `from .surrogate_generators import generate` で名前を束縛しているため、
    surrogate_generators モジュール側を差し替えても _draw() が参照する名前には反映されない。
    _draw() のグローバル名前空間である surrogate モジュール側（sur.generate）を差し替える。
    """
    op = Surrogate(kind="facility", key=KEY)
    op._filter = _RejectAllFilter()
    calls = []
    orig = sur.generate
    monkeypatch.setattr(sur, "generate", lambda f, rng: calls.append(f.category) or orig(f, rng))
    out = op.apply("謎の建物", E("FACILITY", 4), Ctx())
    assert out.startswith("<FACILITY_") and len(calls) == 2 * sur._DRAW_RETRIES and set(calls) == {"other"}


class _RejectFirstNFilter:
    """最初の n 候補だけ却下するフィルタのふり。"""

    def __init__(self, n):
        self.n, self.seen = n, 0

    def reject_reason(self, candidate, **kw):
        self.seen += 1
        return "denylist" if self.seen <= self.n else None


def test_other_category_value_accepted_on_a_late_draw_is_not_turned_into_a_token():
    """M2: OTHER のカテゴリで 11〜20 回目の候補が通る場合も、従来どおり偽名になる（トークンに落ちない）。
    抽選の順序も 2 周に分けていたときと同じ（同じ特徴で同じ rng から引く）なので、hmac の値も変わらない。"""
    op = Surrogate(kind="facility", key=KEY)
    op._filter = _RejectFirstNFilter(2 * sur._DRAW_RETRIES - 1)
    out = op.apply("謎の建物", E("FACILITY", 4), Ctx())
    assert not out.startswith("<FACILITY_") and op._filter.seen == 2 * sur._DRAW_RETRIES


def test_fallback_token_only_swallows_missing_key_for_random_strategy():
    """_fallback_token は ConfigurationError（鍵が無い）を無条件に握りつぶしていたため、
    hmac/vault の設定ミスで鍵が無いときも黙って乱数トークンへ落ち、決定性が失われていた。
    random 戦略（鍵の有無を気にしない）だけがこの握りつぶしの対象。

    最終レビュー I1: strategy="hmac" の偽名経路は _fallback_token より前に _hmac_rng で送出するため、そこで
    確かめても修正前から通ってしまう。_fallback_token に直接届く vault のフィルタ全滅と、auto の対象外の型
    （hmac でも _hmac_rng を通らない _non_target_token）で確かめる。"""
    class NoKeyCtx(Ctx):
        def hmac_token(self, t, n, key=None, length=8):
            if key is None:
                raise ConfigurationError("no key")
            return super().hmac_token(t, n, key, length)

    # (a) vault + フィルタ全滅 → _fallback_token（修正前は乱数の <ORG_…> を返していた）
    vault = Surrogate(kind="org", strategy="vault")                  # 鍵は ctx 任せだが ctx にも無い
    vault._filter = _RejectAllFilter()
    with pytest.raises(ConfigurationError):
        vault.apply("青凪電機", E("ORG", 4), NoKeyCtx())
    # (b) auto + hmac で対象外の型（EMAIL）→ _non_target_token → _fallback_token
    auto = Surrogate(kind="auto", strategy="hmac")
    with pytest.raises(ConfigurationError):
        auto.apply("someone@example.com", E("EMAIL", 19), NoKeyCtx())
    # (c) random は鍵が無くても乱数トークンでよい
    rnd = Surrogate(kind="org", strategy="random")
    rnd._filter = _RejectAllFilter()
    out = rnd.apply("青凪電機", E("ORG", 4), NoKeyCtx())
    assert out.startswith("<ORG_") and out.endswith(">") and "青凪" not in out
    assert Surrogate(kind="auto", strategy="random").apply(
        "someone@example.com", E("EMAIL", 19), NoKeyCtx()).startswith("<EMAIL_")


def test_org_and_facility_fakes_never_reuse_the_core_before_an_inner_type_word():
    """最終レビュー C1: 途中に別の種別語を含む名前（「ネットワークス商船」「タワー美術館」）でコア語が偽名に
    混入していた（実測 52/300・62/300）。鍵を 300 通り変えて一度も含まないこと。"""
    for i in range(300):
        key = f"core-{i}".encode()
        org = Surrogate(kind="org", key=key).apply("株式会社ネットワークス商船", E("ORG", 13), SurrogateCtx(key))
        assert "ネットワークス" not in org, (i, org)
        fac = Surrogate(kind="facility", key=key).apply("タワー美術館", E("FACILITY", 6), SurrogateCtx(key))
        assert "タワー" not in fac, (i, fac)


def test_org_committee_fakes_never_reuse_the_core_for_non_adjacent_modifiers():
    """fix round 2 N2: 委員会の修飾語に「間に最大 6 字まで許す」形を入れたところ、「中央若葉執行委員会」
    「労働青葉安全委員会」のような私的な委員会まで government に誤分類され、しかも一致位置が文字列先頭に
    なるため avoid_parts がコアを切り出せず（government の偽名生成器は ORG_CORES を使うため）地名語幹が
    漏れていた（実測 400 件中: 中央若葉執行委員会 33、労働青葉安全委員会 36）。修飾語を委員会に直接隣接する
    ものだけに戻し、正しく department（偽名生成器は DEPT_FUNCTIONS を使い地名語幹を再利用しない）になった
    ことを、scope を 400 通り変えてコアを一度も含まないことで確認する（本文なし・本文ありの両方）。"""
    for value, core in (("中央若葉執行委員会", "若葉"), ("労働青葉安全委員会", "青葉")):
        for i in range(400):
            out = Surrogate(kind="org", key=KEY).apply(value, E("ORG", len(value)), SurrogateCtx(KEY, scope_id=f"committee-{i}"))
            assert core not in out, (i, value, out)
            text = f"{value}の会合。"
            out2 = Surrogate(kind="org", key=KEY).apply(value, E("ORG", len(value)), SurrogateCtx(KEY, scope_id=f"committee-{i}", text=text))
            assert core not in out2, (i, value, out2, "with text")


def test_org_committee_fakes_never_reuse_the_core_for_removed_province_prefix():
    """fix round 2 N3: 自治体接頭パターンの文字クラスに 省|州 を足したところ、「北斗省エネ推進委員会」
    「九州汐見支社委員会」のような私的な委員会まで government に誤分類され、地名語幹が漏れていた
    （実測 400 件中: 北斗省エネ推進委員会 35、九州汐見支社委員会 25）。省|州 を撤回し、正しく department に
    なったことを、scope を 400 通り変えてコアを一度も含まないことで確認する。"""
    value, core = "北斗省エネ推進委員会", "北斗"
    for i in range(400):
        out = Surrogate(kind="org", key=KEY).apply(value, E("ORG", len(value)), SurrogateCtx(KEY, scope_id=f"province-{i}"))
        assert core not in out, (i, out)


@pytest.mark.parametrize("kind, entity_type, value, parts", [
    ("org", "ORG", "中央青葉人事委員会", ("青葉",)),
    ("org", "ORG", "若葉県教育委員会", ("若葉",)),
    ("org", "ORG", "汐見町議会", ("汐見",)),
    ("facility", "FACILITY", "北斗青葉記念館", ("北斗", "青葉")),
])
def test_fake_core_vocabulary_words_embedded_in_the_core_never_leak(kind, entity_type, value, parts):
    """fix round 3: コーディネーターレビューで、Task 3 レビューの残課題（中央青葉人事委員会 → core 中央青葉、
    青葉 ∈ ORG_CORES が avoid されない）と 2 件の parked finding（若葉県教育委員会 22/300・汐見町議会 15/300 が
    若葉/汐見 を再利用）が同じ根本原因（core_split が切り出すコアをひとかたまりでしか扱わず、内部に埋もれた
    生成器の固有部分の語彙が個別には avoid されない）と判明。avoid_parts が語彙語を追加で avoid するように
    なったことを、scope を 400 通り変えて一度も含まないこと、かつフォールバックトークンに退避しないこと
    （生成に失敗していないこと）で確認する。"""
    prefix = f"<{entity_type}_"
    for i in range(400):
        out = Surrogate(kind=kind, key=KEY).apply(value, E(entity_type, len(value)), SurrogateCtx(KEY, scope_id=f"vocab-{i}"))
        for part in parts:
            assert part not in out, (i, value, out)
        assert not (out.startswith(prefix) and out.endswith(">")), (i, value, out)  # フォールバックに退避したら緩めず報告する
        text = f"{value}について話した。"
        out2 = Surrogate(kind=kind, key=KEY).apply(value, E(entity_type, len(value)), SurrogateCtx(KEY, scope_id=f"vocab-{i}", text=text))
        for part in parts:
            assert part not in out2, (i, value, out2, "with text")
        assert not (out2.startswith(prefix) and out2.endswith(">")), (i, value, out2, "with text")


@pytest.mark.parametrize("value, core", [
    ("若葉商事", "若葉"), ("青葉ホーム", "青葉"), ("東雲電気", "東雲"), ("若葉サービス", "若葉"), ("北斗グループ", "北斗"),
])
def test_org_fakes_never_reuse_the_core_for_category_neutral_suffixes(value, core):
    """レビュー対応 IMPORTANT 1: 商事・ホーム・電気・グループ（従来からの既知欠陥としてサービスも）は
    _ORG_WORDS からカテゴリ語として外れている（= OTHER/service）ため、core_split が表のどの行にも
    一致せず、avoid_parts が名前全体しか避けられずコアが偽名に混入していた
    （実測: 若葉商事 18/300・青葉ホーム 22/300・東雲電気 24/300、既存の若葉サービス 16/300・
    北斗グループ 16/300）。core_split が _ORG_CORE_SUFFIXES にも当たるようになったことを、
    鍵を 300 通り変えてコアを一度も含まないことで確認する。"""
    for i in range(300):
        key = f"orgsuffix-{i}".encode()
        out = Surrogate(kind="org", key=key).apply(value, E("ORG", len(value)), SurrogateCtx(key))
        assert core not in out, (i, value, out)


@pytest.mark.parametrize("value", ["ジョン・スミス", "ジョンスミス"])
def test_katakana_foreign_fakes_never_reuse_the_original_given_name_or_surname(value):
    """最終レビュー推奨 2: カタカナの外国人名は avoid_parts が名前全体しか返さず、元の名・姓を再利用した偽名
    （「ジョン・ブラウン」等）が通っていた。鍵を 300 通り変えて「ジョン」「スミス」を一度も含まないこと。"""
    for i in range(300):
        key = f"kata-{i}".encode()
        out = Surrogate(kind="person", key=key).apply(value, E("PERSON", len(value)), SurrogateCtx(key))
        assert "ジョン" not in out and "スミス" not in out, (i, out)


@pytest.mark.parametrize("value, parts", [
    ("スズキ タロウ", ("スズキ", "タロウ")), ("やまだ はなこ", ("やまだ", "はなこ")), ("Taro Suzuki", ("taro", "suzuki")),
])
def test_separated_full_names_never_reuse_original_parts(value, parts):
    """最終レビューの残件1: 区切り付きの日本語順カナ氏名・平仮名氏名・ローマ字氏名は default_normalize が
    区切りを消してから avoid_parts が呼ばれるため名前全体しか避けられず、生成器が元の姓や名を再利用していた
    （実測: 「スズキ タロウ」35/300、「やまだ はなこ」43/300、「Taro Suzuki」33/300）。鍵を 300 通り変えて
    元の姓・名を一度も含まないこと。"""
    for i in range(300):
        out = Surrogate(kind="person", key=b"k%d" % i).apply(value, E("PERSON", len(value)), Ctx())
        low = out.lower()
        assert not any(p in low for p in parts), (i, out)


def test_linked_kanji_name_with_nakaguro_separator_never_reuses_original_parts():
    """Task 0 レビュー Important: 中黒（「・」）区切りの漢字フルネームは姓リンク経路
    （Surrogate._ok_linked / _apply_person_linked）に入るが、そこは avoid_parts() を経由せず自前で
    split_person(norm.replace(" ", "")) を呼んで split[1] をそのまま avoid にしていた。「山田・太郎」では
    split[1] が「・太郎」（先頭の「・」付き）になり、_draw_linked が作る候補は区切り無しで連結される
    （sep は半角/全角スペースの有無でしか判定しないため）ので「・太郎」が候補に一致せず、名「太郎」が
    再利用されていた（実測 3/300: 安田太郎・太田太郎・大久保太郎）。鍵を 300 通り変えて
    元の姓「山田」・名「太郎」を一度も含まないこと。"""
    for i in range(300):
        out = Surrogate(kind="person", key=b"k%d" % i).apply("山田・太郎", E("PERSON", 5), Ctx())
        assert "太郎" not in out and "山田" not in out, (i, out)


def test_surrogate_uses_head_for_unresolved_org(tmp_path):
    """計画 B: 規則で決まらない ORG（種別語が無い「青葉」）は category_head があればそれで
    カテゴリを決める。min_margin を渡さなければヘッドの推奨値（HeadCategorizer.min_margin）を使う。

    値はカタカナの「アオバ」ではなく漢字の「青葉」にする: gen_org は script=="kata" のとき
    finance/manufacturing 等の業種語を付けない特別分岐があり（カタカナの社名は
    ORG_KATA_PARTS だけで作る）、カテゴリの効果を業種語の有無で見分けられないため。
    """
    pytest.importorskip("numpy")                # ヘッドの書き出し（_write_head）に使う。minimal profile では skip
    pytest.importorskip("safetensors.numpy")
    from privacy_masker.category_head import HeadCategorizer

    from .test_category_head import _vec, _write_head

    _write_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    op = Surrogate(kind="org", key=b"k", category_head=h)
    assert op._resolver.min_margin == {"org": 0.2}
    e = Entity("ORG", 0, 2, 1.0, "t", vector=_vec(0, 1, 0, 0))  # 規則では決まらない「青葉」→ ヘッドが finance
    out = op.apply("青葉", e, SurrogateCtx())
    assert out.endswith(tuple(jp.ORG_INDUSTRY_WORDS["finance"])) and "青葉" not in out
    e2 = Entity("ORG", 0, 2, 1.0, "t", vector=None)  # ベクトル無し → 規則だけ（service）
    op.apply("青葉", e2, SurrogateCtx())  # 例外にならないことだけを見る


class _FakeHead:
    """HeadCategorizer と同じ形（呼び出し + kind ごとの推奨 min_margin）のフェイク。numpy / safetensors 不要。"""

    def __init__(self, category, margin, min_margin):
        self.category, self.margin, self.min_margin = category, margin, min_margin

    def __call__(self, kind, vector):
        return self.category, self.margin


def test_surrogate_mid_margin_below_head_recommendation_falls_back_to_other():
    # margin 0.3 < ヘッドの推奨 0.55 → 規則だけのとき（service）と同じ出力。0.6 なら finance になる
    e = Entity("ORG", 0, 2, 1.0, "t", vector=b"\x00\x00")
    rules_only = Surrogate(kind="org", key=b"k").apply("青葉", Entity("ORG", 0, 2, 1.0, "t"), SurrogateCtx())
    mid = Surrogate(kind="org", key=b"k", category_head=_FakeHead("finance", 0.3, {"org": 0.55}))
    assert mid._resolver.min_margin == {"org": 0.55}
    assert mid.apply("青葉", e, SurrogateCtx()) == rules_only
    high = Surrogate(kind="org", key=b"k", category_head=_FakeHead("finance", 0.6, {"org": 0.55}))
    out = high.apply("青葉", e, SurrogateCtx())
    assert out != rules_only and out.endswith(tuple(jp.ORG_INDUSTRY_WORDS["finance"]))


def test_surrogate_category_head_must_be_callable():
    with pytest.raises(ConfigurationError, match="callable"):
        Surrogate(kind="org", key=b"k", category_head="not-callable")


def test_surrogate_explicit_min_margin_overrides_head_recommendation(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors.numpy")
    from privacy_masker.category_head import HeadCategorizer

    from .test_category_head import _write_head

    _write_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    op = Surrogate(kind="org", key=b"k", category_head=h, min_margin=0.9)
    assert op._resolver.min_margin == 0.9


# ---- 最終修正波（ブランチ全体レビュー + Task 5 レビュー）----

def _assert_never_contains(kind, entity_type, value, parts, label, n=400):
    """hmac・scope を n 通り変え、本文なし・本文ありの両方で、偽名が parts のどれも含まず、
    フォールバックトークンにも退避しないこと（退避したら緩めず報告する）。"""
    prefix = f"<{entity_type}_"
    op = Surrogate(kind=kind, key=KEY)
    for i in range(n):
        for text in (None, f"{value}について話した。"):
            out = op.apply(value, E(entity_type, len(value)), SurrogateCtx(KEY, scope_id=f"{label}-{i}", text=text))
            for part in parts:
                assert part not in out, (i, value, out, text)
            assert not (out.startswith(prefix) and out.endswith(">")), (i, value, out, text)


@pytest.mark.parametrize("value, canon", [("ソ連邦", "ロシア"), ("大韓帝国", "韓国"), ("米合衆国", "アメリカ")])
def test_location_fakes_never_reuse_the_canonical_name_of_an_alias_with_a_country_suffix(value, canon):
    """最終修正波 A1: 別名に 連邦・帝国・合衆国・共和国 が付いて country になった名前は COUNTRY_CANONICAL にも
    COUNTRIES にも完全一致しないため、同じ国の標準名が偽名に出ていた（400 scope 中: ソ連邦→ロシア 6、
    大韓帝国→韓国 7、米合衆国→アメリカ 6。本文なし・本文ありで同じ）。"""
    _assert_never_contains("location", "LOCATION", value, (canon,), "alias-suffix")


@pytest.mark.parametrize("value, core", [("営業委員会", "営業"), ("総務委員会", "総務")])
def test_department_committee_fakes_never_reuse_the_function_word(value, core):
    """最終修正波 A2: department の `^[^株合有]{1,8}(…|委員会)$` は文字列先頭から一致するため core_split が
    コアを切り出せず、avoid_parts が名前全体しか避けられなかった。department の偽名生成器（DEPT_FUNCTIONS ×
    DEPT_UNITS）が元の機能名を再利用していた（400 scope 中: 営業委員会 46、総務委員会 23）。"""
    _assert_never_contains("org", "ORG", value, (core,), "dept-committee")


@pytest.mark.parametrize("value, word", [("東青葉", "青葉"), ("新若葉", "若葉"), ("青葉台", "青葉")])
def test_location_fakes_never_reuse_fake_vocabulary_words_without_a_suffix(value, word):
    """最終修正波 A3: 接尾語の無い地名に埋もれた生成器の語彙語（TOWNS）は location の語彙の行（_vocab_parts）
    だけが避ける（基点では 400 scope 中 東青葉 13・新若葉 17・青葉台 14 が漏れ、HEAD では 0 だったが、
    この行を消しても落ちるテストが無かった）。"""
    _assert_never_contains("location", "LOCATION", value, (word,), "loc-vocab")


@pytest.mark.parametrize("value, word", [("営業部", "営業"), ("営業二課", "営業"), ("人事部", "人事"), ("営業推進委員会", "営業")])
def test_department_fakes_never_reuse_the_original_function_word(value, word):
    """最終修正波（department の機能名）: 部・課・室・チーム・係で終わる名前は avoid_parts が名前全体しか避けられず、
    department の偽名生成器が元の機能名を再利用していた（400 scope 中、基点から: 営業部 24・営業二課 38・人事部 36。
    この枝で私的な委員会が department に移り 営業推進委員会 0 → 42）。"""
    _assert_never_contains("org", "ORG", value, (word,), "dept-function")

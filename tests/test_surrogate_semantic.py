# tests/test_surrogate_semantic.py
"""カテゴリ判定・表層特徴（privacy_masker/operators/semantic.py）。値そのものが Features に入らないことも見る。"""
import dataclasses

import pytest

from privacy_masker.data import ja_pools as jp
from privacy_masker.normalize import default_normalize as norm
from privacy_masker.operators import semantic as sm


@pytest.mark.parametrize("value, expected", [
    ("株式会社青葉電機", "manufacturing"), ("アオバシステムズ株式会社", "it"), ("西八雲銀行", "finance"),
    ("株式会社大和商事", "service"), ("丸昌建設株式会社", "construction"), ("北斗運輸", "transport"),
    ("医療法人伊織会", "medical_corp"), ("学校法人若葉学園", "school_corp"), ("環境省", "government"),
    ("日本医療情報協会", "association"), ("東雲新聞社", "media"), ("みどりの会", "association"), ("民主党", "party"),
    ("営業二課", "department"), ("品質保証部", "department"), ("株式会社アオバ", "service"), ("アオバ", "service"),
])
def test_org_rules(value, expected):
    assert sm.RuleCategorizer.categorize("org", norm(value)) == expected


@pytest.mark.parametrize("name,expected", [
    # 公的機関の語は業種語より先（航空自衛隊・交通局・航空局）
    ("青葉航空自衛隊", "government"), ("若葉市交通局", "government"), ("汐見航空局", "government"),
    ("青葉放送局", "media"), ("若葉郵便局", "service"), ("汐見支局", "service"),           # 局 の除外
    ("青葉開発機構", "government"), ("若葉将軍", "service"),                                 # 機構 → government、将軍 は除外
    # 委員会: 公的な修飾があるときだけ government、それ以外は department。
    # 札幌市環境審議委員会 は「環境審議委員会$」の語彙一致ではなく、都道府県/市区町村の文字を含む
    # 汎用の接頭パターン（^.{1,6}(都|道|府|県|市|区|町|村)[^ ]{0,10}委員会$）で government になる
    ("若葉県教育委員会", "government"), ("選挙管理委員会", "government"), ("公正取引委員会", "government"), ("札幌市環境審議委員会", "government"),
    ("青葉祭実行委員会", "department"), ("党中央委員会", "government"),                      # 中央委員会 は公的扱い（国家・中央）
    # の会 → association、商事 → service、ホーム・グループ → service
    ("みなとの会", "association"), ("若葉商事", "service"), ("青葉ホーム", "service"), ("汐見グループ", "service"),
    # 電気: 鉄道系は transport、裸の 電気 は service。電機 は manufacturing のまま
    ("青葉電気鉄道", "transport"), ("若葉電気軌道", "transport"), ("汐見電鉄", "transport"), ("青葉電気", "service"), ("青葉電機", "manufacturing"),
    # 通信: 通信社・通信 → media、電気通信・テレコム → it
    ("青葉通信社", "media"), ("若葉通信", "media"), ("汐見電気通信", "it"), ("みなとテレコム", "it"),
    # 球団名 → media（ソフト より先）
    ("若葉ホークス", "media"), ("ソフトバンクホークス", "media"), ("青葉ソフト", "it"), ("汐見ジャイアンツ", "media"),
])
def test_rule_quirks_v2_org(name, expected):
    assert sm.RuleCategorizer.categorize("org", name) == expected


def test_type_word_span_follows_new_table_order():
    # 表の順が変わっても type_word_span は categorize と同じ語を返す
    for name in ("青葉航空自衛隊", "若葉電気鉄道", "ソフトバンクホークス", "青葉通信社"):
        sp = sm.RuleCategorizer.type_word_span("org", name)
        assert sp is not None and name[sp[0]:sp[1]] in ("自衛隊", "電気鉄道", "ホークス", "通信社")


@pytest.mark.parametrize("name,expected", [
    # 局: テレビ局・ラジオ局は media（government の局からは除外）。薬局・事務局・編集局は
    # government からも media からも外れて service に落ちる
    ("若葉テレビ局", "media"), ("若葉ラジオ局", "media"), ("青葉薬局", "service"), ("若葉祭事務局", "service"),
    # 汐見新聞編集局: 「新聞」が末尾ではない（新聞+編集局）ので media の $ 語彙には当たらず、
    # government も 編集局 を除外するので service に落ちる
    ("汐見新聞編集局", "service"),
    # 軍: 陸軍・海軍・空軍・国防軍・解放軍・連合軍だけが government。巨人軍は media の球団語、
    # 救世軍・将軍は裸の「軍」を government から外したことでどちらも service
    ("青葉陸軍", "government"), ("青葉巨人軍", "media"), ("救世軍", "service"), ("若葉将軍", "service"),
    # 委員会: 都道府県/市区町村の文字を含む汎用パターンで government（_CITY_SET に頼らない）
    ("札幌市子ども委員会", "government"), ("北海道観光振興委員会", "government"),
])
def test_rule_quirks_v2_org_round1_fixes(name, expected):
    assert sm.RuleCategorizer.categorize("org", name) == expected


@pytest.mark.parametrize("name,expected", [
    # 球団 → media（末尾一致）
    ("若葉球団", "media"),
    # 公的な委員会の修飾語に 欧州|連邦|国連|証券取引|運輸安全 を追加。修飾語は委員会に直接隣接するものだけ
    # （fix round 2 で隣接を復元。「連邦」は連邦通信・連邦準備・連邦取引・連邦選挙・連邦捜査の複合語に限定）
    ("欧州委員会", "government"), ("連邦通信委員会", "government"), ("証券取引委員会", "government"), ("運輸安全委員会", "government"),
    # fix round 1 で自治体接頭パターンの文字クラスに 省|州 を足したが、fix round 2 のレビューで
    # 「社内省エネ推進委員会」等の誤発火と地名語幹の漏洩が見つかり撤回した（代替なし）。
    # 四川省発展委員会・ハルカゼ州選挙委員会 は department に戻る（下の test_rule_quirks_v2_org_round3_fixes 参照）
    # 協議会・評議会は従来どおり government（「議会」が government の語彙に既にあるための副作用。今回は変えない）
    ("青葉協議会", "government"),
])
def test_rule_quirks_v2_org_round2_fixes(name, expected):
    assert sm.RuleCategorizer.categorize("org", name) == expected


@pytest.mark.parametrize("name,expected", [
    # N2: 委員会の修飾語は直接隣接するものだけ（fix round 1 の「間に最大 6 字まで許す」を撤回）。
    # 私的・内部の委員会は department のまま
    ("労働安全衛生委員会", "department"), ("監査等委員会", "department"), ("中央執行委員会", "department"),
    ("人事評価委員会", "department"), ("欧州事業委員会", "department"),
    # N2 の確認（隣接のまま government であり続ける）
    ("連邦通信委員会", "government"), ("欧州委員会", "government"), ("国家公安委員会", "government"), ("党中央委員会", "government"),
    # N3: 自治体接頭パターンから 省|州 を撤回（代替なし）。外国の省・州の委員会は department のまま
    ("四川省発展委員会", "department"), ("ハルカゼ州選挙委員会", "department"),
    ("社内省エネ推進委員会", "department"), ("九州支社安全委員会", "department"),
])
def test_rule_quirks_v2_org_round3_fixes(name, expected):
    assert sm.RuleCategorizer.categorize("org", name) == expected


@pytest.mark.parametrize("value, expected", [
    ("青葉総合病院", "medical"), ("汐見クリニック", "medical"), ("市立若草小学校", "education"), ("北斗大学", "education"),
    ("汐見駅", "transport"), ("新星ホテル", "commerce"), ("八雲市役所", "public"), ("汐見公園", "public"),
    ("グランメゾン白樺", "building"), ("若葉タワー", "building"), ("富士見美術館", "culture"), ("扇の湯温泉", "culture"),
    ("汐見センター", "other"),
])
def test_facility_rules(value, expected):
    assert sm.RuleCategorizer.categorize("facility", norm(value)) == expected


@pytest.mark.parametrize("value, expected", [
    ("フランス", "country"), ("日本", "country"), ("大分県", "prefecture"), ("東京都", "prefecture"),
    ("札幌市", "city"), ("新宿区", "city"), ("東川町", "city"), ("西新宿2丁目", "town"), ("本町3番", "town"),
    ("汐見駅前", "district"), ("パリ", "foreign"), ("ニューヨーク", "foreign"), ("八雲山", "nature"), ("早瀬川", "nature"),
    ("北関東", "other"),
])
def test_location_rules(value, expected):
    assert sm.RuleCategorizer.categorize("location", norm(value)) == expected


@pytest.mark.parametrize("name,expected", [
    # 国名リストの拡充（カタカナ・漢字略称・構成国・歴史的な国）
    ("イラク", "country"), ("ウクライナ", "country"), ("スコットランド", "country"), ("米国", "country"), ("英国", "country"),
    ("ソ連", "country"), ("ギリシア", "country"), ("ギリシャ", "country"), ("日本", "country"),
    # 〜地区・選挙区 → district（city より先）
    ("若葉地区", "district"), ("青葉1区", "district"), ("汐見２区", "district"), ("みなと駅前", "district"),
    # カタカナ + 市/町/州/省、漢字の外国都市 → foreign（city より先）
    ("ミナト市", "foreign"), ("アオバ町", "foreign"), ("ハルカゼ州", "foreign"), ("上海", "foreign"), ("北京", "foreign"), ("ソウル", "foreign"),
    # 日本の市区町村は従来どおり
    ("札幌市", "city"), ("若葉町", "city"), ("青葉区", "city"), ("汐見村", "city"),
    # nature: 本体 2 字以上、市区町村・町域に無いときだけ
    ("八雲山", "nature"), ("早瀬川", "nature"), ("山", "other"), ("川", "other"), ("深川", "other"),     # 本体 1 字の「深川」は nature にしない（pilot で town と判定された癖）
])
def test_rule_quirks_v2_location(name, expected):
    assert sm.RuleCategorizer.categorize("location", name) == expected


def test_country_lists_are_disjoint_and_country_pool_for_generation_is_unchanged():
    assert not set(jp.COUNTRIES) & set(jp.COUNTRIES_EXTRA) and not set(jp.COUNTRIES) & set(jp.COUNTRY_ALIASES)
    assert len(jp.COUNTRIES) == 45                       # 生成プールは据え置き（v0.8 互換）
    assert len(jp.COUNTRIES_EXTRA) >= 150 and len(jp.FOREIGN_CITIES_KANJI) >= 15
    assert len(set(jp.COUNTRIES_EXTRA)) == len(jp.COUNTRIES_EXTRA)   # 重複なし


@pytest.mark.parametrize("name,expected", [
    # 共和国|帝国|合衆国|連邦 で終わる架空の国名 → country。王国・公国は足さない（検証で foreign 判定が多かった）
    ("若葉共和国", "country"), ("青葉帝国", "country"), ("ミナト合衆国", "country"), ("汐見連邦", "country"), ("若葉王国", "other"),
    # 国名の単漢字略称（COUNTRY_ALIASES の追加分）
    ("米", "country"), ("日", "country"), ("英", "country"), ("日本国", "country"), ("満州国", "country"),
])
def test_rule_quirks_v2_location_round2_fixes(name, expected):
    assert sm.RuleCategorizer.categorize("location", name) == expected


def test_categories_are_closed_and_other_is_valid():
    for kind, cats in sm.CATEGORIES.items():
        assert sm.OTHER[kind] in cats
        assert len(cats) == len(set(cats))


def test_person_features():
    f = sm.extract_features("person", "山田 太郎", norm("山田 太郎"), "person")
    assert (f.structure, f.script, f.separator, f.gender_hint) == ("full", "kanji", " ", "m")
    f = sm.extract_features("person", "佐藤花子", norm("佐藤花子"), "person")
    assert (f.structure, f.gender_hint) == ("full", "f")
    f = sm.extract_features("person", "佐藤", norm("佐藤"), "person")
    assert f.structure == "sei" and f.gender_hint == "unknown"
    f = sm.extract_features("person", "ミゲル・サントス", norm("ミゲル・サントス"), "person")
    assert (f.script, f.kata_foreign, f.structure, f.separator) == ("kata", True, "full", "・")
    f = sm.extract_features("person", "スズキ タロウ", norm("スズキ タロウ"), "person")
    assert (f.structure, f.separator, f.gender_hint) == ("full", " ", "m")
    f = sm.extract_features("person", "やまだ はなこ", norm("やまだ はなこ"), "person")
    assert (f.structure, f.separator, f.gender_hint) == ("full", " ", "f")
    f = sm.extract_features("person", "スズキ・タロウ", norm("スズキ・タロウ"), "person")
    assert (f.structure, f.separator, f.gender_hint) == ("full", "・", "m")
    f = sm.extract_features("person", "アッコ", norm("アッコ"), "person")
    assert (f.script, f.structure) == ("kata", "mono")
    f = sm.extract_features("person", "みやさか", norm("みやさか"), "person")
    assert (f.script, f.structure) == ("hira", "mono")
    f = sm.extract_features("person", "John Smith", norm("John Smith"), "person")
    assert (f.script, f.structure) == ("romaji", "full")
    f = sm.extract_features("person", "ｽｽﾞｷ ｷﾞﾞ", norm("ｽｽﾞｷ ｷﾞﾞ"), "person")
    assert f.script == "halfwidth"


def test_org_features_keep_legal_form_position_and_script():
    f = sm.extract_features("org", "株式会社青葉電機", norm("株式会社青葉電機"), "manufacturing")
    assert (f.legal_form, f.legal_form_pos, f.script) == ("株式会社", "pre", "kanji")
    f = sm.extract_features("org", "アオバシステムズ株式会社", norm("アオバシステムズ株式会社"), "it")
    assert (f.legal_form, f.legal_form_pos, f.script) == ("株式会社", "post", "kata")
    f = sm.extract_features("org", "JPX", norm("JPX"), "service")
    assert (f.legal_form, f.script) == (None, "latin_abbr")
    f = sm.extract_features("org", "医療法人伊織会", norm("医療法人伊織会"), "medical_corp")
    assert f.legal_form == "医療法人"


def test_location_and_address_features():
    f = sm.extract_features("location", "大分県", norm("大分県"), "prefecture")
    assert (f.granularity, f.region_block) == ("prefecture", "九州・沖縄")
    f = sm.extract_features("location", "札幌市", norm("札幌市"), "city")
    assert (f.granularity, f.region_block) == ("city", "北海道")
    f = sm.extract_features("location", "青凪市", norm("青凪市"), "city")        # プールに無い市 → ブロック不明
    assert (f.granularity, f.region_block) == ("city", None)
    f = sm.extract_features("address", "東京都新宿区西新宿1-2-3 サンハイツ301", norm("東京都新宿区西新宿1-2-3 サンハイツ301"), "address", keep="city")
    assert (f.region_block, f.has_building, f.chome_style, f.keep) == ("関東", True, False, "city")
    f = sm.extract_features("address", "神奈川県横浜市中区太田町５丁目８番地", norm("神奈川県横浜市中区太田町５丁目８番地"), "address")
    assert (f.region_block, f.chome_style, f.fullwidth_digits) == ("関東", True, True)


def test_features_never_contain_original_strings():
    """Features の文字列フィールドは閉じた語彙だけ（元の値の断片が漏れない）。"""
    allowed = set(sm.LEGAL_FORMS) | set(sm.PERSON_SCRIPTS) | set(sm.STRUCTURES) | set(sm.GENDER_HINTS) | set(jp.REGION_BLOCKS) \
        | set(jp.SCHOOL_PREFIXES) | {"", " ", "・", "short", "medium", "long", "pre", "post", "city", None} \
        | {c for cats in sm.CATEGORIES.values() for c in cats} | {"country", "prefecture", "city", "town", "district", "foreign", "nature", "other",
                                                                  "pref", "banchi", "building", "person", "org", "facility", "location", "address"}
    for kind, value, cat in [("person", "勘解由小路 麗華", "person"), ("org", "株式会社大江戸精密", "manufacturing"),
                             ("facility", "県立神楽坂高等学校", "education"), ("location", "和歌山県", "prefecture"),
                             ("address", "大阪府堺市北区百舌鳥町9-9-9", "address")]:
        f = sm.extract_features(kind, value, norm(value), cat)
        for fld in dataclasses.fields(f):
            v = getattr(f, fld.name)
            if isinstance(v, str) or v is None:
                assert v in allowed, (fld.name, v)


def test_avoid_parts_and_region_block_and_halfwidth():
    assert sm.avoid_parts("person", norm("山田太郎")) == ("山田", "太郎")
    assert sm.avoid_parts("person", norm("佐藤")) == ("佐藤",)
    assert sm.avoid_parts("org", norm("株式会社青葉電機")) == ("青葉",)
    assert sm.avoid_parts("facility", norm("汐見クリニック")) == ("汐見",)
    assert sm.avoid_parts("location", norm("札幌市")) == ("札幌市", "札幌")
    assert sm.avoid_parts("location", norm("東京都")) == ("東京都", "東京")
    assert sm.avoid_parts("location", norm("パリ")) == ("パリ",)
    assert sm.avoid_parts("location", norm("日本")) == ("日本",)
    assert sm.avoid_parts("location", norm("西新宿2丁目")) == ("西新宿2丁目",)
    assert sm.region_block_of("大分県") == "九州・沖縄" and sm.region_block_of("札幌市中央区") == "北海道" and sm.region_block_of("青凪市") is None
    assert sm.to_halfwidth_kana("サトウ タロウ") == "ｻﾄｳ ﾀﾛｳ" and sm.to_halfwidth_kana("スズキ") == "ｽｽﾞｷ"


def test_avoid_parts_location_covers_country_aliases_state_suffix_and_district_stems():
    """レビュー Important I1・I2・Minor M1 の安全性回復。avoid_parts は増やす方向にだけ変える
    （「パリ」はこれまでどおり全体 1 件のみ）。"""
    # I1: 国名の別名・略称・前身国家は、同じ国の標準名（生成プール COUNTRIES の項目）も避ける
    assert sm.avoid_parts("location", norm("米国")) == ("米国", "アメリカ")
    # I2: カタカナ+州・省 は語幹（州・省を除いた部分）も避ける
    assert sm.avoid_parts("location", norm("ハルカゼ州")) == ("ハルカゼ州", "ハルカゼ")
    # M1: 地区・選挙区・自然地名の接尾語を外した語幹も避ける
    assert set(sm.avoid_parts("location", norm("若葉地区"))) >= {"若葉地区", "若葉地", "若葉"}
    assert "青葉" in sm.avoid_parts("location", norm("青葉1区"))
    assert "八雲" in sm.avoid_parts("location", norm("八雲山"))
    # 既存の挙動は変えない（増やす方向にだけ拡張する）
    assert sm.avoid_parts("location", norm("パリ")) == ("パリ",)


def test_country_canonical_values_are_in_countries_pool_and_keys_resolve_to_country():
    assert set(jp.COUNTRY_CANONICAL.values()) <= set(jp.COUNTRIES)
    for key in jp.COUNTRY_CANONICAL:
        assert sm.RuleCategorizer.categorize("location", key) == "country", key


def test_avoid_parts_location_avoids_any_countries_entry_contained_in_the_original():
    """fix round 2 N1: _LOC_COUNTRY_SUFFIX（共和国|帝国|合衆国|連邦）で country になった正式名は
    COUNTRY_CANONICAL に無いため、同じ国の短い名前（COUNTRIES の項目）がそのまま偽名に出ていた
    （実測 400 件中: ロシア連邦→ロシア 6、ドイツ連邦共和国→ドイツ 10、メキシコ合衆国→メキシコ 7、
    フランス共和国→フランス 18、ベトナム社会主義共和国→ベトナム 13、南アフリカ共和国→南アフリカ 7、
    ロシア帝国→ロシア 7）。COUNTRIES の項目を部分文字列に持つ国名（北アイルランド・インドネシア）も同様。
    n に含まれる COUNTRIES の項目（n 自身は除く）をすべて避けるようにした。"""
    assert "ロシア" in sm.avoid_parts("location", norm("ロシア連邦"))
    assert "インド" in sm.avoid_parts("location", norm("インドネシア"))
    assert "イギリス" in sm.avoid_parts("location", norm("大英帝国"))
    assert "ドイツ" in sm.avoid_parts("location", norm("ドイツ連邦共和国"))
    assert "アイルランド" in sm.avoid_parts("location", norm("北アイルランド"))
    # 既存の挙動は変えない
    assert sm.avoid_parts("location", norm("米国")) == ("米国", "アメリカ")
    assert sm.avoid_parts("location", norm("パリ")) == ("パリ",)


def test_avoid_parts_avoids_fake_core_vocabulary_words_embedded_in_the_extracted_core():
    """fix round 3: コーディネーターレビューで、Task 3 レビューの残課題（中央青葉人事委員会 → core 中央青葉、
    青葉 ∈ ORG_CORES が avoid されない）が、2 件の既存の parked finding（若葉県教育委員会 22/300・
    汐見町議会 15/300 が 若葉/汐見 を再利用）と同じ根本原因だと判明した: core_split が切り出すコアは
    種別語の前（または委員会の自治体接頭パターンのように文字列全体）をひとかたまりでしか扱わないため、
    その内部に生成器の固有部分の語彙（ORG_CORES・KATA_CORES は org/facility、TOWNS は location の
    district/nature）が部分文字列として埋もれていても個別には避けられなかった。avoid_parts が
    元の値に含まれる語彙語をすべて追加で avoid するようになったことを確認する。"""
    assert set(sm.avoid_parts("org", norm("中央青葉人事委員会"))) >= {"中央青葉", "中央", "青葉"}
    assert "若葉" in sm.avoid_parts("org", norm("若葉県教育委員会"))
    assert "汐見" in sm.avoid_parts("org", norm("汐見町議会"))
    assert set(sm.avoid_parts("facility", norm("北斗青葉記念館"))) >= {"北斗", "青葉"}
    # location: 既存の先頭要素は変えず、"中央" は重複せず 1 回だけ（TOWNS にも "中央" があるが dedup される）
    loc = sm.avoid_parts("location", norm("中央地区"))
    assert loc[:2] == ("中央地区", "中央地") and loc.count("中央") == 1
    # 語彙語を含まない名前は変化しない
    assert sm.avoid_parts("org", norm("株式会社ネットワークス")) == ("ネットワークス",)


@pytest.mark.parametrize("value, expected", [
    ("大阪府", "pref"), ("北海道", "pref"),                       # 都道府県のみ
    ("東京都新宿区", "city"), ("青凪市栄町", "city"),               # 市区町村止まり（末尾が市区町村、プール外の市も可）
    ("東京都新宿区西新宿", "town"),                                # 町名まで（末尾が市区町村ではない）
    ("青凪市栄町2丁目5番地", "banchi"),                            # 番地の数字あり
    ("サンモールビル別館302号室", "building"),                     # 建物語あり（市区町村を含まない）
])
def test_address_granularity_levels(value, expected):
    # Critical #1: 元の情報量（都道府県だけ・市区町村まで等）を超えて番地・建物を捏造しないための粒度判定
    f = sm.extract_features("address", value, norm(value), "address")
    assert f.granularity == expected


def test_avoid_parts_for_address_excludes_city_and_town_but_not_bare_prefecture():
    # 都道府県だけの入力は市区町村を含まないため avoid_parts は空（同一ブロックの別の都道府県は許容）
    assert sm.avoid_parts("address", norm("大阪府")) == ()
    assert sm.avoid_parts("address", norm("東京都新宿区")) == ("東京都新宿区",)
    assert sm.avoid_parts("address", norm("青凪市栄町2丁目5番地")) == ("青凪市", "栄町")
    assert sm.avoid_parts("address", norm("大阪府堺市北区中百舌鳥町2丁目39番1号")) == ("大阪府堺市北区", "中百舌鳥町")
    assert sm.avoid_parts("address", norm("サンモールビル別館302号室")) == ()   # 市区町村を含まない


def test_resolver_uses_rules_first_then_head_then_other():
    calls = []

    def head(kind, vector):
        calls.append((kind, vector))
        return ("medical", 0.9)

    r = sm.CategoryResolver(head=head, min_margin=0.15)
    assert r.resolve("facility", norm("青葉総合病院"), b"\x00") == "medical" and calls == []       # 規則で決まればヘッドを呼ばない
    assert r.resolve("facility", norm("アオバ"), b"\x00") == "medical" and calls == [("facility", b"\x00")]
    assert r.resolve("facility", norm("アオバ"), None) == "other"                                   # ベクトル無し → 退避
    weak = sm.CategoryResolver(head=lambda k, v: ("medical", 0.05))
    assert weak.resolve("facility", norm("アオバ"), b"\x00") == "other"                            # 余裕不足 → 退避
    bad = sm.CategoryResolver(head=lambda k, v: ("not-a-category", 0.9))
    assert bad.resolve("facility", norm("アオバ"), b"\x00") == "other"
    assert sm.CategoryResolver().resolve("person", norm("山田太郎"), None) == "person"


def test_resolver_per_kind_min_margin_mapping():
    # HeadCategorizer.min_margin のような kind → 数値の Mapping。margin < その kind の値なら退避カテゴリ、
    # Mapping に kind が無ければ既定の 0.15
    margin = {"v": 0.0}
    r = sm.CategoryResolver(head=lambda k, v: ({"facility": "medical", "org": "finance"}[k], margin["v"]),
                            min_margin={"facility": 0.5})
    margin["v"] = 0.4
    assert r.resolve("facility", norm("アオバ"), b"\x00") == "other"             # 0.4 < 0.5
    margin["v"] = 0.5
    assert r.resolve("facility", norm("アオバ"), b"\x00") == "medical"           # 境界は採用（>=）
    margin["v"] = 0.14
    assert r.resolve("org", norm("アオバ"), b"\x00") == "service"                 # org は Mapping に無い → 0.15
    margin["v"] = 0.15
    assert r.resolve("org", norm("アオバ"), b"\x00") == "finance"


@pytest.mark.parametrize("kind, value", [("person", "山田太郎"), ("address", "青凪市栄町2丁目5番地")])
def test_resolver_never_calls_head_for_person_or_address(kind, value):
    calls = []

    def head(k, v):
        calls.append(k)
        return ("medical", 1.0)

    assert sm.CategoryResolver(head=head, min_margin=0.0).resolve(kind, norm(value), b"\x00") == kind
    assert calls == []


def test_new_pools_are_appended_not_reordered():
    """既存タプルの先頭要素（ゴールデン値を支える）が変わっていないこと。"""
    assert jp.SURNAMES_KANJI[:3] == ("佐藤", "鈴木", "高橋") and jp.GIVEN_KANJI_M[0] == "太郎" and jp.ORG_CORES[0] == "曙"
    assert jp.PREF_CITIES[0] == ("北海道", "札幌市中央区") and jp.TOWNS[0] == "中央"
    assert len(jp.PREFECTURES) == 47 and sum(len(v) for v in jp.REGION_BLOCKS.values()) == 47
    assert set(jp.PREFECTURES) == {p for v in jp.REGION_BLOCKS.values() for p in v}
    assert len(jp.COUNTRIES) >= 40 and "日本" not in jp.COUNTRIES               # 日本は置換先に使わない（元が日本なら別の国へ）


def test_public_address_helpers_and_type_word_span():
    assert sm.address_city_and_town(norm("東京都新宿区西新宿1-2-3")) == ("東京都新宿区", "西新宿")
    assert sm.address_granularity(norm("大阪府")) == "pref"
    assert sm.HAS_BUILDING_RE.search("サンハイツ301")
    assert not hasattr(sm, "_address_city_and_town") and not hasattr(sm, "_HAS_BUILDING_RE")
    # 種別語のスパン（norm 上の位置）。avoid_parts が RuleCategorizer の私有属性に触らずに済む
    assert sm.RuleCategorizer.type_word_span("org", "青葉電機") == (2, 4)
    assert sm.RuleCategorizer.type_word_span("facility", "汐見クリニック") == (2, 7)
    assert sm.RuleCategorizer.type_word_span("org", "アオバ") is None
    assert sm.RuleCategorizer.type_word_span("location", "札幌市") is None
    assert sm.avoid_parts("org", norm("青葉電機")) == ("青葉",)             # 挙動は従来どおり


def test_gender_overrides_fix_unisex_names():
    # カイト / かいと / Kaito はプールの奇数番目（女性側）に置かれているが男性名。上書きで直す
    assert sm.GENDER_OVERRIDES == {"カイト": "m", "かいと": "m", "Kaito": "m"}
    for name in ("カイト", "かいと", "Kaito"):
        assert sm.given_gender(name) == "m" and sm.gender_hint(name) == "m"
    assert sm.given_gender("ハナコ") == "f" and sm.given_gender("タロウ") == "m" and sm.given_gender("謎") == "unknown"


def test_long_katakana_foreign_name_without_separator_is_full():
    f = sm.extract_features("person", "マイケルジョンソン", norm("マイケルジョンソン"), "person")
    assert (f.script, f.kata_foreign, f.structure, f.separator) == ("kata", True, "full", "")
    g = sm.extract_features("person", "アキ", norm("アキ"), "person")
    assert (g.structure, g.kata_foreign) == ("mono", False)


def test_features_has_no_place_prefix():
    assert "place_prefix" not in {fl.name for fl in dataclasses.fields(sm.Features)}
    f = sm.extract_features("facility", "汐見駅", norm("汐見駅"), "transport")
    assert f.region_block is None          # facility では使わない（設計 §6: 施設名は全部架空）


@pytest.mark.parametrize("value, expected", [
    ("三重県四日市市諏訪町1-2", ("三重県四日市市", "諏訪町")),
    ("四日市市諏訪町1-2", ("四日市市", "諏訪町")),                       # 都道府県なし
    ("広島県廿日市市宮島町1", ("広島県廿日市市", "宮島町")),
    ("新潟県十日町市本町1", ("新潟県十日町市", "本町")),                 # 途中が「町」
    ("長崎県大村市玖島1", ("長崎県大村市", "玖島")),                     # 途中が「村」
    ("東京都新宿区西新宿1-2", ("東京都新宿区", "西新宿")),
    # 町名が「市」「区」で始まるときは延長しない（最終レビュー C2: 旧ループは「新宿区市」まで取り込んでいた）
    ("東京都新宿区市谷加賀町1-1", ("東京都新宿区", "市谷加賀町")),
    ("千葉県千葉市中央区市場町1", ("千葉県千葉市中央区", "市場町")),
    ("岐阜県多治見市市之倉町2", ("岐阜県多治見市", "市之倉町")),
    ("東京都北区区役所前1", ("東京都北区", "区役所前")),
])
def test_city_with_inner_shi_is_extended_only_by_vocabulary(value, expected):
    """「四日市市」のように市区町村名の途中に 市/区/町/村 を含むと、非貪欲一致が手前で切れて末尾の「市」が rest 側に
    残っていた（keep="city" で「三重県四日市」+ 偽の町名となり「市」が 1 字落ちる。町名の漏れではない）。
    延長は該当する市の語彙（MUNICIPALITIES_WITH_INNER_SHI）で 1 字だけ行い、「市谷加賀町」のように町名が
    市/区で始まる場合は延長しない。"""
    assert sm.address_city_and_town(norm(value)) == expected
    assert sm.is_municipality(expected[0])


def test_municipalities_with_inner_shi_vocabulary():
    assert sm.MUNICIPALITIES_WITH_INNER_SHI == frozenset(
        {"四日市市", "廿日市市", "野々市市", "十日町市", "大町市", "大村市", "羽村市", "田村市"})


def test_city_extension_requires_a_boundary_before_the_vocabulary():
    """最終レビューの残件2: MUNICIPALITIES_WITH_INNER_SHI による延長は境界を見ずに語彙の文字列末尾一致
    だけで判定していたため、「野田村市場」のように語彙「田村市」が別の町名の内部にたまたま一致すると
    誤って1字取り込んでいた。語彙の直前が文字列の先頭、または 都/道/府/県/郡 の1字のときだけ延長する。"""
    assert sm.address_city_and_town(norm("岩手県九戸郡野田村市場1")) == ("岩手県九戸郡野田村", "市場")
    assert sm.address_city_and_town(norm("福島県石川郡平田村市野1")) == ("福島県石川郡平田村", "市野")
    assert sm.address_city_and_town(norm("三重県四日市市諏訪町1-2")) == ("三重県四日市市", "諏訪町")   # 既存ケースは不変
    assert sm.address_city_and_town(norm("福島県田村市船引町1")) == ("福島県田村市", "船引町")


def test_chome_kanji_feature():
    f = sm.extract_features("address", "東京都新宿区西新宿一丁目2-3", norm("東京都新宿区西新宿一丁目2-3"), "address")
    assert f.chome_style and f.chome_kanji
    g = sm.extract_features("location", "西新宿2丁目", norm("西新宿2丁目"), "town")
    assert g.chome_style and not g.chome_kanji


def test_hokkaido_extra_cities_are_known_to_semantic_only():
    from privacy_masker.data import ja_pools as jp

    assert jp.PREF_CITIES_EXTRA == (("北海道", "函館市"), ("北海道", "小樽市"), ("北海道", "帯広市"), ("北海道", "釧路市"), ("北海道", "北見市"))
    assert sm.region_block_of("函館市") == "北海道" and sm.RuleCategorizer.categorize("location", "函館市") == "city"
    assert jp.PREF_CITIES[-1] != ("北海道", "北見市")       # 既存タプルは不変


def test_avoid_parts_core_is_the_type_word_at_the_end_not_the_first_type_word():
    """最終レビュー Critical C1: avoid_parts が type_word_span（表の順で最初に一致した語。末尾かどうかは問わない）
    を使ったため、途中に別の種別語を含む名前（「ネットワークス商船」の「ネットワーク」、「タワー美術館」の「タワー」）で
    コア語を切り出せず名前全体を返し、コア語（「ネットワークス」「タワー」）が偽名に混入しうる状態になっていた。
    avoid_parts は core_split（表の順に、末尾で一致し開始位置が 2 以上の最初の語）を使う。"""
    assert sm.avoid_parts("org", norm("ネットワークス商船")) == ("ネットワークス",)
    assert sm.avoid_parts("org", norm("株式会社ネットワークス商船")) == ("ネットワークス",)
    assert sm.avoid_parts("facility", norm("タワー美術館")) == ("タワー",)
    # 従来のケースは変わらない
    assert sm.avoid_parts("org", norm("株式会社青葉電機")) == ("青葉",)
    assert sm.avoid_parts("org", norm("青葉電機")) == ("青葉",)
    assert sm.avoid_parts("facility", norm("汐見クリニック")) == ("汐見",)
    assert sm.avoid_parts("org", norm("アオバ")) == ("アオバ",)
    assert sm.RuleCategorizer.core_split("org", "ネットワークス商船") == 7
    assert sm.RuleCategorizer.core_split("facility", "タワー美術館") == 3
    assert sm.RuleCategorizer.core_split("org", "青葉電機") == 2
    assert sm.RuleCategorizer.core_split("org", "電機") is None          # 開始位置 2 未満
    assert sm.RuleCategorizer.core_split("org", "アオバ") is None
    assert sm.RuleCategorizer.core_split("location", "札幌市") is None
    # type_word_span の意味（表の順で最初に一致した語。categorize と同じ語）は変えない（計画 B が依存）。
    # ORG の表の順が変わり（RuleCategorizer v2）、「ネットワークス商船」は transport（商船）が it（ネットワーク）より
    # 先に当たるようになったため、最初に一致する語そのものが変わった（categorize も transport を返す）
    assert sm.RuleCategorizer.type_word_span("org", "ネットワークス商船") == (7, 9)
    assert sm.RuleCategorizer.type_word_span("facility", "タワー美術館") == (0, 3)
    # type_word_span と core_split の違いを示す別例（表の途中に当たる語と末尾の語が異なる）。
    # 「青葉交通データ」は transport の「交通」(2, 4) が categorize の結果になる一方、
    # コア語の切り出しは末尾の「データ」(it) の 4（コアは「青葉交通」）
    assert sm.RuleCategorizer.type_word_span("org", "青葉交通データ") == (2, 4)
    assert sm.RuleCategorizer.core_split("org", "青葉交通データ") == 4
    # fix round 3: コア「青葉交通」自体に加え、内部に埋もれた語彙語「青葉」（ORG_CORES）も avoid される
    assert sm.avoid_parts("org", norm("青葉交通データ")) == ("青葉交通", "青葉")


@pytest.mark.parametrize("name,expected_start", [
    # _ORG_WORDS のどの行にも当たらない語尾（商事・ホーム・電気・サービス・グループ）は、
    # カテゴリとしては OTHER=service に落ちるが、core_split は _ORG_CORE_SUFFIXES で
    # コアを切り出せる（切り出せないと avoid_parts が名前全体しか避けられず、コアが偽名に漏れる）
    ("若葉商事", 2), ("青葉ホーム", 2), ("東雲電気", 2), ("若葉サービス", 2), ("北斗グループ", 2),
])
def test_core_split_falls_back_to_category_neutral_org_suffixes(name, expected_start):
    assert sm.RuleCategorizer.categorize("org", name) == "service"
    assert sm.RuleCategorizer.core_split("org", name) == expected_start
    core = name[:expected_start]
    assert sm.avoid_parts("org", norm(name)) == (core,)


def _old_avoid_parts_org_facility(kind, norm_value):
    """v0.9（意味を保つ Surrogate の導入時点）の avoid_parts の org / facility 分岐をそのまま書き直した参照実装。
    レビュー対応（商事・ホーム・電気等の core_split 漏れ修正）で org は表の不一致時に
    `_ORG_CORE_SUFFIXES` も試すようになったため、この参照実装も同じ条件で追従する。"""
    n = norm_value.replace(" ", "")
    _form, _pos, rest = sm.split_legal_form(n)
    core = rest
    for _cat, rx in (sm.RuleCategorizer._ORG_WORDS if kind == "org" else sm.RuleCategorizer._FAC_WORDS):
        m = rx.search(core)
        if m and m.end() == len(core) and m.start() >= 2:
            core = core[: m.start()]
            break
    else:
        if kind == "org":
            m = sm.RuleCategorizer._ORG_CORE_SUFFIXES.search(core)
            if m and m.end() == len(core) and m.start() >= 2:
                core = core[: m.start()]
    if len(core) >= 2 and core != n:
        return (core,)
    return (n,) if len(n) >= 2 else ()


def test_avoid_parts_org_facility_equals_the_old_loop():
    """C1 の修正が従来の avoid_parts（種別語の表を順に見て末尾一致を探すループ）と同値であること。
    生成器が出しうる全候補から決定的に 20,000 件を抜き出し、途中と末尾に種別語を持つ手書きの名前も加えて比べる
    （全候補 654,258 件 × 2 種での一致は修正時に別途確認済み）。
    fix round 3: avoid_parts が生成器の固有部分の語彙（ORG_CORES・KATA_CORES）を追加で avoid するようになった
    ため、参照実装（コア/全体の 1 件のみを返す旧ループ）とは完全一致ではなく**前方一致**になる（旧ループが
    返す値が avoid_parts の結果の先頭に必ず残ることだけを確認し、追加された語彙語は許容する）。"""
    import random

    from privacy_masker.operators.surrogate_generators import enumerate_candidates

    universe = sorted(set(enumerate_candidates()))
    sample = random.Random(20260928).sample(universe, 20000)
    extra = ["ネットワークス商船", "株式会社ネットワークス商船", "タワー美術館", "アオバITシステム", "システム開発システム",
             "データ銀行", "ビル管理病院", "IT", "商船", "青葉電機", "医療法人伊織会", "学校法人若葉学園", "営業二課",
             "東雲信用金庫株式会社", "汐見駅", "若葉タワー", "グランメゾン白樺", "市立若草小学校", "", "a"]
    for value in sample + extra:
        n = norm(value)
        for kind in ("org", "facility"):
            old = _old_avoid_parts_org_facility(kind, n)
            got = sm.avoid_parts(kind, n)
            assert got[:len(old)] == old, (kind, value, got, old)
            for part in got:
                assert part in n.replace(" ", ""), (kind, value, part)


def test_avoid_parts_splits_katakana_foreign_names():
    """最終レビュー推奨 2: 姓プールが漢字だけなので split_person はカタカナの外国人名を分けられず、avoid_parts が
    名前全体を返していた（偽名が元の名「ジョン」を再利用しうる）。「・」区切りならその要素、区切り無しなら
    GIVEN_KATA の最長前方一致と残りに分ける（2 字以上の要素だけ）。"""
    assert sm.avoid_parts("person", norm("ジョン・スミス")) == ("ジョン", "スミス")
    assert sm.avoid_parts("person", norm("ｼﾞｮﾝ･ｽﾐｽ")) == ("ジョン", "スミス")          # 半角は正規化で全角になる
    assert sm.avoid_parts("person", norm("ジョン スミス")) == ("ジョン", "スミス")       # 空白は正規化で消える
    assert sm.avoid_parts("person", norm("マイケルジョンソン")) == ("マイケル", "ジョンソン")
    assert sm.avoid_parts("person", norm("ミゲル・サントス")) == ("ミゲル", "サントス")
    assert sm.avoid_parts("person", norm("スズキ・タロウ")) == ("スズキ", "タロウ")
    assert sm.avoid_parts("person", norm("ジョン・ジョン")) == ("ジョン",)               # 重複は 1 件
    assert sm.avoid_parts("person", norm("ア・イ")) == ("ア・イ",)                       # 2 字以上の要素が無ければ全体
    assert sm.avoid_parts("person", norm("アッコ")) == ("アッコ",)                       # 分けられなければ全体
    assert sm.avoid_parts("person", norm("マイケル")) == ("マイケル",)
    # 漢字名・その他の文字種は従来どおり
    assert sm.avoid_parts("person", norm("山田太郎")) == ("山田", "太郎")
    assert sm.avoid_parts("person", norm("佐藤")) == ("佐藤",)
    assert sm.avoid_parts("person", norm("John Smith")) == ("JohnSmith",)
    # 残件1修正後: default_normalize が空白を全除去するため「やまだ はなこ」と正規化後は同一文字列になり、
    # プール語一致（やまだ・はなこ）も全体に加えて返る（全体だけを避けていた従来の挙動は不十分だった）
    assert sm.avoid_parts("person", norm("やまだはなこ")) == ("やまだはなこ", "やまだ", "はなこ")


def test_avoid_parts_person_separated_kana_hira_romaji():
    """最終レビューの残件1: default_normalize が区切り（空白）を消してから avoid_parts が呼ばれるため、
    区切り付きの日本語順カナ氏名・平仮名氏名・ローマ字氏名では名前全体しか避けられず、生成器が元の姓や名を
    再利用していた（設計 §7「元の姓・名の混入」の must 違反）。プール語（SURNAMES_*/GIVEN_*）の前方一致・
    後方一致で姓・名を additionally 抽出する。"""
    assert set(sm.avoid_parts("person", norm("スズキ タロウ"))) >= {"スズキ", "タロウ"}
    assert set(sm.avoid_parts("person", norm("やまだ はなこ"))) >= {"やまだ", "はなこ"}
    assert {p.lower() for p in sm.avoid_parts("person", norm("Taro Suzuki"))} >= {"taro", "suzuki"}
    assert "・" not in "".join(sm.avoid_parts("person", norm("山田・太郎")))          # 要素から「・」を除く
    assert all(len(p) >= 2 for p in sm.avoid_parts("person", norm("リン")))          # 2 字未満は出ない


def test_gender_overrides_is_read_only():
    with pytest.raises(TypeError):
        sm.GENDER_OVERRIDES["ハナコ"] = "m"


# ---- 最終修正波（ブランチ全体レビュー + Task 5 レビュー）----

@pytest.mark.parametrize("suffix", ["連邦", "帝国", "合衆国", "共和国"])
def test_avoid_parts_avoids_the_canonical_name_of_an_alias_inside_a_suffixed_country_name(suffix):
    """最終修正波 A1: 別名に 連邦・帝国・合衆国・共和国 が付いて country になった名前（「ソ連邦」「大韓帝国」
    「米合衆国」）は COUNTRY_CANONICAL にも COUNTRIES にも完全一致しないため、同じ国の標準名が避けられず偽名に
    出ていた（別名×語尾の総当たりの大半で漏れ）。COUNTRY_CANONICAL の全キー × 語尾で、country になるなら
    avoid_parts に標準名が入ること。"""
    checked = 0
    for key, canon in jp.COUNTRY_CANONICAL.items():
        name = norm(key + suffix)
        if sm.RuleCategorizer.categorize("location", name) != "country":
            continue
        checked += 1
        parts = sm.avoid_parts("location", name)
        assert parts[0] == name and canon in parts, (name, canon, parts)
    assert checked == len(jp.COUNTRY_CANONICAL)        # 語尾付きは全部 country（総当たりが空回りしていない）


def test_country_canonical_covers_variant_spellings_of_former_states():
    """最終修正波 A1: 表記ゆれの前身国家（ソヴィエト）と清の正式名を COUNTRY_CANONICAL に足した。"""
    assert jp.COUNTRY_CANONICAL["ソヴィエト連邦"] == "ロシア"
    assert jp.COUNTRY_CANONICAL["ソヴィエト社会主義共和国連邦"] == "ロシア"
    assert jp.COUNTRY_CANONICAL["大清帝国"] == "中国"
    assert sm.avoid_parts("location", norm("ソ連邦")) == ("ソ連邦", "ロシア")
    assert sm.avoid_parts("location", norm("大韓帝国")) == ("大韓帝国", "韓国")
    assert sm.avoid_parts("location", norm("米合衆国")) == ("米合衆国", "アメリカ")


def test_core_split_cuts_committee_names_matched_from_the_start():
    """最終修正波 A2: 自治体接頭の委員会（government の `^.{1,6}(都|…|村)…委員会$`）と department の
    `^[^株合有]{1,8}(…|委員会)$` は文字列先頭から一致するため開始位置が 0 になり、core_split がコアを切り出せず
    名前全体しか避けられなかった（「営業委員会」で department の偽名生成器が「営業」を再利用）。委員会 を
    core_split 専用の接尾語（_ORG_CORE_SUFFIXES）に足した。categorize / type_word_span は変えない。"""
    assert sm.avoid_parts("org", norm("朝霧市教育委員会"))[0] == "朝霧市教育"
    assert sm.avoid_parts("org", norm("営業委員会")) == ("営業",)
    got = sm.avoid_parts("org", norm("若葉県教育委員会"))
    assert got[0] == "若葉県教育" and "若葉" in got[1:]
    assert sm.RuleCategorizer.core_split("org", "営業委員会") == 2
    assert sm.RuleCategorizer.core_split("org", "朝霧市教育委員会") == 5
    # 開始位置 2 以上で表の行に当たる委員会は従来どおりその行で切る（「教育委員会」の 2）
    assert sm.RuleCategorizer.core_split("org", "青葉教育委員会") == 2
    # カテゴリ・種別語のスパンは変わらない（委員会 は core_split 専用）
    assert sm.RuleCategorizer.categorize("org", "営業委員会") == "department"
    assert sm.RuleCategorizer.categorize("org", "朝霧市教育委員会") == "government"
    assert sm.RuleCategorizer.type_word_span("org", "営業委員会") == (0, 5)


@pytest.mark.parametrize("value, stem", [
    ("朝霧地区", "朝霧"), ("朝霧1区", "朝霧"), ("朝霧山", "朝霧"), ("アサギリ州", "アサギリ"),
])
def test_avoid_parts_location_stems_outside_the_fake_vocabulary(value, stem):
    """最終修正波 A3: 地区・選挙区・自然地名の語幹のループと州・省の語幹は、既存テストの語幹（若葉・青葉・本町・
    八雲・ハルカゼ）が全部 _FAKE_CORE_VOCAB の語だったため、語彙の行が同じ語幹を足して互いに補い、どちらを
    消してもテストが通っていた。語彙に無い架空の語幹で、それぞれの行だけが語幹を足すことを確認する。"""
    assert stem not in sm._FAKE_CORE_VOCAB and sm._vocab_parts(value) == ()
    assert stem in sm.avoid_parts("location", norm(value))


@pytest.mark.parametrize("value, word", [("東青葉", "青葉"), ("新若葉", "若葉"), ("青葉台", "青葉")])
def test_avoid_parts_location_avoids_fake_vocabulary_words_without_a_suffix(value, word):
    """最終修正波 A3: 接尾語の無い地名（other）は地名そのもの以外に語幹を切り出す行が無く、生成器の語彙（TOWNS）に
    ある語は location の語彙の行（_vocab_parts）だけが避ける。"""
    assert sm.RuleCategorizer.categorize("location", norm(value)) == "other"
    assert sm.avoid_parts("location", norm(value)) == (value, word)


@pytest.mark.parametrize("name, expected", [
    ("若葉連邦委員会", "department"), ("連邦委員会", "department"),           # 裸の「連邦」は公的扱いにしない
    ("連邦通信委員会", "government"), ("連邦準備委員会", "government"), ("連邦捜査委員会", "government"),
])
def test_renpou_committee_is_government_only_as_a_compound(name, expected):
    """最終修正波 C3: 「連邦」の委員会は複合語（連邦通信・連邦準備・連邦取引・連邦選挙・連邦捜査）だけを
    government にする（複合部分を省略可能にしていたため裸の「連邦委員会」も government になっていた）。"""
    assert sm.RuleCategorizer.categorize("org", name) == expected


@pytest.mark.parametrize("value, word", [("営業部", "営業"), ("営業二課", "営業"), ("人事部", "人事"), ("営業推進委員会", "営業")])
def test_avoid_parts_org_avoids_department_function_words(value, word):
    """最終修正波（department の機能名）: department の規則は先頭から一致するため、部・課・室・チーム・係で終わる名前は
    core_split がコアを切り出せず名前全体しか避けられず、department の偽名生成器（DEPT_FUNCTIONS × DEPT_UNITS）が元の
    機能名を再利用していた。ORG では DEPT_FUNCTIONS の語（2 字以上）も避ける。FACILITY・LOCATION の生成器の語彙には
    DEPT_FUNCTIONS の語が現れないため、両者の avoid_parts は変えない。"""
    assert sm.RuleCategorizer.categorize("org", norm(value)) == "department"
    assert word not in sm._FAKE_CORE_VOCAB                         # 既存の語彙の行では足されない
    assert word in sm.avoid_parts("org", norm(value))
    assert word not in sm.avoid_parts("facility", norm(value)) and word not in sm.avoid_parts("location", norm(value))

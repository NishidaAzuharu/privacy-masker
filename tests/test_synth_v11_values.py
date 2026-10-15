"""値プールと表記生成（tools/synth_v11_values.py）。Drive・ネットワークは読まない。"""
import os
import random
import re
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import synth_v11_values as vals  # noqa: E402


@pytest.fixture
def pools():
    personas = [
        {"uuid": "u1", "sei": "雪見沢", "mei": "拓真", "age": 34, "sex": "男", "occupation": "看護師", "prefecture": "大分県", "hobby": "釣り"},
        {"uuid": "u2", "sei": "朱鷺野", "mei": "美咲", "age": 52, "sex": "女", "occupation": "会社員", "prefecture": "東京都", "hobby": "書道"},
        {"uuid": "u3", "sei": "田中", "mei": "花子", "age": 29, "sex": "女", "occupation": "教員", "prefecture": "北海道", "hobby": "登山"},
    ]
    ken = [("大分県", "大分市", "府内町"), ("東京都", "新宿区", "西新宿"), ("北海道", "札幌市中央区", "北一条西"), ("大阪府", "堺市堺区", "")]
    # 「田中」は評価セットの表層に入っている想定 → 除外されること
    return vals.build_pools(personas=personas, ken_rows=ken, excl_person={"田中", "田中花子"}, excl_place={"西新宿"})


def test_render_person_kanji_is_consistent_and_excluded_surfaces_never_appear(pools):
    rng = random.Random(1)
    seen = set()
    for _ in range(200):
        p = vals.render_person(rng, pools, "kanji", persona=None)
        assert p["full"].startswith(p["sei"]) and p["mei"] in p["full"]
        for k in ("full", "sei", "mei", "nick"):
            assert p[k] and not vals.is_excluded(p[k], pools.excl_person), p
        seen.add(p["sei"])
    assert "田中" not in seen


def test_render_person_uses_persona_name_when_given(pools):
    rng = random.Random(2)
    p = vals.render_person(rng, pools, "kanji", persona=pools.personas[0])
    assert p["sei"] == "雪見沢" and p["mei"] == "拓真" and p["full"].replace(" ", "").replace("　", "") == "雪見沢拓真"


@pytest.mark.parametrize("script", ["kata_ja", "kata_foreign", "hira", "romaji", "halfwidth", "historic"])
def test_render_person_scripts_have_expected_character_classes(pools, script):
    rng = random.Random(3)
    checks = {
        "kata_ja": lambda s: all("ァ" <= c <= "ヶ" or c in "ー 　" for c in s),
        "kata_foreign": lambda s: all("ァ" <= c <= "ヶ" or c in "ー・ 　" for c in s),
        "hira": lambda s: all("ぁ" <= c <= "ゖ" or c in "ー 　" for c in s),
        "romaji": lambda s: s.isascii(),
        "halfwidth": lambda s: any("ｦ" <= c <= "ﾟ" for c in s),
        "historic": lambda s: all("一" <= c <= "龥" or c == "々" for c in s),
    }
    for _ in range(50):
        p = vals.render_person(rng, pools, script, persona=None)
        assert checks[script](p["full"]), (script, p)
        assert p["sei"] and p["mei"]


def test_kata_foreign_covers_separator_variants(pools):
    rng = random.Random(4)
    fulls = [vals.render_person(rng, pools, "kata_foreign", None)["full"] for _ in range(300)]
    assert any("・" in f for f in fulls)                      # 中黒あり
    assert any(" " in f for f in fulls)                       # 空白
    assert any("・" not in f and " " not in f and len(f) >= 4 for f in fulls)   # 区切りなしの連結


def test_halfwidth_conversion_is_one_to_one():
    assert vals.to_halfwidth_kana("サトウ タロウ") == "ｻﾄｳ ﾀﾛｳ"
    assert vals.to_halfwidth_kana("グミ") == "ｸﾞﾐ"
    assert vals.to_hiragana("タカハシ") == "たかはし"


def test_gen_address_forms(pools):
    rng = random.Random(5)
    a = vals.gen_address(rng, pools, "pref_first", prefecture="大分県")
    assert a.startswith("大分県大分市府内町") and any(ch.isdigit() for ch in a)
    b = vals.gen_address(rng, pools, "banchi_first")
    assert b[0].isdigit()                                     # 番地先行
    c = vals.gen_address(rng, pools, "fullwidth")
    assert any("０" <= ch <= "９" for ch in c)
    d = vals.gen_address(rng, pools, "city_first")
    assert not any(d.startswith(p) for p in ("大分県", "東京都", "北海道", "大阪府"))
    for _ in range(100):
        assert "西新宿" not in vals.gen_address(rng, pools, "pref_first")   # 町域の除外


def test_gen_location_forms(pools):
    rng = random.Random(6)
    assert vals.gen_location(rng, pools, "prefecture") in {"大分県", "東京都", "北海道", "大阪府"}
    assert vals.gen_location(rng, pools, "country") in vals.COUNTRIES
    assert vals.gen_location(rng, pools, "foreign_kata") in vals.FOREIGN_PLACES_KATA
    assert vals.gen_location(rng, pools, "station").endswith("駅")
    towns = {vals.gen_location(rng, pools, "town") for _ in range(50)}
    assert "西新宿" not in towns and towns <= {"府内町", "北一条西"}


def test_gen_org_forms_and_facility(pools):
    rng = random.Random(7)
    assert any(k in vals.gen_org(rng, pools, "corporate") for k in ("株式会社", "合同会社", "（株）", "法人"))
    kb = vals.gen_org(rng, pools, "kata_brand")
    assert all("ァ" <= c <= "ヶ" or c == "ー" for c in kb) and not kb.endswith("株式会社")
    assert vals.gen_org(rng, pools, "abbr_latin").isupper()
    assert vals.gen_org(rng, pools, "dept").endswith(("課", "部", "室", "局"))
    assert vals.gen_facility(rng, pools)


def test_gen_unlabeled_values_have_expected_shapes():
    rng = random.Random(8)
    for _ in range(20):
        assert re.fullmatch(r"0\d{1,3}-\d{2,4}-\d{4}", vals.gen_unlabeled(rng, "PHONE"))
        assert re.fullmatch(r"[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]+", vals.gen_unlabeled(rng, "EMAIL"))
        assert vals.gen_unlabeled(rng, "URL").startswith("https://")
        assert re.fullmatch(r"〒?\d{3}-\d{4}", vals.gen_unlabeled(rng, "POSTAL"))
        assert vals.gen_unlabeled(rng, "DATE") and vals.gen_unlabeled(rng, "ID")


def test_snapshot_ken_all_parses_csv_rows(tmp_path):
    # KEN_ALL の実際の列: コード, 旧郵便番号, 郵便番号, 都道府県カナ, 市区郡カナ, 町域カナ, 都道府県, 市区郡, 町域, フラグ×6
    raw = ('01101,"060  ","0600000","ﾎｯｶｲﾄﾞｳ","ｻｯﾎﾞﾛｼﾁｭｳｵｳｷ","ｲｶﾞｵｷﾆｹｲｻｹｸﾞｲｱｹﾉ","北海道","札幌市中央区","以下に掲載がない場合",0,0,0,0,0,0\n'
           '13104,"160  ","1600023","ﾄｳｷｮｳﾄ","ｼﾆｼﾞｮｹｺ","ﾆｼｼﾃｼﾞｮｸ(1ﾃｮｳﾂﾆ)","東京都","新宿区","西新宿（１丁目）",0,0,1,0,0,0\n')
    zpath = tmp_path / "utf_ken_all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("utf_ken_all.csv", raw)
    out = tmp_path / "ken_all_v11.csv"
    m = vals.snapshot_ken_all(str(out), url="file://" + str(zpath))
    rows = vals.load_ken_all(str(out))
    assert ("北海道", "札幌市中央区", "") in rows and ("東京都", "新宿区", "西新宿") in rows
    assert m["rows"] == 2 and len(m["sha256"]) == 64


def test_load_personas_reads_snapshot_and_name_regex(tmp_path):
    p = tmp_path / "personas_v11.jsonl"
    p.write_text('{"uuid":"a","sei":"山本","mei":"健一","age":40,"sex":"男","occupation":"教員","prefecture":"奈良県","hobby":"将棋"}\n', encoding="utf-8")
    assert vals.load_personas(str(p))[0]["sei"] == "山本"
    assert vals.extract_name("山本 健一は奈良県に住む40歳の教員。") == ("山本", "健一")
    assert vals.extract_name("奈良県に住む男性。") is None

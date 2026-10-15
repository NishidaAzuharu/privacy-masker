"""KWDLC → 盲検 v2 変換（tools/blind_v2_from_kwdlc.py）。括弧の戻し・NE の位置・ラベル写像を人工の KNP で確認する。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from blind_v2_from_kwdlc import map_entities, parse_knp, rebuild_document  # noqa: E402

KNP = """# S-ID:w-1-01 括弧削除 KNP:4.2 MEMO:
* 1D
+ 1D
会場 かいじょう 会場 名詞 6 普通名詞 1 * 0 * 0 NIL
は は は 助詞 9 副助詞 2 * 0 * 0 NIL
* -1D
+ 2D <NE:LOCATION:蒼波区東原>
蒼波 あおなみ 蒼波 名詞 6 地名 4 * 0 * 0 NIL
区 く 区 接尾辞 14 名詞性特殊接尾辞 4 * 0 * 0 NIL
東原 ひがしはら 東原 名詞 6 地名 4 * 0 * 0 NIL
+ -1D <NE:DATE:１９９３年９月６日>
１９９３ せんきゅうひゃくきゅうじゅうさん １９９３ 名詞 6 数詞 7 * 0 * 0 NIL
年 ねん 年 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 NIL
９ きゅう ９ 名詞 6 数詞 7 * 0 * 0 NIL
月 がつ 月 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 NIL
６ ろく ６ 名詞 6 数詞 7 * 0 * 0 NIL
日 にち 日 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 NIL
です です だ 判定詞 4 * 0 判定詞 25 デス列基本形 27 NIL
。 。 。 特殊 1 句点 1 * 0 * 0 NIL
EOS
# S-ID:w-1-02 括弧始:（ 括弧終:） 括弧位置:13 KNP:4.2 MEMO:
* -1D
+ -1D <NE:OPTIONAL:平成５年>
平成 へいせい 平成 名詞 6 普通名詞 1 * 0 * 0 NIL
５ ご ５ 名詞 6 数詞 7 * 0 * 0 NIL
年 ねん 年 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 NIL
EOS
# S-ID:w-2 KNP:4.2 MEMO:
* -1D
+ -1D <NE:PERSON:山田花子>
山田 やまだ 山田 名詞 6 人名 5 * 0 * 0 NIL
花子 はなこ 花子 名詞 6 人名 5 * 0 * 0 NIL
+ -1D <NE:ORGANIZATION:若草台小学校>
が が が 助詞 9 格助詞 1 * 0 * 0 NIL
若草台 わかくさだい 若草台 名詞 6 固有名詞 3 * 0 * 0 NIL
小学校 しょうがっこう 小学校 名詞 6 普通名詞 1 * 0 * 0 NIL
に に に 助詞 9 格助詞 1 * 0 * 0 NIL
来た きた 来る 動詞 2 * 0 カ変動詞 14 タ形 10 NIL
。 。 。 特殊 1 句点 1 * 0 * 0 NIL
EOS
"""


def test_rebuild_reinserts_parentheses_and_splits_crossing_ne(tmp_path):
    p = tmp_path / "w.knp"
    p.write_text(KNP, encoding="utf-8")
    sents = parse_knp(str(p))
    assert [s["sid"] for s in sents] == ["w-1-01", "w-1-02", "w-2"]
    text, nes, warns = rebuild_document(sents)
    # 括弧位置 13 = 「会場は蒼波区東原１９９３年」の後（１９９３年 の直後）に （平成５年） が戻る
    assert text.startswith("会場は蒼波区東原１９９３年（平成５年）９月６日です。")
    assert text.endswith("山田花子が若草台小学校に来た。")
    kinds = {(t, sf) for t, sf, s, e in nes}
    assert ("LOCATION", "蒼波区東原") in kinds and ("OPTIONAL", "平成５年") in kinds
    assert ("PERSON", "山田花子") in kinds and ("ORGANIZATION", "若草台小学校") in kinds
    # 括弧をまたぐ DATE は 2 つに割れている（表層は割れた部分）
    dates = [(sf, text[s:e]) for t, sf, s, e in nes if t == "DATE"]
    assert sorted(sf for sf, _ in dates) == ["１９９３年", "９月６日"]
    assert all(text[s:e] == sf for t, sf, s, e in nes)
    assert any("分割" in w for w in warns)


def test_map_entities_labels_and_address_merge():
    text = "会場は蒼波区東原６−２１−６の歴史庭園。山田花子が若草台小学校に来た。２００１年、価格は１万円。"
    nes = [("LOCATION", "蒼波区東原", 3, 8), ("ARTIFACT", "歴史庭園", 15, 19), ("PERSON", "山田花子", 20, 24),
           ("ORGANIZATION", "若草台小学校", 25, 31), ("DATE", "２００１年", 35, 40), ("MONEY", "１万円", 44, 47)]
    ents, neutral, c = map_entities(text, nes)
    assert [text[s:e] for s, e, _ in ents] == ["蒼波区東原６−２１−６", "山田花子", "若草台小学校"]
    assert [lab for _, _, lab in ents] == ["ADDRESS", "PERSON", "ORG"]
    assert neutral == [[15, 19]]
    assert c["location_to_address"] == 1 and c["dropped_DATE"] == 1 and c["dropped_MONEY"] == 1 and c["neutral_ARTIFACT"] == 1

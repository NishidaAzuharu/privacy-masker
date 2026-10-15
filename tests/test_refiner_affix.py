"""種別語尾・法人格プレフィックスへのスパン拡張（AffixExpander、v0.8）。

拡張のみ（PII を落とさない）。隣接エンティティで停止。PERSON は対象外。
"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.operators import Tokenize
from privacy_masker.refiner import SpanRefiner
from privacy_masker.refiner_affix import AffixExpander
from privacy_masker.types import Entity
from tests.conftest import FakeDetector


def E(t, text, surface, score=0.9):
    s = text.index(surface)
    return Entity(t, s, s + len(surface), score, "test")


def expanded(text, ents, **kw):
    return [text[x.start:x.end] for x in AffixExpander(**kw).expand(text, ents)]


@pytest.mark.parametrize("text, etype, detected, expected", [
    ("原因は水戸データセンターの回線輻輳。", "FACILITY", "水戸", "水戸データセンター"),
    ("弊社浦和サービスセンターまでご連絡ください。", "FACILITY", "浦和", "浦和サービスセンター"),
    ("夜間はグリーンヒル横浜訪問看護ステーションへ。", "FACILITY", "グリーンヒル横浜", "グリーンヒル横浜訪問看護ステーション"),
    ("紛争は仙台地方裁判所を専属的合意管轄とする。", "ORG", "仙台", "仙台地方裁判所"),
    ("貴社鴻ノ巣興産本社ビル別館にて実施。", "ORG", "鴻ノ巣興産", "鴻ノ巣興産本社ビル別館"),
    ("提供事業所：株式会社琉伸メディカルサポート（訪問リハ）", "ORG", "株式会社琉伸", "株式会社琉伸メディカルサポート"),
    ("集合は高輪ゲートウェイ駅の改札。", "FACILITY", "高輪ゲートウェイ", "高輪ゲートウェイ駅"),
    ("名古屋市に本社を置く。", "LOCATION", "名古屋", "名古屋市"),
    ("東京都庁に提出。", "LOCATION", "東京", "東京都庁"),
])
def test_suffix_expansion(text, etype, detected, expected):
    assert expanded(text, [E(etype, text, detected)]) == [expected]


@pytest.mark.parametrize("text, detected, expected", [
    ("参加: 社会福祉法人ひだまり会相談員", "ひだまり会", "社会福祉法人ひだまり会"),
    ("紹介先：医療法人碧水会 ひなた丘温泉病院", "碧水会", "医療法人碧水会"),
    ("介護記録（特別養護老人ホームやすらぎの郷 2階）", "やすらぎの郷", "特別養護老人ホームやすらぎの郷"),  # 階数は既定では取らない
    ("提供事業所：株式会社 琉伸", "琉伸", "株式会社 琉伸"),  # 法人格と名称の間の空白 1 個
])
def test_prefix_expansion(text, detected, expected):
    assert expanded(text, [E("ORG", text, detected)]) == [expected]


def test_numeric_suffix_can_be_enabled_per_type():
    # 旧挙動（v0.8.0）: numeric_suffix_types で型を指定すれば階数も取り込む
    text = "介護記録（特別養護老人ホームやすらぎの郷 2階）"
    ents = [E("ORG", text, "やすらぎの郷")]
    assert expanded(text, ents, numeric_suffix_types=("ORG",)) == ["特別養護老人ホームやすらぎの郷 2階"]
    assert expanded(text, ents, numeric_suffix_types=("FACILITY",)) == ["特別養護老人ホームやすらぎの郷"]


def test_stops_at_adjacent_entity():
    text = "青葉電力送配電の点検"
    ents = [E("ORG", text, "青葉"), E("ORG", text, "電力送配電")]
    # 「青葉」の右隣は別エンティティなので「電力」を取り込まない
    assert expanded(text, ents) == ["青葉", "電力送配電"]


def test_person_is_not_expanded():
    text = "田中クリニックの田中院長"
    ents = [E("PERSON", text, "田中")]
    assert expanded(text, ents) == ["田中"]


def test_no_expansion_without_adjacent_affix():
    text = "水戸で会議。株式会社の設立。"
    ents = [E("LOCATION", text, "水戸")]
    assert expanded(text, ents) == ["水戸"]


def test_admin_suffix_not_doubled():
    text = "名古屋市市議会"
    ents = [E("LOCATION", text, "名古屋市")]
    assert expanded(text, ents) == ["名古屋市"]  # 既に「市」で終わる → 2 個目の市は取り込まない


def test_max_added_chars_caps_growth():
    text = "琉伸メディカルサポートシステムズ"
    ents = [E("ORG", text, "琉伸")]
    assert expanded(text, ents, max_added_chars=8) == ["琉伸メディカル"]  # サポート(4)で予算超過


def test_positions_score_and_source_preserved():
    text = "水戸データセンターの障害"
    ent = E("FACILITY", text, "水戸", score=0.77)
    (out,) = AffixExpander().expand(text, [ent])
    assert (out.start, out.end) == (0, len("水戸データセンター"))
    assert out.score == 0.77 and out.source == "test" and out.entity_type == "FACILITY"


def test_disabled_expander_is_identity():
    text = "水戸データセンター"
    ents = [E("FACILITY", text, "水戸")]
    assert expanded(text, ents, suffixes=(), prefixes=(), admin_suffix_chars="") == ["水戸"]


# ---- Refiner / Engine 統合 ----

def test_span_refiner_expands_by_default_and_can_be_disabled():
    text = "原因は水戸データセンターの回線輻輳。"
    ents = [E("FACILITY", text, "水戸")]
    assert [text[x.start:x.end] for x in SpanRefiner().refine(text, ents)] == ["水戸データセンター"]
    assert [text[x.start:x.end] for x in SpanRefiner(expand_affixes=False).refine(text, ents)] == ["水戸"]


def test_span_refiner_trims_then_expands():
    text = "連絡先は 浦和サービスセンターまで"
    ents = [Entity("FACILITY", 4, 7, 0.9, "test")]  # " 浦和" 先頭空白込み
    assert [text[x.start:x.end] for x in SpanRefiner().refine(text, ents)] == ["浦和サービスセンター"]


def test_engine_masks_whole_compound_name():
    text = "障害は水戸データセンターで発生した。"
    eng = PrivacyEngine(detectors=[FakeDetector([E("FACILITY", text, "水戸")])],
                        policy=Policy(default=Tokenize(strategy="sequential")))
    assert eng.transform(text).text == "障害は<FACILITY_001>で発生した。"


def test_morphological_refiner_expands():
    pytest.importorskip("janome")
    from privacy_masker import MorphologicalRefiner
    text = "紛争は仙台地方裁判所を専属的合意管轄とする。"
    ents = [E("ORG", text, "仙台")]
    assert [text[x.start:x.end] for x in MorphologicalRefiner().refine(text, ents)] == ["仙台地方裁判所"]
    assert [text[x.start:x.end] for x in MorphologicalRefiner(expand_affixes=False).refine(text, ents)] == ["仙台"]


@pytest.mark.parametrize("text, etype, detected, expected", [
    ("物件はサンモールビル別館302号室です。", "FACILITY", "サンモールビル別館", "サンモールビル別館302号室"),
    ("受付は本社別館 3階です。", "FACILITY", "本社別館", "本社別館 3階"),
    ("集合は東雲ビル５Ｆロビー。", "FACILITY", "東雲ビル", "東雲ビル５Ｆ"),
    ("搬入は第二倉庫B棟へ。", "FACILITY", "第二倉庫", "第二倉庫B棟"),
    ("会場は青葉会館1号館。", "FACILITY", "青葉会館", "青葉会館1号館"),
])
def test_numeric_suffix_expansion(text, etype, detected, expected):
    """数字付き語尾は既定では付けない（文字ヘッドが自分で取り込む。付けると gold 外の追加になる）。
    numeric_suffix_types で型を指定すれば v0.8.0 の挙動に戻る。"""
    assert expanded(text, [E(etype, text, detected)]) == [detected]
    assert expanded(text, [E(etype, text, detected)], numeric_suffix_types=(etype,)) == [expected]


# ---- LOCATION の語尾は地形・交通・公共機関だけ（2026-09-14）

@pytest.mark.parametrize("text, detected, expected", [
    ("・鳥取市支社の技術部門へ依頼", "鳥取市", "鳥取市"),                # 企業語尾は付けない
    ("大阪本社移転までの流れ", "大阪", "大阪"),
    ("大阪カスタマーセンターへ", "大阪", "大阪"),
    ("集合は高輪ゲートウェイ駅の改札。", "高輪ゲートウェイ", "高輪ゲートウェイ駅"),   # 交通
    ("東京都庁に提出。", "東京", "東京都庁"),                          # 行政区画 + 公共機関
    ("青凪市役所本庁舎3階", "青凪", "青凪市役所"),
    ("名古屋市に本社を置く。", "名古屋", "名古屋市"),
])
def test_location_suffix_is_restricted(text, detected, expected):
    assert expanded(text, [E("LOCATION", text, detected)]) == [expected]


def test_org_still_gets_corporate_suffixes():
    text = "・鳥取支社の技術部門へ依頼"
    assert expanded(text, [E("ORG", text, "鳥取")]) == ["鳥取支社"]


def test_admin_char_rule_known_limitation():
    """行政区画の 1 文字語尾（市・区…）は「北米市場」の 市 も付けてしまう（既知の限界、盲検 v1 で 1 件）。
    語尾リストの制限とは別の規則なので今回は変えていない。挙動が変わったら気付けるように固定する。"""
    text = "北米市場で絶大な人気"
    assert expanded(text, [E("LOCATION", text, "北米")]) == ["北米市"]


def test_numeric_suffix_does_not_absorb_unrelated_numbers():
    text = "水戸データセンターは2026年に開設。"
    ents = [E("FACILITY", text, "水戸")]
    assert expanded(text, ents) == ["水戸データセンター"]  # 「2026年」は取り込まない


# ---- 住所の末尾（盲検評価 v1 の ADDRESS 漏れ）

@pytest.mark.parametrize("text, etype, detected, expected", [
    ("ご請求先: 千葉県東雲市栄浜1-1-2 スカイコート栄浜 908　大楠 様", "ADDRESS", "千葉県東雲市栄浜1-1-2 スカイコート栄浜", "千葉県東雲市栄浜1-1-2 スカイコート栄浜 908"),
    ("交換品は長野県蒼波市東原5685 同病院へ", "ADDRESS", "長野県蒼波市東原", "長野県蒼波市東原5685"),
    ("東雲市栄浜2-1-9 グランメゾン栄浜206へ", "ADDRESS", "東雲市栄浜2-1", "東雲市栄浜2-1-9"),   # 建物名までは取らない（数値末尾だけ）
    ("栄浜3-4先の集積所は廃止", "LOCATION", "栄浜", "栄浜3-4"),
    ("住所 千葉県東雲市栄浜2-14-9 です", "ADDRESS", "千葉県東雲市栄浜2", "千葉県東雲市栄浜2-14-9"),
])
def test_address_tail_expansion(text, etype, detected, expected):
    assert expanded(text, [E(etype, text, detected)]) == [expected]


@pytest.mark.parametrize("text, detected", [
    ("住所 千葉県東雲市栄浜2-14-9　047-000-0000\n備考: 色", "千葉県東雲市栄浜2-14-9"),
    ("栄浜2-14-9　090-0000-0000\n担当", "栄浜2-14-9"),
    ("東京都港区芝1-2-3 03-1234-5678", "東京都港区芝1-2-3"),
    ("東京都港区芝1-2-3 0120-123-456", "東京都港区芝1-2-3"),
])
def test_address_tail_does_not_absorb_phone_numbers(text, detected):
    """空白 + 3〜4 桁 が部屋番号のパターンに合い、続く -000-0000 を番地の連鎖として
    2 段で取り込んでいた（盲検 v1 で見た形。例は架空）。電話番号は住所の一部ではない。"""
    assert expanded(text, [E("ADDRESS", text, detected)]) == [detected]


def test_address_tail_room_number_after_space_is_still_taken():
    # 電話番号ガードで、建物名の後の部屋番号（既存の盲検 v1 の漏れ対策）を壊していないこと
    text = "ご請求先: 千葉県東雲市栄浜1-1-2 スカイコート栄浜 908　大楠 様"
    assert expanded(text, [E("ADDRESS", text, "千葉県東雲市栄浜1-1-2 スカイコート栄浜")]) == \
        ["千葉県東雲市栄浜1-1-2 スカイコート栄浜 908"]
    text2 = "所在地: 港区赤坂1-2-3 サンモールビル別館　302号室"
    assert expanded(text2, [E("ADDRESS", text2, "港区赤坂1-2-3 サンモールビル別館")]) == \
        ["港区赤坂1-2-3 サンモールビル別館　302号室"]


def test_address_tail_digit_budget():
    # 保険: 取り込んだ数字が 9 桁以上なら住所の末尾ではない
    text = "口座 1234567890 港区赤坂1 2345 6789 0123"
    ents = [E("ADDRESS", text, "港区赤坂1")]
    out = expanded(text, ents)
    assert out == ["港区赤坂1"] or len([c for c in out[0] if c.isdigit()]) <= 9


@pytest.mark.parametrize("text, etype, detected", [
    ("京都3日目は伏見へ", "LOCATION", "京都"),                                   # 数量・日付は取らない
    ("東京都千代田区丸の内1-9-2 2025年に移転", "ADDRESS", "東京都千代田区丸の内1-9-2"),
    ("神奈川県厚木市中町2-6-1 ハルカワ精密厚木事業所", "ADDRESS", "神奈川県厚木市中町2-6-1"),   # 空白の後が数字でない
    ("大阪 300 さ 12-34", "LOCATION", "大阪"),                                   # ナンバープレート: 空白＋数字は LOCATION では取らない
    ("東京2020大会", "LOCATION", "東京"),
])
def test_address_tail_not_expanded(text, etype, detected):
    assert expanded(text, [E(etype, text, detected)]) == [detected]

import pytest

pytest.importorskip("janome", reason="requires [janome] extra")

from privacy_masker.engine import PrivacyEngine
from privacy_masker.operators import Surrogate
from privacy_masker.policy import Policy
from privacy_masker.refiner_janome import MorphologicalRefiner
from privacy_masker.types import Entity

from .conftest import FakeDetector


def E(t, s, e):
    return Entity(entity_type=t, start=s, end=e, score=1.0, source="test")


@pytest.fixture(scope="module")
def refiner():
    return MorphologicalRefiner()


def spans(refiner, text, ents):
    return [(x.start, x.end, x.entity_type) for x in refiner.refine(text, ents)]


def test_trims_trailing_particle(refiner):
    text = "加藤の件です"
    assert spans(refiner, text, [E("PERSON", 0, 3)]) == [(0, 2, "PERSON")]


def test_trims_chained_hiragana_function_words(refiner):
    # ルール式SpanRefinerでは安全側で残していた「などでは」連鎖を品詞情報で解決
    text = "日本などでは主流だ"
    assert spans(refiner, text, [E("LOCATION", 0, 6)]) == [(0, 2, "LOCATION")]


def test_trims_particle_after_hiragana_noun(refiner):
    # ひらがな終わり固有名詞+助詞: 「みなと」は名詞と分かるので「まで」だけ削れる
    text = "みなとまで歩く"
    assert spans(refiner, text, [E("LOCATION", 0, 5)]) == [(0, 3, "LOCATION")]


def test_keeps_hiragana_noun_itself(refiner):
    text = "みなとまで歩く"
    assert spans(refiner, text, [E("LOCATION", 0, 3)]) == [(0, 3, "LOCATION")]


def test_trims_honorific_suffix_for_person_only(refiner):
    text = "田中さんは休み"
    assert spans(refiner, text, [E("PERSON", 0, 4)]) == [(0, 2, "PERSON")]
    # ORGでは接尾トリムはしない（人名接尾のみが対象なので影響なし）
    text2 = "田中さんは休み"
    assert spans(refiner, text2, [E("ORG", 0, 2)]) == [(0, 2, "ORG")]


def test_trims_leading_particle(refiner):
    text = "対の大韓民国が"
    assert spans(refiner, text, [E("LOCATION", 1, 6)]) == [(2, 6, "LOCATION")]


def test_trims_whitespace_and_punctuation(refiner):
    text = "担当:\n碇谷、以上"
    assert spans(refiner, text, [E("PERSON", 3, 7)]) == [(4, 6, "PERSON")]


def test_mid_morpheme_end_expanded_to_boundary(refiner):
    # v0.6: スパン端が形態素の途中にある場合は境界まで拡張する（旧: 触らない）。
    # 企業+部門連結の後半切れ（青葉電力送配電→青葉電力送）を修復できる
    text = "青葉電力送配電の件"
    got = spans(refiner, text, [E("ORG", 0, 5)])   # 「青葉電力送」(送配の途中で切れた)
    # Janomeは 青葉|電力|送配|電 と分割するため、端が触れている形態素（送配）の
    # 境界=6まで拡張される（完全修復には辞書登録が必要だが欠け幅は縮む）
    assert got == [(0, 6, "ORG")]


def test_drops_empty_after_trim(refiner):
    text = "a の b"
    assert spans(refiner, text, [E("LOCATION", 1, 4)]) == []


def test_engine_integration_surrogate_keeps_function_words(refiner):
    text = "みなとまで配送します。"
    engine = PrivacyEngine(
        detectors=[FakeDetector([E("LOCATION", 0, 5)])],
        policy=Policy(rules={"LOCATION": Surrogate(kind="address")}),
        secret_key=b"k" * 16,
        refiner=refiner,
    )
    out = engine.transform(text).text
    assert "まで配送します。" in out and "みなと" not in out


# ---- 辞書に無い語の誤解析への耐性（NEologd議論起点） ----

def test_unknown_hiragana_name_not_eroded(refiner):
    # 「つじの」は つ(助動詞)+じ(助動詞)+の と誤解析されるが、
    # アンカーが自立語でないため削らない（PII欠落の防止）
    text = "担当のつじのに引き継ぐ"
    assert spans(refiner, text, [E("PERSON", 3, 6)]) == [(3, 6, "PERSON")]


def test_ambiguous_hiragana_tail_kept(refiner):
    # 「みことまで」: みこ(名詞,一般)+と+まで と解析されるが、みこは固有名詞では
    # ないため信用せず、まで・とを削らない（「みこと」が名前の可能性を守る）
    text = "みことまで連絡"
    assert spans(refiner, text, [E("PERSON", 0, 5)]) == [(0, 5, "PERSON")]


def test_proper_noun_anchor_allows_trim(refiner):
    # 「みなと」は固有名詞と解析されるので「まで」を削ってよい（既存挙動の維持）
    text = "みなとまで歩く"
    assert spans(refiner, text, [E("LOCATION", 0, 5)]) == [(0, 3, "LOCATION")]


def test_user_dict_enables_confident_trim(tmp_path):
    # ユーザー辞書に「みこと」を固有名詞として登録すると、境界を確信して削れる
    udic = tmp_path / "names.csv"
    udic.write_text(
        "みこと,1288,1288,1000,名詞,固有名詞,人名,名,*,*,みこと,ミコト,ミコト\n",
        encoding="utf-8")
    r = MorphologicalRefiner(user_dict=str(udic))
    text = "みことまで連絡"
    assert spans(r, text, [E("PERSON", 0, 5)]) == [(0, 3, "PERSON")]


# ---- v0.6: 語境界スナップ ----

def _ents(*triples):
    return [Entity(t, s, e, 0.99, "test") for s, e, t in triples]


def test_snap_merges_split_entity_same_type():
    # 鮫|島病院 と割れた同typeスパンが形態素境界へ拡張され1スパンに統合される
    r = MorphologicalRefiner()
    text = "鮫島病院からの紹介"
    out = r.refine(text, _ents((0, 1, "FACILITY"), (1, 4, "FACILITY")))
    assert [(e.start, e.end, e.entity_type) for e in out] == [(0, 4, "FACILITY")]
    assert text[out[0].start:out[0].end] == "鮫島病院"


def test_snap_expands_each_type_separately_on_overlap():
    # 型が違う場合は統合せず、各スパンを境界まで拡張して返す（重複はresolverが解決）
    r = MorphologicalRefiner()
    text = "鮫島病院からの紹介"
    out = r.refine(text, _ents((0, 1, "ADDRESS"), (1, 4, "FACILITY")))
    got = {(e.start, e.end, e.entity_type) for e in out}
    assert (0, 2, "ADDRESS") in got          # 鮫→鮫島（形態素境界）
    assert (0, 4, "FACILITY") in got         # 島病院→鮫島病院


def test_snap_drops_fragment_inside_common_morpheme():
    # 「拡張」（サ変名詞）の内部断片「拡」は誤検出として棄却
    r = MorphologicalRefiner()
    text = "収縮期血圧 139\n拡張期血圧 69"
    ents = _ents((10, 11, "PERSON"))
    assert text[10:11] == "拡"
    assert r.refine(text, ents) == []


def test_snap_expands_fragment_inside_proper_noun():
    # 固有名詞形態素の内部断片は棄却でなく境界まで拡張（山→山田）
    r = MorphologicalRefiner()
    text = "担当は山田です"
    out = r.refine(text, _ents((3, 4, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(3, 5)]
    assert text[3:5] == "山田"


def test_ascii_fragment_dropped():
    # 1文字・小文字混じりの英字断片（S・di等）は棄却、3文字以上は残す
    r = MorphologicalRefiner()
    assert r.refine("S) H24年", _ents((0, 1, "PERSON"))) == []
    assert r.refine("Hb di値", _ents((3, 5, "ORG"))) == []
    out = r.refine("NTTと田中", _ents((0, 3, "ORG")))
    assert [(e.start, e.end) for e in out] == [(0, 3)]


def test_two_letter_uppercase_acronym_kept():
    """2文字でも全て大文字なら実在の固有名詞として残す。

    棄却すると検出済みエンティティが平文に戻るため、4評価セットでの実測
    （gold 10件を救い過剰マスク7件増、うち4件はPERSON/ADDRESS）に基づき
    大文字のみのアクロニムは残す側に倒す。小文字を含むものは断片とみなす。
    """
    r = MorphologicalRefiner()
    for text, span, ty in (("NS:田中", (0, 2), "ORG"),
                           ("LG製のテレビ", (0, 2), "ORG"),
                           ("AIと呼ばれる", (0, 2), "ORG"),
                           ("JR東海の駅", (0, 2), "FACILITY")):
        out = r.refine(text, _ents((*span, ty)))
        assert [(e.start, e.end) for e in out] == [span], (text, out)


def test_ascii_fragment_dropped_even_with_surrounding_space():
    # 生スパンが空白込み（" di"等）でも英字断片として棄却できる
    r = MorphologicalRefiner()
    assert r.refine("HS diastolic murmur", _ents((2, 5, "ORG"))) == []


def test_ascii_fragment_checked_before_expansion():
    # 英字断片の判定はスナップ拡張前の原スパンに対して行う
    # （diastolicの「di」が形態素拡張で単語全体に化けて生き残るのを防ぐ）
    r = MorphologicalRefiner()
    text = "HS diastolic murmur"
    assert r.refine(text, _ents((3, 5, "ORG"))) == []


def test_snap_disabled_keeps_previous_behavior():
    r = MorphologicalRefiner(snap_boundaries=False, drop_ascii_len=0)
    text = "収縮期血圧 139\n拡張期血圧 69"
    out = r.refine(text, _ents((10, 11, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(10, 11)]


def test_snap_aligned_spans_unchanged():
    r = MorphologicalRefiner()
    text = "久遠寺富美子です"
    out = r.refine(text, _ents((0, 6, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(0, 6)]


# ---- v0.6レビュー対応の回帰テスト ----

def test_initial_with_honorific_not_dropped():
    # Critical回帰: 「Kさん」の敬称トリム後に残るイニシャルは棄却しない（PII漏れ防止）
    r = MorphologicalRefiner()
    out = r.refine("Kさんが来た", _ents((0, 3, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(0, 1)]
    out2 = r.refine("TK君に渡す", _ents((0, 3, "PERSON")))
    assert [(e.start, e.end) for e in out2] == [(0, 2)]


def test_fragment_in_common_noun_expanded_not_dropped():
    # IPADIC非収載の実名（東雲=名詞,一般の1形態素）の断片は棄却でなく拡張（漏れ防止）
    r = MorphologicalRefiner()
    text = "東雲さんに連絡"
    out = r.refine(text, _ents((0, 1, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(0, 2)]
    assert text[0:2] == "東雲"


def test_merge_keeps_max_score_on_containment():
    # 包含統合でも高scoreを保持する（resolverのmin_score併用で消えないように）
    r = MorphologicalRefiner()
    ents = [Entity("FACILITY", 0, 4, 0.3, "t"), Entity("FACILITY", 1, 3, 0.99, "t")]
    out = r.refine("鮫島病院からの紹介", ents)
    assert len(out) == 1 and out[0].score == 0.99


# ---- v0.7: 助詞で隔てられた自立語（動詞等）のトリム ----

def test_verb_tail_not_trimmed_by_design():
    """動詞句のトリムは実装しない（v0.7で試みたが撤回）。

    IPADICは 之/ノ/が を格助詞と解析するため、隔てとして使うと
    勝之進→勝・元湯ノ出→元湯・浅が野→野 のようにPIIが分断される。
    過剰マスク数文字の改善に対しPII漏れの代償が大きいため、
    「松本駅から歩いて」は縮めないまま残す（安全側）。
    """
    r = MorphologicalRefiner()
    text = "松本駅から歩いて10分の店に集合"
    out = r.refine(text, _ents((0, 8, "FACILITY")))
    assert [text[e.start:e.end] for e in out] == ["松本駅から歩いて"]


def test_leading_conjunction_not_trimmed():
    # 自立語（接続詞・副詞・動詞）のトリムは実装しない（PII漏れの代償が大きい）
    r = MorphologicalRefiner()
    text = "追って村井へ連絡"
    out = r.refine(text, _ents((0, 5, "PERSON")))
    assert [text[e.start:e.end] for e in out] == ["追って村井"]


def test_keeps_name_misparsed_as_verb():
    # 「皆見」は 皆+見(動詞) と誤解析されるが、助詞で隔てられていないため削らない
    r = MorphologicalRefiner()
    out = r.refine("皆見の到着が遅れます", _ents((0, 2, "PERSON")))
    assert [(e.start, e.end) for e in out] == [(0, 2)]


def test_keeps_hiragana_names_misparsed_as_content_words():
    # ひらがな姓は動詞・フィラー・感動詞・形容詞に誤解析されるが、いずれも削らない
    r = MorphologicalRefiner()
    for name in ("あべ", "おおた", "あおき", "かえで", "ありた", "つむぎ", "たかなし"):
        text = f"{name}さんに連絡"
        out = r.refine(text, _ents((0, len(name), "PERSON")))
        assert [text[e.start:e.end] for e in out] == [name], name


def test_noun_compound_not_trimmed():
    # 名詞の連続（大学+学部）は削らない（Surrogate置換でむしろ自然な範囲）
    r = MorphologicalRefiner()
    text = "常盤台工科大学情報工学部 卒業"
    out = r.refine(text, _ents((0, 12, "ORG")))
    assert [(e.start, e.end) for e in out] == [(0, 12)]


def test_keeps_kana_org_head_misparsed_as_verb():
    # 「みずき信用金庫」の みずき は み(動詞)+ず+き と誤解析されるが削らない
    r = MorphologicalRefiner()
    text = "みずき信用金庫の窓口"
    out = r.refine(text, _ents((0, 7, "ORG")))
    assert [text[e.start:e.end] for e in out] == ["みずき信用金庫"]


def test_keeps_katakana_name_head_misparsed_as_adverb():
    # 「イザベラ・フェレイラ」の イザ は副詞と誤解析されるが削らない
    r = MorphologicalRefiner()
    text = "イザベラ・フェレイラ様よりご連絡"
    out = r.refine(text, _ents((0, 10, "PERSON")))
    assert [text[e.start:e.end] for e in out] == ["イザベラ・フェレイラ"]



# ---- v0.7レビューで実証されたPII漏れの回帰テスト ----

@pytest.mark.parametrize("text,span,expected", [
    # 閉クラス漢字（IPADICで接続詞・副詞になる一般的な人名用字）
    ("尚子を訪問しました。", (0, 2), "尚子"),
    ("将也さんに連絡してください", (0, 2), "将也"),
    ("又川産業と契約しました", (0, 4), "又川産業"),
    ("尚美学園の卒業生です", (0, 4), "尚美学園"),
    ("只越様よりご連絡", (0, 2), "只越"),
    ("猶村の担当です", (0, 2), "猶村"),
    ("彌永建設に発注", (0, 4), "彌永建設"),
    ("志賀又へ連絡", (0, 3), "志賀又"),
    ("山尚の物件", (0, 2), "山尚"),
    # PII内部の記号・連体化「の」が隔てとして誤用されるケース
    ("見の越駅で待ち合わせ", (0, 4), "見の越駅"),
    ("見の越運輸に発注", (0, 5), "見の越運輸"),
    ("里の見に行く", (0, 3), "里の見"),
    ("与々木さんに連絡", (0, 3), "与々木"),
    ("与々木製作所に発注", (0, 6), "与々木製作所"),
    ("田中 走が担当します", (0, 4), "田中 走"),
    ("田中　実は本日休みです。", (0, 4), "田中　実"),
    ("担当は佐藤・誠に変更します。", (3, 7), "佐藤・誠"),
    ("鈴木 泳が担当します。", (0, 4), "鈴木 泳"),
    # 漢字の助詞（之 = 助詞,格助詞）を名前の一部として守る（v0.6以前からの既存バグ）
    ("有之さんに連絡", (0, 2), "有之"),
    ("将之は本日休みです。", (0, 2), "将之"),
    ("宗之を訪問しました。", (0, 2), "宗之"),
    # 2文字以上の断片は棄却せず形態素境界まで広げる（見越さ の内部の 見越。同既存バグ）
    ("見越さんに連絡", (0, 2), "見越"),
])
def test_no_pii_truncation(text, span, expected):
    """PIIが1文字も欠けないことを検証する（境界の外側への拡張は安全側なので許容）。

    v0.7の自立語トリムで実証された漏れ（尚子→子、見の越駅→見の、与々木→木、
    田中␣走→田中 等）の回帰テスト。expectedは最低限含まれるべき表層。
    """
    r = MorphologicalRefiner()
    out = r.refine(text, _ents((span[0], span[1], "PERSON")))
    assert len(out) == 1, f"スパンが消えた/分裂した: {[text[e.start:e.end] for e in out]}"
    got = text[out[0].start:out[0].end]
    assert expected in got, f"PIIが欠けた: {got!r} に {expected!r} が含まれない"



@pytest.mark.parametrize("text,span", [
    # 漢字・カタカナの助詞が付属語トリムで削られていた（v0.3.2以来の漏れ）
    ("有之さんに連絡", (0, 2)),
    ("会議に勝之進も参加します。", (3, 6)),
    ("担当はブルーノです。", (3, 7)),
    ("患者ミサヲの検査結果", (2, 5)),
    ("マリノさんへ連絡", (0, 3)),
    ("本日デボラを訪問しました。", (2, 5)),
    # スパン全体が付属語と解析され、実在のかな・カナ人名が消えていた（同上）
    ("担当はチャンです。", (3, 6)),
    ("氏名: かな", (4, 6)),
    ("担当はりかです。", (3, 5)),
    ("担当はよねです。", (3, 5)),
    ("患者はなの検査結果", (2, 4)),
    ("担当はさえです。", (3, 5)),
])
def test_pii_never_truncated_or_dropped(text, span):
    """PIIが欠けない・消えないことの回帰テスト（v0.7レビューで実モデル実証済み）。"""
    r = MorphologicalRefiner()
    gold = text[span[0]:span[1]]
    out = r.refine(text, _ents((span[0], span[1], "PERSON")))
    assert out, f"スパンが消えた: {gold!r}"
    got = text[out[0].start:out[0].end]
    assert gold in got, f"PIIが欠けた: {got!r} に {gold!r} が含まれない"


# ---- v0.8: 分布内評価で見つかった欠け（ひらがな名の端・単字姓）----

@pytest.mark.parametrize("text, name", [
    ("社　人事部\n石川まりの\nkashp", "石川まりの"),      # 末尾「の」= 連体化助詞と解析、直後は改行
    ("掲示をあさひなが設置", "あさひな"),                 # 末尾「な」= 助動詞と解析、直後は助詞
    ("後任は佐藤まりも氏です。", "佐藤まりも"),            # 末尾「も」、直後は敬称
    ("請求者 矢野ねねか\n理由 医療", "矢野ねねか"),       # 3 形態素の助詞run、直後は改行
    ("研究分担者 金子かの、殿崎麻椰", "金子かの"),        # 直後は読点
    ("名前は篠原秋もさん。", "篠原秋も"),
    ("はすみさん、今日の", "はすみ"),                     # 先頭「は」= 係助詞、ひらがなだけの名前
    ("記入 はすみ", "はすみ"),
    ("担当ははすみです。", "はすみ"),                      # 直前も助詞
    ("平が担当。", "平"),                                 # Janome が「平が」を 1 形態素にする単字姓
    ("森より連絡", "森"),
])
def test_hiragana_edges_and_single_kanji_surnames_are_kept(refiner, text, name):
    s = text.index(name)
    out = refiner.refine(text, [E("PERSON", s, s + len(name))])
    assert [text[x.start:x.end] for x in out] == [name]


@pytest.mark.parametrize("text, span, expected", [
    ("田中は本日休みです。", "田中は", "田中"),     # 直後が内容語 → 助詞は削る
    ("田中さんに連絡", "田中さん", "田中"),        # 敬称は文脈に関わらず削る
    ("の田中が担当", "の田中", "田中"),            # 漢字名の先頭助詞は削る
])
def test_context_guard_does_not_block_legitimate_trims(refiner, text, span, expected):
    s = text.index(span)
    out = refiner.refine(text, [E("PERSON", s, s + len(span))])
    assert [text[x.start:x.end] for x in out] == [expected]


def test_space_separated_name_not_trimmed():
    """空白区切りの複合人名は先頭/末尾トリムの対象外にする。

    「かな 麻里 松本」の「かな」は独立の平仮名語として助詞・接尾語トリムに
    食われるが、実際には名の一部。OpenPII 実データで 14 件の分断を確認した。
    """
    r = MorphologicalRefiner()
    for surf in ("かな 麻里 松本", "えりか なな", "もとのぶ ひろみち 横山",
                 "りな なほみ Ivers", "Vildan 上田", "帯広市 南の森西"):
        text = f"氏名: {surf}\n担当者一覧"
        s = text.index(surf)
        out = r.refine(text, _ents((s, s + len(surf), "PERSON")))
        assert [text[e.start:e.end] for e in out] == [surf], (surf, out)


def test_space_separated_name_full_width_space():
    # 全角空白でも同様に分断しない
    r = MorphologicalRefiner()
    surf = "かな　麻里"
    text = f"氏名: {surf}\n担当者一覧"
    s = text.index(surf)
    out = r.refine(text, _ents((s, s + len(surf), "PERSON")))
    assert [text[e.start:e.end] for e in out] == [surf]


def test_single_word_trim_still_applies():
    # 空白を含まないスパンでは従来どおり助詞・接尾語をトリムする
    r = MorphologicalRefiner()
    for text, surf, ty, expect in (("田中は大阪出身です。", "田中は", "PERSON", "田中"),
                                   ("佐藤さんに確認した。", "佐藤さん", "PERSON", "佐藤"),
                                   ("医療法人伊織会の理事長", "医療法人伊織会の", "ORG", "医療法人伊織会")):
        s = text.index(surf)
        out = r.refine(text, _ents((s, s + len(surf), ty)))
        assert [text[e.start:e.end] for e in out] == [expect], (surf, out)

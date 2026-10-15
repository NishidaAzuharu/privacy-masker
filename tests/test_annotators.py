"""識別性ベース匿名化（v0.8）: Annotatorステージ・PublicKnowledge・IdentifiabilityAnnotator・Policy preset。

設計: docs/design_identifiability.md。原則は「保持は正の証拠があるときだけ（fail-closed）」。
"""
import gzip

import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.annotators import (
    DEFAULT_PUBLIC_TITLES,
    Annotator,
    IdentifiabilityAnnotator,
    PublicKnowledge,
)
from privacy_masker.exceptions import AnnotatorError, ConfigurationError
from privacy_masker.operators import Keep, Tokenize
from privacy_masker.types import Entity
from tests.conftest import FakeDetector

KNOWN = [
    "安倍晋三", "岸田文雄", "ジル・バイデン", "新宿駅", "東京都庁", "トヨタ自動車", "豊田市",
    "奈良市", "大阪", "西新宿五丁目駅", "田中", "阿久津", "NDS", "ホワイトハウス",
]


@pytest.fixture
def knowledge():
    return PublicKnowledge.from_iterable(KNOWN)


def ents(text, *specs):
    """(surface, type) の列から text 内の位置を探して Entity を作る（テスト用）。"""
    out, pos = [], 0
    for surface, etype in specs:
        start = text.index(surface, pos)
        out.append(Entity(etype, start, start + len(surface), 0.9, "fake"))
        pos = start + len(surface)
    return out


def types(text, annotator, *specs):
    return [e.entity_type for e in annotator.annotate(text, ents(text, *specs))]


# ---------------------------------------------------------------- PublicKnowledge

def test_knowledge_contains_with_normalization(knowledge):
    assert "安倍晋三" in knowledge
    assert "岸田文雄" in knowledge
    assert "岸田 文雄" in knowledge  # 本文側の空白ゆれ
    assert "岸田　文雄" in knowledge  # 全角空白（NFKC）
    assert "ジル・バイデン" in knowledge
    assert "ＮＤＳ" in knowledge  # 全角英字（NFKC）
    assert "山田太郎" not in knowledge
    assert "" not in knowledge
    assert 123 not in knowledge
    assert len(knowledge) == len(KNOWN)


def test_knowledge_spaced_title_matches_compact_surface():
    k = PublicKnowledge.from_iterable(["山田 太郎"])
    assert "山田太郎" in k
    assert "山田 太郎" in k
    assert len(k) == 2  # 空白あり・なしの両方を登録


def test_knowledge_save_load_roundtrip(tmp_path, knowledge):
    p = tmp_path / "k.pmk"
    knowledge.save(p)
    loaded = PublicKnowledge.load(p)
    assert len(loaded) == len(knowledge)
    assert "新宿駅" in loaded
    assert "山田太郎" not in loaded


def test_knowledge_load_from_env(tmp_path, knowledge, monkeypatch):
    p = tmp_path / "k.pmk"
    knowledge.save(p)
    monkeypatch.setenv("PRIVACY_MASKER_KNOWLEDGE", str(p))
    assert "東京都庁" in PublicKnowledge.load()
    monkeypatch.delenv("PRIVACY_MASKER_KNOWLEDGE")
    with pytest.raises(ConfigurationError):
        PublicKnowledge.load()


def test_knowledge_load_rejects_garbage(tmp_path):
    p = tmp_path / "bad.pmk"
    p.write_bytes(b"nope")
    with pytest.raises(ConfigurationError):
        PublicKnowledge.load(p)
    p.write_bytes(b"PMK1" + (5).to_bytes(8, "little") + b"\x00" * 8)  # 件数と中身が不一致
    with pytest.raises(ConfigurationError):
        PublicKnowledge.load(p)


def test_knowledge_from_titles_file_gz_skips_header(tmp_path):
    p = tmp_path / "titles.txt.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("page_title\n新宿駅\n岸田_文雄\n\n")
    k = PublicKnowledge.from_titles_file(p)
    assert "新宿駅" in k
    assert "岸田 文雄" in k
    assert "岸田文雄" in k
    assert "page_title" not in k


# ---------------------------------------------------------------- IdentifiabilityAnnotator

def test_strict_level_is_passthrough(knowledge):
    ann = IdentifiabilityAnnotator(level="strict", knowledge=knowledge)
    text = "安倍元総理は奈良市で演説した。"
    es = ents(text, ("安倍", "PERSON"), ("奈良市", "LOCATION"))
    assert ann.annotate(text, es) == es


def test_public_title_makes_bare_surname_public(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "安倍元総理と岸田首相が会談し、山本部長と田中さんが同席した。"
    assert types(
        text, ann, ("安倍", "PERSON"), ("岸田", "PERSON"), ("山本", "PERSON"), ("田中", "PERSON")
    ) == ["PERSON_PUBLIC", "PERSON_PUBLIC", "PERSON", "PERSON"]


def test_public_title_inside_span(knowledge):
    text = "岸田首相は語った。"
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    assert types(text, ann, ("岸田首相", "PERSON")) == ["PERSON_PUBLIC"]


def test_title_rule_works_without_knowledge():
    ann = IdentifiabilityAnnotator(knowledge=None)
    text = "岸田首相と岸田文雄氏。"
    assert types(text, ann, ("岸田", "PERSON"), ("岸田文雄", "PERSON")) == ["PERSON_PUBLIC", "PERSON"]


def test_fullname_allowlist_requires_four_chars_or_separator(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "岸田文雄、ジル・バイデン、田中、阿久津の4名。"
    # 田中・阿久津は見出し語にあるが、裸姓・3文字は許可リストの対象外
    assert types(
        text, ann,
        ("岸田文雄", "PERSON"), ("ジル・バイデン", "PERSON"), ("田中", "PERSON"), ("阿久津", "PERSON"),
    ) == ["PERSON_PUBLIC", "PERSON_PUBLIC", "PERSON", "PERSON"]


def test_places_public_only_with_evidence(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "新宿駅の周辺には高層ビルが立ち並び、西口には東京都庁と株式会社カナメ精工がある。"
    assert types(
        text, ann, ("新宿駅", "FACILITY"), ("東京都庁", "ORG"), ("株式会社カナメ精工", "ORG")
    ) == ["FACILITY_PUBLIC", "ORG_PUBLIC", "ORG"]


def test_attribution_to_private_person_overrides_public(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "田中さんの最寄りは西新宿五丁目駅で、勤務先はトヨタ自動車です。"
    assert types(
        text, ann, ("田中", "PERSON"), ("西新宿五丁目駅", "FACILITY"), ("トヨタ自動車", "ORG")
    ) == ["PERSON", "FACILITY", "ORG"]


def test_attribution_needs_private_person_nearby(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge, window=10)
    text = "岸田首相の最寄りは新宿駅だ。"  # 公人しかいない → 帰属なし
    assert types(text, ann, ("岸田", "PERSON"), ("新宿駅", "FACILITY")) == [
        "PERSON_PUBLIC", "FACILITY_PUBLIC",
    ]
    far = "田中さんは元気だ。" + "あ" * 30 + "最寄りは新宿駅。"  # 私人が window の外
    assert types(far, ann, ("田中", "PERSON"), ("新宿駅", "FACILITY")) == ["PERSON", "FACILITY_PUBLIC"]


def test_field_label_counts_as_cue(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "氏名: 田中一郎\n住所: 豊田市"
    assert types(text, ann, ("田中一郎", "PERSON"), ("豊田市", "LOCATION")) == ["PERSON", "LOCATION"]


def test_short_ascii_acronym_is_not_public(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "NDSとの契約。"
    assert types(text, ann, ("NDS", "ORG")) == ["ORG"]


def test_unknown_place_stays_masked_even_without_private_person(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "田淵案件は松本チームが担当する。"  # 私人不在でも正の証拠なし → 匿名化
    assert types(text, ann, ("田淵案件", "ORG")) == ["ORG"]


def test_minimal_level_keeps_all_places_but_masks_private_persons(knowledge):
    ann = IdentifiabilityAnnotator(level="minimal", knowledge=knowledge)
    text = "田中さんの最寄りは西新宿五丁目駅で、勤務先は株式会社カナメ精工です。"
    assert types(
        text, ann, ("田中", "PERSON"), ("西新宿五丁目駅", "FACILITY"), ("株式会社カナメ精工", "ORG")
    ) == ["PERSON", "FACILITY_PUBLIC", "ORG_PUBLIC"]


def test_other_types_pass_through_unchanged(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "連絡先 a@example.com、住所 東京都新宿区西新宿4-15-8。"
    es = ents(text, ("a@example.com", "EMAIL"), ("東京都新宿区西新宿4-15-8", "ADDRESS"))
    out = ann.annotate(text, es)
    assert out == es
    assert out[0] is es[0] and out[1] is es[1]


def test_annotator_does_not_move_spans(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "安倍元総理は奈良市で演説した。"
    es = ents(text, ("安倍", "PERSON"), ("奈良市", "LOCATION"))
    out = ann.annotate(text, es)
    assert [(e.start, e.end, e.score, e.source) for e in out] == [
        (e.start, e.end, e.score, e.source) for e in es
    ]


def test_custom_titles_can_exclude_giin(knowledge):
    text = "蒼波議員が質問した。"
    assert types(text, IdentifiabilityAnnotator(knowledge=knowledge), ("蒼波", "PERSON")) == [
        "PERSON_PUBLIC",
    ]
    titles = tuple(t for t in DEFAULT_PUBLIC_TITLES if t != "議員")
    ann = IdentifiabilityAnnotator(knowledge=knowledge, public_titles=titles)
    assert types(text, ann, ("蒼波", "PERSON")) == ["PERSON"]


def test_invalid_config():
    with pytest.raises(ConfigurationError):
        IdentifiabilityAnnotator(level="loose")
    with pytest.raises(ConfigurationError):
        IdentifiabilityAnnotator(window=-1)
    with pytest.raises(ConfigurationError):
        IdentifiabilityAnnotator(public_suffix="")
    with pytest.raises(ConfigurationError):
        IdentifiabilityAnnotator(knowledge=42)


def test_is_annotator_protocol(knowledge):
    assert isinstance(IdentifiabilityAnnotator(knowledge=knowledge), Annotator)


# ---------------------------------------------------------------- Policy presets

def test_policy_presets_for_identifiability():
    p = Policy.preset("standard")
    for t in ("PERSON", "ORG", "LOCATION", "FACILITY"):
        assert isinstance(p.operator_for(t + "_PUBLIC"), Keep)
    assert isinstance(p.operator_for("PERSON"), Tokenize)  # 私人は default でトークン化
    assert isinstance(p.operator_for("ADDRESS"), Tokenize)
    strict = Policy.preset("strict")
    assert not strict.rules and isinstance(strict.default, Tokenize)
    custom = Policy.preset("minimal", public_types=("PERSON",))
    assert isinstance(custom.operator_for("PERSON_PUBLIC"), Keep)
    assert isinstance(custom.operator_for("ORG_PUBLIC"), Tokenize)


# ---------------------------------------------------------------- PrivacyEngine 統合

def make_engine(text, specs, policy, annotators):
    return PrivacyEngine(
        detectors=[FakeDetector(ents(text, *specs))],
        policy=policy, annotators=annotators, refiner=None,
    )


def test_engine_keeps_public_and_masks_private(knowledge):
    text = "安倍元総理は奈良市で演説し、田中さんは新宿駅で待った。"
    specs = [("安倍", "PERSON"), ("奈良市", "LOCATION"), ("田中", "PERSON"), ("新宿駅", "FACILITY")]
    eng = make_engine(text, specs, Policy.preset("standard"), [IdentifiabilityAnnotator(knowledge=knowledge)])
    r = eng.transform(text)
    assert r.text == "安倍元総理は奈良市で演説し、<PERSON_001>さんは新宿駅で待った。"
    assert [e.entity_type for e in r.entities] == [
        "PERSON_PUBLIC", "LOCATION_PUBLIC", "PERSON", "FACILITY_PUBLIC",
    ]
    assert [e.operator for e in r.entities] == ["keep", "keep", "tokenize", "keep"]


def test_engine_fail_closed_without_public_rules(knowledge):
    text = "安倍元総理が語った。"
    eng = make_engine(text, [("安倍", "PERSON")], Policy(), [IdentifiabilityAnnotator(knowledge=knowledge)])
    r = eng.transform(text)
    assert r.text == "<PERSON_PUBLIC_001>元総理が語った。"  # 規則が無ければ default に落ちる
    assert r.entities[0].entity_type == "PERSON_PUBLIC"


def test_engine_detect_and_batch_are_annotated(knowledge):
    text = "安倍元総理は奈良市で演説し、田中さんは新宿駅で待った。"
    specs = [("安倍", "PERSON"), ("奈良市", "LOCATION"), ("田中", "PERSON"), ("新宿駅", "FACILITY")]
    eng = make_engine(text, specs, Policy.preset("standard"), [IdentifiabilityAnnotator(knowledge=knowledge)])
    assert [e.entity_type for e in eng.detect(text)] == [
        "PERSON_PUBLIC", "LOCATION_PUBLIC", "PERSON", "FACILITY_PUBLIC",
    ]
    single = eng.transform(text)
    batch = eng.transform_batch([text, "", text])
    assert batch[0].text == single.text == batch[2].text
    assert batch[1].text == ""


def test_engine_without_annotators_is_unchanged():
    text = "安倍元総理が語った。"
    eng = PrivacyEngine(detectors=[FakeDetector(ents(text, ("安倍", "PERSON")))], refiner=None)
    assert eng.annotators == []
    assert eng.transform(text).text == "<PERSON_001>元総理が語った。"


def test_engine_rejects_object_without_annotate():
    with pytest.raises(ConfigurationError):
        PrivacyEngine(detectors=[FakeDetector([])], annotators=[object()])


class _OutOfBoundsAnnotator:
    name = "oob"

    def annotate(self, text, entities):
        return [Entity("PERSON", 0, len(text) + 5, 0.5, "oob")]


class _MovingAnnotator:
    name = "moving"

    def annotate(self, text, entities):
        return [Entity("", e.start, e.end, e.score, e.source) for e in entities]


class _CrashingAnnotator:
    name = "crash"

    def annotate(self, text, entities):
        raise ValueError(text)  # 例外に原文が乗っても外へは出さない


def test_engine_validates_annotator_output():
    text = "田中さん"
    for bad in (_OutOfBoundsAnnotator(), _MovingAnnotator()):
        eng = PrivacyEngine(
            detectors=[FakeDetector(ents(text, ("田中", "PERSON")))], annotators=[bad], refiner=None,
        )
        with pytest.raises(AnnotatorError):
            eng.transform(text)
    eng = PrivacyEngine(detectors=[FakeDetector([])], annotators=[_CrashingAnnotator()], refiner=None)
    with pytest.raises(AnnotatorError) as ei:
        eng.transform(text)
    assert "田中" not in str(ei.value)
    assert ei.value.__cause__ is None


def test_place_public_check_ignores_legal_forms(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "トヨタ自動車株式会社と株式会社カナメ精工の面談。"
    assert types(text, ann, ("トヨタ自動車株式会社", "ORG"), ("株式会社カナメ精工", "ORG")) == [
        "ORG_PUBLIC", "ORG",
    ]


def test_split_compound_fragment_is_not_public_when_adjacent_to_private_place(knowledge):
    # NER が「株式会社アオバ情報システム」を 株式会社アオバ | 情報システム に分割し、
    # 一般名詞「情報システム」が見出し語に一致しても、隣接する非公知 ORG の一部として扱う
    k = PublicKnowledge.from_iterable(list(KNOWN) + ["情報システム", "リース"])
    ann = IdentifiabilityAnnotator(knowledge=k)
    text = "委託先は株式会社アオバ情報システムとハシバ事務機 リース株式会社です。"
    assert types(text, ann, ("株式会社アオバ", "ORG"), ("情報システム", "ORG"),
                 ("ハシバ事務機", "ORG"), ("リース株式会社", "ORG")) == ["ORG", "ORG", "ORG", "ORG"]
    # 離れていれば通常どおり公知
    text2 = "情報システムの刷新について、株式会社アオバと協議した。"
    assert types(text2, ann, ("情報システム", "ORG"), ("株式会社アオバ", "ORG")) == ["ORG_PUBLIC", "ORG"]


def test_adjacent_public_places_stay_public(knowledge):
    ann = IdentifiabilityAnnotator(knowledge=knowledge)
    text = "新宿駅 東京都庁 方面へ。"
    assert types(text, ann, ("新宿駅", "FACILITY"), ("東京都庁", "ORG")) == ["FACILITY_PUBLIC", "ORG_PUBLIC"]


# ------------------------------------------------- 住所成分ガード（ADDRESS/LOCATION 型衝突）
# NER は「番地があれば ADDRESS、なければ LOCATION」で学習している（学習データ実測: ADDRESS
# 4,698 件の 100% が番地付き / LOCATION 6,075 件の 99.98% が番地なし）。そのため住所の町名部分は
# LOCATION として出力され、jawiki 見出し語に一致して _PUBLIC = Keep になり平文で残る。
# 4 評価セット実測（standard / token30m）で本物の住所 122 文字が漏れていた。

ADDR_KNOWN = ["野田町", "壬生辻町", "豊田市", "西新宿", "アオバビル", "アオバ", "新宿区"]


@pytest.fixture
def addr_knowledge():
    return PublicKnowledge.from_iterable(ADDR_KNOWN)


@pytest.fixture
def addr_ann(addr_knowledge):
    return IdentifiabilityAnnotator(level="standard", knowledge=addr_knowledge)


def test_address_guard_banchi_before(addr_ann):
    # 「住所: 4-22-10 野田町」— 直前が番地
    assert types("住所: 4-22-10 野田町に届けてください", addr_ann, ("野田町", "LOCATION")) == ["LOCATION"]


def test_address_guard_banchi_after(addr_ann):
    # 「壬生辻町 3-2-1」— 直後が番地（町名 → 番地の順）
    assert types("壬生辻町 3-2-1 で待ち合わせ", addr_ann, ("壬生辻町", "LOCATION")) == ["LOCATION"]


def test_address_guard_postal_code(addr_ann):
    # 〒が近傍（帳票では別行にある）
    assert types("〒791-1234\n野田町\n電話 03-0000-0000", addr_ann,
                 ("野田町", "LOCATION")) == ["LOCATION"]


def test_address_guard_field_label(addr_ann):
    assert types("所在地: 野田町", addr_ann, ("野田町", "LOCATION")) == ["LOCATION"]


def test_address_guard_adjacent_address_entity(addr_ann):
    # 隣接する ADDRESS エンティティが住所文脈の証拠になる
    text = "移転先は野田町の1-2-2号室です"
    assert types(text, addr_ann, ("野田町", "LOCATION"), ("1-2-2号室", "ADDRESS")) == \
        ["LOCATION", "ADDRESS"]


def test_address_guard_surface_contains_chome(addr_ann):
    # 表層自体に丁目があれば文脈を見ずに住所成分（市区町村の例外より優先）
    assert types("西新宿五丁目1番1号", addr_ann, ("西新宿五丁目1番1号", "LOCATION")) == ["LOCATION"]


def test_address_guard_keeps_city_level(addr_ann):
    """市区町村以上は住所文脈でも公知のまま（Surrogate(kind="address", keep="city") と同粒度）。"""
    assert types("住所: 豊田市 4-22-10", addr_ann, ("豊田市", "LOCATION")) == ["LOCATION_PUBLIC"]


def test_address_guard_does_not_touch_plain_place_reference(addr_ann):
    """住所文脈が無ければ従来どおり公知。ニュース記事の地名参照を過剰マスクしない。"""
    assert types("野田町の街並みを撮影した", addr_ann, ("野田町", "LOCATION")) == ["LOCATION_PUBLIC"]


def test_address_guard_building_in_address(addr_ann):
    # 住所内の建物名（FACILITY）も番地が直前にあれば住所成分
    assert types("西新宿4-15-8 アオバビル 501", addr_ann, ("アオバビル", "FACILITY")) == ["FACILITY"]


def test_address_guard_exempts_legal_form_org(addr_ann):
    # 「株式会社アオバ」は番地の隣にあっても住所の構成要素ではない
    assert types("西新宿4-15-8 株式会社アオバ", addr_ann,
                 ("株式会社アオバ", "ORG")) == ["ORG_PUBLIC"]


def test_address_guard_applies_at_minimal_level(addr_knowledge):
    """minimal は LOCATION を全部 _PUBLIC にするが、住所成分は例外（直接識別子）。"""
    ann = IdentifiabilityAnnotator(level="minimal", knowledge=addr_knowledge)
    assert types("住所: 4-22-10 野田町", ann, ("野田町", "LOCATION")) == ["LOCATION"]
    assert types("野田町の街並み", ann, ("野田町", "LOCATION")) == ["LOCATION_PUBLIC"]


def test_address_guard_can_be_disabled(addr_knowledge):
    ann = IdentifiabilityAnnotator(level="standard", knowledge=addr_knowledge, address_guard=False)
    assert types("住所: 4-22-10 野田町", ann, ("野田町", "LOCATION")) == ["LOCATION_PUBLIC"]


def test_address_guard_not_applied_at_strict_level(addr_knowledge):
    ann = IdentifiabilityAnnotator(level="strict", knowledge=addr_knowledge)
    assert types("住所: 4-22-10 野田町", ann, ("野田町", "LOCATION")) == ["LOCATION"]


def test_address_guard_gap_does_not_cross_newline(addr_ann):
    """改行を挟んだら別ブロック。番地の直後の改行後の地名まで住所成分にはしない。"""
    assert types("4-22-10\n\n野田町の観光案内", addr_ann, ("野田町", "LOCATION")) == ["LOCATION_PUBLIC"]

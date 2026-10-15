from privacy_masker.engine import PrivacyEngine
from privacy_masker.operators import Tokenize
from privacy_masker.policy import Policy
from privacy_masker.refiner import SpanRefiner
from privacy_masker.types import Entity

from .conftest import FakeCrypto, FakeDetector


def E(t, s, e, score=1.0):
    return Entity(entity_type=t, start=s, end=e, score=score, source="test")


def spans(text, ents, **kw):
    return [(x.start, x.end, x.entity_type) for x in SpanRefiner(**kw).refine(text, ents)]


# ---- 空白トリム ----

def test_trims_surrounding_whitespace():
    text = "住所\n碇谷 です"
    assert spans(text, [E("PERSON", 2, 5)]) == [(3, 5, "PERSON")]   # "\n碇谷" -> "碇谷"
    assert spans(text, [E("PERSON", 3, 6)]) == [(3, 5, "PERSON")]   # "碇谷 " -> "碇谷"


def test_drops_span_that_becomes_empty():
    text = "a \n b"
    assert spans(text, [E("PERSON", 1, 4)]) == []


# ---- 敬称トリム（PERSONのみ） ----

def test_trims_trailing_honorific_for_person():
    text = "田中さんと佐藤様"
    assert spans(text, [E("PERSON", 0, 4)]) == [(0, 2, "PERSON")]
    assert spans(text, [E("PERSON", 5, 8)]) == [(5, 7, "PERSON")]


def test_honorific_not_trimmed_for_org():
    text = "くらしさん株式会社"
    # 敬称「さん」は ORG では削らない。v0.8 の語尾拡張は隣接する「株式会社」を取り込む
    assert spans(text, [E("ORG", 0, 5)], expand_affixes=False) == [(0, 5, "ORG")]
    assert spans(text, [E("ORG", 0, 5)]) == [(0, 9, "ORG")]


def test_honorific_only_span_is_kept():
    text = "さんです"
    assert spans(text, [E("PERSON", 0, 2)]) == [(0, 2, "PERSON")]


# ---- 助詞トリム（非ひらがなアンカーのみ） ----

def test_trims_trailing_particle_after_kanji_or_katakana():
    text = "日本の未来"
    assert spans(text, [E("LOCATION", 0, 3)]) == [(0, 2, "LOCATION")]
    text2 = "ユニクロの店"
    assert spans(text2, [E("ORG", 0, 5)]) == [(0, 4, "ORG")]


def test_trims_trailing_particle_run():
    text = "日本では有名"
    assert spans(text, [E("LOCATION", 0, 4)]) == [(0, 2, "LOCATION")]


def test_keeps_trailing_kana_when_anchor_is_hiragana():
    # 「りの」のようなひらがな終わりの名前を誤って削らない
    text = "りのです"
    assert spans(text, [E("PERSON", 0, 2)]) == [(0, 2, "PERSON")]


def test_trims_leading_particle():
    text = "対の大韓民国が"
    assert spans(text, [E("LOCATION", 1, 6)]) == [(2, 6, "LOCATION")]


def test_particle_trim_is_hard_capped_across_passes():
    # fixpointループを跨いでも累積で片側 _MAX_PARTICLE_RUN(=2) 文字までしか削らない
    # （カスタムparticlesで非ひらがな文字を指定しても、スパン深部まで削れない）
    text = "A11111本日"   # 直後に内容語（文末・句読点だと v0.8 の文脈ガードで末尾助詞を残す）
    got = SpanRefiner(particles="1").refine(text, [E("ORG", 0, 6)])
    assert [(x.start, x.end) for x in got] == [(0, 4)]   # "A111" まで（"A"まで削れたらNG）


def test_trims_head_and_tail_in_single_call():
    text = "対の日本の下"
    assert spans(text, [E("LOCATION", 1, 5)]) == [(2, 4, "LOCATION")]  # "の日本の" -> "日本"


# ---- 設定で無効化 ----

def test_all_rules_disabled_leaves_spans_unchanged():
    text = " 田中さんの "
    r = SpanRefiner(trim_whitespace=False, honorifics=(), particles="")
    got = r.refine(text, [E("PERSON", 0, 7)])
    assert [(x.start, x.end) for x in got] == [(0, 7)]


def test_refine_preserves_other_fields():
    text = "田中さん"
    got = SpanRefiner().refine(text, [E("PERSON", 0, 4, score=0.7)])
    assert got[0].score == 0.7 and got[0].source == "test"


# ---- エンジン統合 ----

def test_engine_refines_by_default():
    text = "田中さんが来た"
    engine = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 4)])])
    result = engine.transform(text)
    assert result.text == "<PERSON_001>さんが来た"   # さんはスパン外に残る


def test_engine_detect_returns_refined_spans():
    text = "田中さんが来た"
    engine = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 4)])])
    assert [(e.start, e.end) for e in engine.detect(text)] == [(0, 2)]


def test_engine_refiner_none_disables():
    text = "田中さんが来た"
    engine = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 4)])], refiner=None)
    assert engine.transform(text).text == "<PERSON_001>が来た"


def test_engine_batch_applies_refiner():
    text = "田中さんです"
    engine = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 4)])])
    assert engine.transform_batch([text])[0].text == "<PERSON_001>さんです"


def test_vault_token_consistent_across_honorific_variants():
    # 敬称の境界ゆれ（百舌鳥孝一さん vs 百舌鳥孝一）でも同一人物=同一トークン
    doc1, doc2 = "百舌鳥孝一さんの記録", "百舌鳥孝一の記録"
    policy = Policy(rules={}, default=Tokenize(strategy="vault"))
    e1 = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 7)])], policy=policy,
                       crypto=FakeCrypto(), secret_key=b"k" * 16)
    r1 = e1.transform(doc1, scope_id="s1")
    e2 = PrivacyEngine(detectors=[FakeDetector([E("PERSON", 0, 5)])], policy=policy,
                       crypto=FakeCrypto(), secret_key=b"k" * 16, store=e1.store)
    r2 = e2.transform(doc2, scope_id="s1")
    assert r1.entities[0].replacement == r2.entities[0].replacement
    # 復元は各文書の表層形どおり（doc1のさんはスパン外なのでそのまま残る）
    assert e1.restore(r1.text, scope_id="s1").text == doc1
    assert e2.restore(r2.text, scope_id="s1").text == doc2


# ---- 複数文字の機能語トリム（Surrogate置換の自然さ対策） ----

def test_trims_trailing_function_words():
    # 「大阪から」→大阪 のように機能語を巻き込んだスパンを補正する
    # （Surrogateで全体置換すると助詞・機能語が消えて不自然になるため）
    text1 = "大阪から出荷します"
    assert spans(text1, [E("LOCATION", 0, 4)]) == [(0, 2, "LOCATION")]
    text2 = "矢吹町にある工場"
    assert spans(text2, [E("LOCATION", 0, 6)]) == [(0, 3, "LOCATION")]
    text3 = "柏市まで配送"
    assert spans(text3, [E("LOCATION", 0, 4)]) == [(0, 2, "LOCATION")]
    text4 = "東京より安い"
    assert spans(text4, [E("LOCATION", 0, 4)]) == [(0, 2, "LOCATION")]


def test_function_word_kept_when_anchor_is_hiragana():
    # ひらがな終わりの固有名詞を誤って削らない（「みなと」+まで）
    text = "みなとまで歩く"
    assert spans(text, [E("LOCATION", 0, 5)]) == [(0, 5, "LOCATION")]


def test_function_word_after_kanji_anchor():
    # 4文字機能語（において）もアンカーが漢字なら削れる
    text = "日本において主流だ"
    assert spans(text, [E("LOCATION", 0, 6)]) == [(0, 2, "LOCATION")]


def test_chained_hiragana_function_words_kept_safe_side():
    # 「などでは」のようにひらがなが連鎖するとアンカー規則により削らない（安全側の既知限界）
    text = "日本などでは主流だ"
    assert spans(text, [E("LOCATION", 0, 6)]) == [(0, 6, "LOCATION")]


# ---- v0.8: 末尾助詞の文脈ガードと、ひらがな名の先頭助詞 ----

def test_tail_particle_kept_before_honorific_punctuation_or_newline():
    for text, name in [("後任は篠原秋も氏です。", "篠原秋も"), ("研究分担者 金子かの、殿崎", "金子かの"),
                       ("人事部\n石川まりの\nkashp", "石川まりの"), ("記録者 矢野ねねの", "矢野ねねの")]:
        s = text.index(name)
        assert spans(text, [E("PERSON", s, s + len(name))]) == [(s, s + len(name), "PERSON")]


def test_tail_particle_still_trimmed_before_content_word():
    text = "田中は本日休みです。"
    assert spans(text, [E("PERSON", 0, 3)]) == [(0, 2, "PERSON")]


def test_head_particle_kept_for_all_hiragana_name_or_after_particle():
    for text, name in [("はすみさんに連絡", "はすみ"), ("担当ははすみです。", "はすみ"), ("記入 はなさん", "はな")]:
        s = text.index(name)
        assert spans(text, [E("PERSON", s, s + len(name))]) == [(s, s + len(name), "PERSON")]
    assert spans("の田中が担当", [E("PERSON", 0, 3)]) == [(1, 3, "PERSON")]

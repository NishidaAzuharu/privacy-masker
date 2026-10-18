"""正規表現の番号類（CREDIT_CARD・MY_NUMBER・CORPORATE_NUMBER）の誤検出対策:
文脈語の単語照合、カード番号の先頭桁と境界、文脈語の無い数字列の中間スコア、URL・メールの内側の数字の吸収。"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.context_words import compile_context_words
from privacy_masker.detectors import RegexDetector
from privacy_masker.detectors.regex import CHECKSUM_ONLY_SCORE, corporate_check_digit


def found(text, etype):
    det = RegexDetector.default_ja()
    return [(text[e.start:e.end], e.score) for e in det.detect(text) if e.entity_type == etype]


def corp(body12="234567890123"):
    return f"{corporate_check_digit(body12)}{body12}"


def engine():
    return PrivacyEngine(detectors=[RegexDetector.default_ja()], policy=Policy.preset("strict"), refiner=None)


# ---- 1. 文脈語は英数字だけなら単語として照合（T が test・https に反応しない）

def test_context_word_T_matches_only_as_a_word():
    num = corp()
    assert found(f"番号は {num} です。test", "CORPORATE_NUMBER") == [(num, CHECKSUM_ONLY_SCORE)]
    assert found(f"https://x.example {num}", "CORPORATE_NUMBER") == [(num, CHECKSUM_ONLY_SCORE)]
    assert found(f"Total {num}", "CORPORATE_NUMBER") == [(num, CHECKSUM_ONLY_SCORE)]
    assert found(f"T{num}", "CORPORATE_NUMBER") == [(num, 1.0)]            # 適格請求書の登録番号 T + 13 桁
    assert found(f"登録番号 T {num}", "CORPORATE_NUMBER") == [(num, 1.0)]
    assert found(f"法人番号 {num}", "CORPORATE_NUMBER") == [(num, 1.0)]
    assert found(f"ST{num}", "CORPORATE_NUMBER") == []                      # 英字直結の型番は番号としない


def test_compile_context_words_rule():
    # 窓は NFKC・casefold 済みの前提（呼び出し側が normalize_context する）
    m = compile_context_words(("T", "tel", "法人番号", "my number"))
    assert m.search("登録番号 t 123") and m.search("tel:") and m.search("x法人番号y") and m.search("my number is")
    assert not m.search("test") and not m.search("https") and not m.search("hotel") and not m.search("telephone")
    assert compile_context_words(()) is None


# ---- 3. カード番号: 主要ブランドの先頭桁 + Luhn、英字・_・/・小数点が直結したら番号としない

@pytest.mark.parametrize("num", ["4111111111111111", "4222222222222", "378282246310005", "3530111333300000",
                                 "5555555555554444", "2223003122003222", "30569309025904", "6011111111111117",
                                 "6200000000000005"])
def test_card_brands_are_detected(num):
    assert found(f"番号 {num}", "CREDIT_CARD") == [(num, CHECKSUM_ONLY_SCORE)]


@pytest.mark.parametrize("num", ["1234567890128", "7987654321098762", "9234567890123455", "352099001761481"])
def test_luhn_valid_but_not_a_card_prefix(num):
    # Luhn は合うが主要ブランドの先頭桁でない（伝票番号・IMEI など）
    assert found(f"番号 {num}", "CREDIT_CARD") == []


@pytest.mark.parametrize("text", ["x4111111111111111", "4111111111111111x", "ID_4111111111111111", "/4111111111111111",
                                  "0.4111111111111111", "4111111111111111.5", "v9.4111111111111111", "4111111111111111/2"])
def test_card_not_matched_when_glued_to_letters_or_paths(text):
    assert found(text, "CREDIT_CARD") == []


@pytest.mark.parametrize("text", ["カード番号4111111111111111", "No.4111111111111111", "合計4111111111111111.", "（4111111111111111）"])
def test_card_boundaries_still_allow_japanese_text_and_sentence_end(text):
    assert [v for v, _ in found(text, "CREDIT_CARD")] == ["4111111111111111"]


def test_my_number_boundaries():
    assert found("123456789018/", "MY_NUMBER") == [] and found("_123456789018", "MY_NUMBER") == []
    assert found("x123456789018", "MY_NUMBER") == [] and found("123456789018.5", "MY_NUMBER") == []
    assert [v for v, _ in found("番号は123456789018。", "MY_NUMBER")] == ["123456789018"]


# ---- 4. 文脈語の無い数字列は CHECKSUM_ONLY_SCORE、文脈語があれば 1.0。既定（閾値 0）では両方マスク

@pytest.mark.parametrize("etype, with_ctx, without_ctx, num", [
    ("CREDIT_CARD", "カード 4111 1111 1111 1111", "番号 4111 1111 1111 1111", "4111 1111 1111 1111"),
    ("CREDIT_CARD", "Visa 4111111111111111", "伝票 4111111111111111", "4111111111111111"),
    ("MY_NUMBER", "個人番号 123456789018", "伝票番号 123456789018", "123456789018"),
    ("MY_NUMBER", "マイナンバー 1234-5678-9018", "受付 1234-5678-9018", "1234-5678-9018"),
    ("CORPORATE_NUMBER", f"法人番号 {corp()}", f"番号 {corp()}", corp()),
])
def test_context_words_decide_the_score(etype, with_ctx, without_ctx, num):
    assert found(with_ctx, etype) == [(num, 1.0)]
    assert found(without_ctx, etype) == [(num, CHECKSUM_ONLY_SCORE)]


def test_threshold_keeps_only_numbers_with_context():
    eng = engine()
    plain, ctx = "伝票番号 123456789018 を発行", "個人番号 123456789018 を確認"
    assert eng.transform(plain).text == "伝票番号 <MY_NUMBER_001> を発行"                   # 既定は今まで通り全部マスク
    assert eng.transform(plain, score_threshold=0.6).text == plain                            # 文脈語なしは残る
    assert eng.transform(plain, score_threshold={"MY_NUMBER": 0.6}).text == plain
    assert eng.transform(ctx, score_threshold=0.6).text == "個人番号 <MY_NUMBER_001> を確認"


# ---- 2. URL・メールの内側の数字は URL / EMAIL として丸ごと扱う（分断しない）

@pytest.mark.parametrize("text, expected, types", [
    ("https://example.com/?card=4111111111111111 を参照", "<URL_001> を参照", ["URL"]),
    ("123456789018@example.com に送付", "<EMAIL_001> に送付", ["EMAIL"]),
    ("https://example.com/?to=a@b.example", "<URL_001>", ["URL"]),
])
def test_numbers_inside_url_or_email_are_absorbed(text, expected, types):
    r = engine().transform(text)
    assert r.text == expected and [e.entity_type for e in r.entities] == types


def test_number_inside_url_is_still_detected_by_the_regex_itself():
    # 吸収は EntityResolver の仕事（検出器は ?card= の 16 桁を CREDIT_CARD として返す）
    assert [v for v, _ in found("https://example.com/?card=4111111111111111", "CREDIT_CARD")] == ["4111111111111111"]

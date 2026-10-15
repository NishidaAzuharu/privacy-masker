import pytest

from privacy_masker.detectors import RegexDetector


def types_at(text):
    det = RegexDetector.default_ja()
    return {(e.entity_type, text[e.start:e.end]) for e in det.detect(text)}


def test_email():
    assert ("EMAIL", "taro@example.co.jp") in types_at("連絡先: taro@example.co.jp まで")


def test_phone_numbers():
    assert ("PHONE_NUMBER", "03-1234-5678") in types_at("電話は03-1234-5678です")
    assert ("PHONE_NUMBER", "090-1234-5678") in types_at("携帯090-1234-5678")
    assert ("PHONE_NUMBER", "(03)1234-5678") in types_at("代表(03)1234-5678")


def test_postal_code():
    assert ("POSTAL_CODE", "〒160-0023") in types_at("〒160-0023 東京都新宿区")
    assert ("POSTAL_CODE", "160-0023") in types_at("郵便番号は160-0023")


def test_postal_not_matched_inside_phone():
    found = types_at("電話は03-1234-5678です")
    assert not any(t == "POSTAL_CODE" for t, _ in found)


def test_credit_card_luhn():
    assert ("CREDIT_CARD", "4111111111111111") in types_at("カード番号4111111111111111")
    assert ("CREDIT_CARD", "4111 1111 1111 1111") in types_at("番号: 4111 1111 1111 1111")
    # Luhn不成立は検出しない
    assert not types_at("番号4111111111111112")


def test_ip_address():
    assert ("IP_ADDRESS", "192.168.1.1") in types_at("接続元192.168.1.1から")
    assert not any(t == "IP_ADDRESS" for t, _ in types_at("バージョン300.168.1.1"))


def test_my_number():
    # チェックデジット計算: 123456789018 は有効（末尾8が検査用数字）
    assert ("MY_NUMBER", "123456789018") in types_at("個人番号123456789018")
    assert not any(t == "MY_NUMBER" for t, _ in types_at("番号123456789019"))


def test_credit_card_not_joined_across_unrelated_numbers():
    # 無関係な数値の隣接を結合してマッチしない（レビュー指摘の再現ケース）
    assert not any(t == "CREDIT_CARD" for t, _ in types_at("在庫は99900 10000456個です"))
    assert not any(t == "CREDIT_CARD" for t, _ in types_at("型番99900-10000456番"))


def test_credit_card_formats():
    assert ("CREDIT_CARD", "4111-1111-1111-1111") in types_at("番号: 4111-1111-1111-1111")
    assert ("CREDIT_CARD", "3782 822463 10005") in types_at("AmEx: 3782 822463 10005")


def test_score_and_source():
    det = RegexDetector.default_ja()
    ents = det.detect("taro@example.com")
    assert ents[0].score == 1.0
    assert ents[0].source == "regex"


# ---- 盲検評価 v1 で見つかった取りこぼし

def _types(text, etype):
    det = RegexDetector.default_ja()
    return [text[e.start:e.end] for e in det.detect(text) if e.entity_type == etype]


@pytest.mark.parametrize("text, expected", [
    ("次回連絡は０９０－１２３４－５６７８へ", ["０９０－１２３４－５６７８"]),          # 全角数字・全角ハイフン
    ("お客様: ０９０の１２３４の５６７８です。", ["０９０の１２３４の５６７８"]),      # 読み上げの「の」区切り
    ("電話 03 1234 5678 まで", ["03 1234 5678"]),                                   # 空白区切り
    ("My mobile is +44 7700 900123.", ["+44 7700 900123"]),                         # 国際番号（空白）
    ("Tel: +81-3-1234-5678", ["+81-3-1234-5678"]),                                  # 国際番号（ハイフン）
    ("王同学（+886 912 345 678）", ["+886 912 345 678"]),
    ("TEL 048-222-7815 / FAX 048-222-7816", ["048-222-7815", "048-222-7816"]),      # 既存形式は変わらない
])
def test_phone_formats_from_blind_eval(text, expected):
    assert _types(text, "PHONE_NUMBER") == expected


def test_phone_space_variant_requires_10_or_11_digits():
    assert _types("在庫 03 12 345 です", "PHONE_NUMBER") == []      # 9 桁
    assert _types("番号 0 1234 5678 9012", "PHONE_NUMBER") == []   # 続きの数字がある
    assert _types("合計 +3 45 678 円", "PHONE_NUMBER") == []        # 短い


def test_ip_before_period_and_my_number_with_spaces():
    assert _types("5 failed attempts from 198.51.100.201. I have", "IP_ADDRESS") == ["198.51.100.201"]
    assert _types("バージョン 1.2.3.4.5 ではなく", "IP_ADDRESS") == []
    assert _types("マイナンバー（1234 5678 9018）を含む", "MY_NUMBER") == ["1234 5678 9018"]
    assert _types("マイナンバー 1234-5678-9018", "MY_NUMBER") == ["1234-5678-9018"]
    assert _types("マイナンバー 1234 5678 9012", "MY_NUMBER") == []          # 検査用数字が合わない
    # 16 桁カード番号の先頭 12 桁を個人番号として取らない
    assert _types("カード 4111 1111 1111 1111", "MY_NUMBER") == []
    assert _types("カード 4111 1111 1111 1111", "CREDIT_CARD") == ["4111 1111 1111 1111"]


def test_postal_code_fullwidth_hyphen():
    assert _types("〒３２１－０９５３ 栃木県", "POSTAL_CODE") == ["〒３２１－０９５３"]

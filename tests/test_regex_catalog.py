"""v0.8 の正規表現カタログ拡張（日本固有識別子・URL・IPv6・MAC・DATE/AGE）と文脈スコア、ContextEnhancer。"""
import pytest

from privacy_masker import ContextEnhancer, PrivacyEngine, Policy
from privacy_masker.detectors import RegexDetector
from privacy_masker.detectors.regex import corporate_check_digit
from privacy_masker.types import Entity
from tests.conftest import FakeDetector


def found(text, etype, **kw):
    det = RegexDetector.default_ja(**kw)
    return [(text[e.start:e.end], round(e.score, 2)) for e in det.detect(text) if e.entity_type == etype]


def test_corporate_number_check_digit_and_context():
    body = "000012050002"
    num = f"{corporate_check_digit(body)}{body}"
    assert found(f"法人番号 {num} の会社", "CORPORATE_NUMBER") == [(num, 1.0)]
    assert found(f"番号は {num} です", "CORPORATE_NUMBER") == [(num, 0.7)]       # 文脈なしは基本スコア
    bad = f"{(corporate_check_digit(body) + 1) % 10}{body}"
    assert found(f"法人番号 {bad}", "CORPORATE_NUMBER") == []                       # 検査用数字が合わない


def test_drivers_license_requires_context_and_does_not_shadow_my_number():
    assert found("免許証番号 123456789012 を確認", "DRIVERS_LICENSE") == [("123456789012", 0.9)]
    assert found("受付番号 123456789012", "DRIVERS_LICENSE") == []                    # 文脈なしは出さない
    # 12 桁の有効なマイナンバーは MY_NUMBER でも検出され、EntityResolver で MY_NUMBER が勝つ
    eng = PrivacyEngine(detectors=[RegexDetector.default_ja()], policy=Policy.preset("strict"), refiner=None)
    r = eng.transform("免許証の代わりにマイナンバー 1234 5678 9018 を提示")
    assert [e.entity_type for e in r.entities] == ["MY_NUMBER"]


def test_passport_and_residence_card():
    assert found("旅券番号 TK1234567", "PASSPORT") == [("TK1234567", 1.0)]
    assert found("コード TK1234567", "PASSPORT") == [("TK1234567", 0.6)]
    assert found("型番 TK12345678", "PASSPORT") == []                                  # 桁が違う
    assert found("在留カード番号 AB12345678CD", "RESIDENCE_CARD") == [("AB12345678CD", 1.0)]
    assert found("AB12345678CD", "RESIDENCE_CARD") == [("AB12345678CD", 0.85)]


def test_pension_and_bank_account_need_context():
    assert found("基礎年金番号 1234-567890", "PENSION_NUMBER") == [("1234-567890", 1.0)]
    assert found("品番 1234-567890", "PENSION_NUMBER") == []
    assert found("振込先: ムサシノ銀行 船橋支店 普通 1187224", "BANK_ACCOUNT") == [("1187224", 0.9)]
    assert found("受理番号 1187224", "BANK_ACCOUNT") == []


def test_url_ipv6_mac():
    assert found("詳細は https://track.yamano.example.jp/3319?x=1 を参照。", "URL") == [("https://track.yamano.example.jp/3319?x=1", 1.0)]
    assert found("サイト www.example.co.jp/path です", "URL") == [("www.example.co.jp/path", 1.0)]
    assert found("メール a@b.example.com", "URL") == []                                 # メールは URL にしない
    ips = found("接続元 2001:db8::1 と fe80::1ff:fe23:4567:890a、時刻 12:30:45", "IP_ADDRESS")
    assert [v for v, _ in ips] == ["2001:db8::1", "fe80::1ff:fe23:4567:890a"]
    assert found("MAC 00:1A:2B:3C:4D:5E / 00-1a-2b-3c-4d-5e", "MAC_ADDRESS") == [("00:1A:2B:3C:4D:5E", 1.0), ("00-1a-2b-3c-4d-5e", 1.0)]


def test_dates_and_age_are_opt_in():
    text = "生年月日 1990年3月3日。入社は 2015/04/01 で令和7年9月1日 現在も在籍。34歳（年齢 満34歳）"
    assert found(text, "DATE") == [] and found(text, "AGE") == []                      # 既定では検出しない
    assert found("生年月日 1990年3月3日", "DATE", dates=True) == [("1990年3月3日", 1.0)]  # 生年月日文脈で 1.0
    assert found("入社は 2015/04/01 です", "DATE", dates=True) == [("2015/04/01", 0.5)]
    assert found("令和7年9月1日 現在", "DATE", dates=True) == [("令和7年9月1日", 0.5)]
    assert [v for v, _ in found(text, "DATE", dates=True)] == ["1990年3月3日", "2015/04/01", "令和7年9月1日"]
    a = found(text, "AGE", dates=True)
    assert [v for v, _ in a] == ["34歳", "満34歳"] and a[1][1] == 1.0


def test_new_types_do_not_change_existing_detections():
    text = "TEL 048-222-7815 / r.miwa@shinonome-lg.example.jp 〒343-0824 カード 4111 1111 1111 1111"
    det = RegexDetector.default_ja()
    types = sorted(e.entity_type for e in det.detect(text))
    assert types == ["CREDIT_CARD", "EMAIL", "PHONE_NUMBER", "POSTAL_CODE"]


def test_context_enhancer_boosts_scores_near_cue_words():
    text = "氏名: 田中太郎  住所: 東京都新宿区西新宿4-15-8"
    ents = [Entity("PERSON", 4, 8, 0.5, "ner"), Entity("ADDRESS", 14, 30, 0.5, "ner"), Entity("ORG", 4, 8, 0.5, "ner")]
    out = ContextEnhancer().enhance(text, ents)
    assert [round(e.score, 2) for e in out] == [0.8, 0.8, 0.5]                          # ORG には手掛かり語がない
    custom = ContextEnhancer(rules={"PROJECT": ["案件"]}, boost=0.5, extend_default=False)
    e = Entity("PROJECT", 2, 8, 0.4, "x")
    assert custom.enhance("案件 ハヤブサ計画", [e])[0].score == 0.9
    assert custom.enhance("氏名 ハヤブサ計画", [e])[0].score == 0.4


def test_enhancer_in_engine_with_threshold():
    text = "氏名: 田中太郎。担当は佐藤。"
    ents = [Entity("PERSON", 4, 8, 0.45, "ner"), Entity("PERSON", 13, 15, 0.45, "ner")]
    eng = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy.preset("strict"), refiner=None,
                        enhancer=ContextEnhancer(), score_threshold=0.6)
    r = eng.transform(text)
    # 「氏名:」「担当は」の文脈で両方 0.75 に上がり閾値を超える。enhancer なしなら両方落ちる
    assert len(r.entities) == 2
    eng2 = PrivacyEngine(detectors=[FakeDetector(ents)], policy=Policy.preset("strict"), refiner=None, score_threshold=0.6)
    assert len(eng2.transform(text).entities) == 0

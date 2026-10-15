"""Surrogate(kind="address", keep="city"): 市区町村まで保持し番地以下を置換（識別性ベース匿名化 §3.5）。"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.data import ja_pools as jp
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Surrogate
from privacy_masker.types import Entity
from tests.conftest import FakeDetector

KEY = b"address-keep-test-key"


def run(address: str, *, keep="city", key=KEY) -> str:
    text = f"住所は{address}です。"
    ent = Entity("ADDRESS", 3, 3 + len(address), 0.9, "fake")
    eng = PrivacyEngine(
        detectors=[FakeDetector([ent])],
        policy=Policy(rules={"ADDRESS": Surrogate(kind="address", key=key, keep=keep)}),
        refiner=None,
    )
    out = eng.transform(text).text
    assert out.startswith("住所は") and out.endswith("です。")
    return out[3:-3]


@pytest.mark.parametrize("address, prefix", [
    ("東京都新宿区西新宿4-15-8 サンハイツ301", "東京都新宿区"),
    ("神奈川県横浜市中区太田町5丁目8番地", "神奈川県横浜市中区"),
    ("北海道上川郡東川町東町1-2-3", "北海道上川郡東川町"),
    ("青凪市栄町2丁目5番地", "青凪市"),  # 都道府県なし
    ("青凪市栄町1-2-3", "青凪市"),       # 都道府県なし・ハイフン番地
    ("大阪府大阪市北区曽根崎新地一丁目4番20号", "大阪府大阪市北区"),
])
def test_keep_city_preserves_coarse_part_and_replaces_fine_part(address, prefix):
    out = run(address)
    assert out.startswith(prefix)
    fine = address[len(prefix):]
    assert out != address
    assert fine not in out  # 町名・番地・建物はそのまま残らない
    assert len(out) > len(prefix)  # 偽の町名・番地が続く


def test_keep_city_is_deterministic_under_hmac():
    a = run("東京都新宿区西新宿4-15-8 サンハイツ301")
    b = run("東京都新宿区西新宿4-15-8 サンハイツ301")
    assert a == b
    assert a != run("東京都新宿区西新宿4-15-8 サンハイツ301", key=b"other-key")


@pytest.mark.parametrize("address", [
    "サンモールビル別館302号室",   # 市区町村を含まない
    "東京都新宿区",                # 市区町村で終わる（番地以下がない）
])
def test_keep_city_falls_back_to_full_surrogate_when_unparseable(address):
    out = run(address)
    assert out != address
    assert not out.startswith(address)


@pytest.mark.parametrize("address, town", [
    ("本町1-2-3", "本町"),
    ("栄町5-1", "栄町"),
    ("市場町1-2-3", "市場町"),   # 「市」を含むだけの町名（is_municipality の判定が緩かった）
])
def test_keep_city_never_keeps_a_bare_town_name_as_if_it_were_the_city(address, town):
    """Critical: 分解に使っていた正規表現の市区町村群 `.{1,8}?[市区町村]` は「◯◯町」という
    町名にも一致するため、市区町村を含まない住所で実在の町名がそのまま残っていた
    （本町1-2-3 → 本町◯◯4-7-3）。市区町村が特定できない入力は丸ごと偽住所に落とす（fail-closed）。"""
    out = run(address)
    assert out != address
    assert town not in out
    assert len(out) > len(town)                       # 偽住所そのものが返る（空でも断片でもない）


def test_keep_city_falls_back_to_full_surrogate_for_bare_prefecture():
    # 都道府県のみの入力はパース不能（市区町村がない）で丸ごと偽住所に落ちる（fail-closed）。
    # Critical #1 修正: 偽住所は入力と同じ粒度（都道府県だけ）までしか生成しないため、
    # 出力は同じ地方ブロックの別の都道府県のみで、市区町村・番地は捏造されない
    for address in ("東京都", "北海道", "大阪府"):
        out = run(address)
        assert out != address
        assert not out.startswith(address)
        assert out in jp.PREFECTURES
        assert not any(ch.isdigit() for ch in out)


def test_keep_none_is_unchanged_behaviour():
    address = "東京都新宿区西新宿4-15-8 サンハイツ301"
    out = run(address, keep=None)
    assert out != address
    assert not out.startswith("東京都新宿区")  # 全体置換（既定）


def test_invalid_keep_config():
    with pytest.raises(ConfigurationError):
        Surrogate(kind="address", key=KEY, keep="town")
    with pytest.raises(ConfigurationError):
        Surrogate(kind="person", key=KEY, keep="city")


def test_keep_city_yokkaichi():
    """「四日市市」のように市区町村名の途中に「市」を含むと、市区町村の分解が途中の「市」で切れて
    「三重県四日市」+ 偽の町名となり、市区町村名の末尾の「市」が 1 字落ちていた不具合の修正
    （町名の「諏訪町」が漏れていたのではない）。"""
    out = run("三重県四日市市諏訪町1-2")
    assert out.startswith("三重県四日市市") and "諏訪町" not in out


def test_keep_city_does_not_swallow_a_town_name_starting_with_shi():
    """最終レビュー C2: 「市」「区」で始まる町名（市谷加賀町）の先頭を市区町村名に取り込み、keep="city" が
    「東京都新宿区市」+ 偽の町名という壊れた住所を出していた。残すのはちょうど「東京都新宿区」。"""
    for i in range(100):
        key = f"ichigaya-{i}".encode()
        out = run("東京都新宿区市谷加賀町1-1", key=key)
        assert out.startswith("東京都新宿区") and not out.startswith("東京都新宿区市"), (i, out)
        assert "市谷加賀町" not in out and "谷加賀町" not in out, (i, out)


def test_keep_none_fake_never_contains_the_ward_when_the_town_starts_with_shi():
    """最終レビュー C2: avoid_parts の市区町村が「東京都新宿区市」になり「新宿区」を避けられず、
    keep=None の偽住所に元の区が混入していた（実測 18/400）。"""
    for i in range(100):
        key = f"ichigaya-none-{i}".encode()
        out = run("東京都新宿区市谷加賀町1-1", keep=None, key=key)
        assert "新宿区" not in out and "市谷加賀町" not in out, (i, out)

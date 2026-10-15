# tests/test_surrogate_filter.py
"""安全フィルタ（privacy_masker/operators/surrogate_filter.py）。設計 §7。"""
from array import array

import pytest

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.normalize import default_normalize as norm
from privacy_masker.operators import surrogate_filter as sf

EMPTY = PublicKnowledge(array("Q"))


def make(**kw):
    kw.setdefault("packaged", EMPTY)
    return sf.SurrogateFilter(**kw)


def test_same_and_partial_original_are_rejected():
    f = make()
    assert f.reject_reason("山田太郎", original_norm=norm("山田太郎")) == "same_as_original"
    assert f.reject_reason("山田 太郎", original_norm=norm("山田太郎")) == "same_as_original"       # 正規化後に同一
    assert f.reject_reason("山田健一", original_norm=norm("山田太郎"), avoid_parts=("山田", "太郎")) == "contains_original_part"
    assert f.reject_reason("佐藤太郎", original_norm=norm("山田太郎"), avoid_parts=("山田", "太郎")) == "contains_original_part"
    assert f.reject_reason("佐藤健一", original_norm=norm("山田太郎"), avoid_parts=("山田", "太郎")) is None


def test_public_entities_rejected_by_packaged_or_runtime_knowledge():
    packaged = PublicKnowledge.from_iterable(["佐藤健一"])
    runtime = PublicKnowledge.from_iterable(["東京大学", "青葉電機"])
    f = make(packaged=packaged, knowledge=runtime)
    assert f.reject_reason("佐藤健一", original_norm="x") == "public_entity"
    assert f.reject_reason("東京大学", original_norm="x") == "public_entity"
    assert f.reject_reason("株式会社青葉電機", original_norm="x") is None                 # 完全一致だけ（部分一致では弾かない）
    assert f.reject_reason("佐藤健一", original_norm="x", check_public=False) is None      # 姓だけ・実在地名は照合しない
    assert f.reject_reason("佐藤健二", original_norm="x") is None


def test_denylist_and_document_text():
    f = make(denylist=("禁止太郎",))
    assert f.reject_reason("禁止 太郎", original_norm="x") == "denylist"
    text = "会議には佐藤健一と鈴木一郎が出席した。"
    assert f.reject_reason("鈴木一郎", original_norm="x", text=text) == "present_in_document"
    assert f.reject_reason("鈴木一郎", original_norm="x", text=None) is None
    assert f.reject_reason("高橋二郎", original_norm="x", text=text) is None


def test_present_in_document_matches_after_normalization():
    """候補・本文とも正規化してから照合する（空白入り候補が空白なしの本文にも一致する）。"""
    f = make()
    text = "会議には鈴木一郎が出席した。"
    assert f.reject_reason("鈴木 一郎", original_norm="x", text=text) == "present_in_document"


def test_knowledge_accepts_path(tmp_path):
    p = tmp_path / "k.pmk"
    PublicKnowledge.from_iterable(["日本赤十字社"]).save(p)
    f = make(knowledge=str(p))
    assert f.reject_reason("日本赤十字社", original_norm="x") == "public_entity"


def test_packaged_denylist_loads_or_is_empty(monkeypatch):
    monkeypatch.setattr(sf, "PACKAGED_DENYLIST_NAME", "does_not_exist.pmk")
    sf._load_packaged_cached.cache_clear()
    with pytest.warns(UserWarning):
        empty = sf.load_packaged_denylist()
    assert isinstance(empty, PublicKnowledge)
    assert len(empty) == 0
    sf._load_packaged_cached.cache_clear()


def test_packaged_denylist_corrupt_file_raises_configuration_error(tmp_path, monkeypatch):
    """ファイルはあるが .pmk として読めない場合は空集合に縮退させず、明示的に失敗する（fail-closed）。"""
    corrupt = tmp_path / "corrupt.pmk"
    corrupt.write_bytes(b"not a valid pmk file at all")
    monkeypatch.setattr(sf, "PACKAGED_DENYLIST_NAME", "corrupt.pmk")
    monkeypatch.setattr(sf.resources, "files", lambda package: tmp_path)
    sf._load_packaged_cached.cache_clear()
    # lru_cache は例外をキャッシュしない（Python の仕様）ので、2 回呼んでも両方とも送出されるはず
    with pytest.raises(ConfigurationError, match="corrupt.pmk"):
        sf.load_packaged_denylist()
    with pytest.raises(ConfigurationError, match="corrupt.pmk"):
        sf.load_packaged_denylist()
    sf._load_packaged_cached.cache_clear()


def test_document_text_is_normalized_once_per_document():
    """Important I6: present_in_document の照合が候補ごとに文書全文を正規化していたため、
    候補 N 件 × 文書長のコストが掛かっていた。同じ text オブジェクトなら 1 回で済ませる。"""
    calls = []

    def counting(value):
        calls.append(value)
        return norm(value)

    text = "佐藤健一と鈴木一郎は同僚だ。" * 50
    f = make(normalizer=counting)
    for i in range(50):
        assert f.reject_reason(f"架空太郎{i}", original_norm="山田太郎", text=text) is None
    assert calls.count(text) == 1          # 文書は 1 回だけ正規化
    assert len(calls) == 51                # 候補 50 件 + 文書 1 回

    other = text + "追記あり。"            # 別の文書に変わったらちゃんと引き直す
    assert f.reject_reason("架空花子", original_norm="山田太郎", text=other) is None
    assert calls.count(other) == 1
    assert calls.count(text) == 1


def test_present_in_document_still_matches_after_memoization():
    text = "担当は青葉太郎です。"
    f = make()
    assert f.reject_reason("青葉太郎", original_norm="山田太郎", text=text) == "present_in_document"
    assert f.reject_reason("架空次郎", original_norm="山田太郎", text=text) is None
    assert f.reject_reason("青葉太郎", original_norm="山田太郎", text=text) == "present_in_document"


def test_reason_order_follows_design_section_7():
    """§7 の順: 同一 → 部分混入 → 公知 → 文書内既出 → 利用者 denylist。両方に当たる候補は present_in_document"""
    f = make(denylist=("禁止太郎",))
    assert f.reject_reason("禁止太郎", original_norm="x", text="禁止太郎が出席") == "present_in_document"
    assert f.reject_reason("禁止太郎", original_norm="x") == "denylist"


def test_forget_document_drops_the_memo():
    f = make()
    text = "会議には佐藤健一が出席した。"
    assert f.reject_reason("佐藤健一", original_norm="x", text=text) == "present_in_document"
    assert f._text_cache is not None and f._text_cache[0] is text
    f.forget_document()
    assert f._text_cache is None


def test_empty_text_and_empty_avoid_parts():
    f = make()
    assert f.reject_reason("佐藤健一", original_norm="x", text="", avoid_parts=()) is None


def test_unreadable_packaged_file_is_configuration_error(tmp_path, monkeypatch):
    """壊れていない（読めばそれと分かる）が読めないファイル、例えば同名のディレクトリが
    誤って置かれた場合も OSError を握りつぶさず ConfigurationError に包む（fail-closed）。
    既存の corrupt file テストと同じ差し替え点（PACKAGED_DENYLIST_NAME・resources.files）を使う。"""
    d = tmp_path / sf.PACKAGED_DENYLIST_NAME
    d.mkdir()          # open() で IsADirectoryError（OSError のサブクラス）になる
    monkeypatch.setattr(sf.resources, "files", lambda package: tmp_path)
    sf._load_packaged_cached.cache_clear()
    with pytest.raises(ConfigurationError, match="surrogate_denylist"):
        sf.load_packaged_denylist()
    sf._load_packaged_cached.cache_clear()


import pytest

from privacy_masker.exceptions import StoreError
from privacy_masker.stores import MemoryStore, SQLiteStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryStore()
    else:
        s = SQLiteStore(str(tmp_path / "test.db"))
        yield s
        s.close()


def test_put_and_get_token(store):
    assert store.get_token("s1", "PERSON", "vh1") is None
    store.put_token("s1", "PERSON", "vh1", "<PERSON_001>", None)
    assert store.get_token("s1", "PERSON", "vh1") == "<PERSON_001>"
    assert store.get_token("s2", "PERSON", "vh1") is None  # scope分離


def test_get_value(store):
    store.put_token("s1", "PERSON", "vh1", "<PERSON_X>", b"encrypted-blob")
    assert store.get_value("s1", "<PERSON_X>") == b"encrypted-blob"
    assert store.get_value("s1", "<UNKNOWN>") is None
    assert store.get_value("s2", "<PERSON_X>") is None


def test_token_uniqueness_within_scope(store):
    store.put_token("s1", "PERSON", "vh1", "<T1>", None)
    with pytest.raises(StoreError):
        store.put_token("s1", "PERSON", "vh2", "<T1>", None)
    store.put_token("s2", "PERSON", "vh1", "<T1>", None)  # 別scopeはOK


def test_value_hash_uniqueness_within_scope(store):
    store.put_token("s1", "PERSON", "vh1", "<T1>", None)
    with pytest.raises(StoreError):
        store.put_token("s1", "PERSON", "vh1", "<T2>", None)  # 同一value_hashへの別token登録
    store.put_token("s2", "PERSON", "vh1", "<T2>", None)  # 別scopeはOK


def test_next_seq_increments_per_scope_and_type(store):
    assert store.next_seq("s1", "PERSON") == 1
    assert store.next_seq("s1", "PERSON") == 2
    assert store.next_seq("s1", "ORG") == 1
    assert store.next_seq("s2", "PERSON") == 1


def test_delete_scope(store):
    store.put_token("s1", "PERSON", "vh1", "<T1>", b"blob")
    store.next_seq("s1", "PERSON")
    store.put_token("s2", "PERSON", "vh1", "<T2>", None)
    store.delete_scope("s1")
    assert store.get_token("s1", "PERSON", "vh1") is None
    assert store.get_value("s1", "<T1>") is None
    assert store.next_seq("s1", "PERSON") == 1  # カウンタもリセット
    assert store.get_token("s2", "PERSON", "vh1") == "<T2>"  # 他scopeは無傷


def test_sqlite_persists_across_connections(tmp_path):
    path = str(tmp_path / "p.db")
    s1 = SQLiteStore(path)
    s1.put_token("s1", "PERSON", "vh1", "<T1>", b"blob")
    s1.close()
    s2 = SQLiteStore(path)
    assert s2.get_token("s1", "PERSON", "vh1") == "<T1>"
    s2.close()


def test_iter_tokens_lists_scope_only_in_insertion_order(store):
    assert list(store.iter_tokens("s1")) == []
    store.put_token("s1", "PERSON", "vh1", "佐藤健一", b"blob1")
    store.put_token("s1", "ORG", "vh2", "株式会社青葉電機", None)
    store.put_token("s2", "PERSON", "vh3", "<PERSON_X>", None)
    assert list(store.iter_tokens("s1")) == [("PERSON", "佐藤健一"), ("ORG", "株式会社青葉電機")]
    assert list(store.iter_tokens("s2")) == [("PERSON", "<PERSON_X>")]
    store.delete_scope("s1")
    assert list(store.iter_tokens("s1")) == []

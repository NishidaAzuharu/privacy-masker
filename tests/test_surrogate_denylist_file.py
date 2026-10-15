# tests/test_surrogate_denylist_file.py
"""同梱の禁止リスト（privacy_masker/data/surrogate_denylist.pmk）が存在し、読め、生成器の候補全体から作られていること。"""
import json
import os

from privacy_masker.annotators import PublicKnowledge, normalize_surface
from privacy_masker.operators import surrogate_filter as sf
from privacy_masker.operators.surrogate_generators import enumerate_candidates

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "privacy_masker", "data")


def test_packaged_denylist_exists_and_loads():
    path = os.path.join(DATA, sf.PACKAGED_DENYLIST_NAME)
    assert os.path.exists(path), "tools/build_surrogate_denylist.py で生成してパッケージに置く"
    k = PublicKnowledge.load(path)
    assert len(k) > 0
    m = json.load(open(os.path.join(DATA, "surrogate_denylist.manifest.json"), encoding="utf-8"))
    assert m["candidates"] >= 20000 and m["hits"] == m["entries_written"] > 0
    assert m["knowledge_sha256"] and m["created_at"]


def test_packaged_denylist_is_subset_of_candidate_universe():
    """禁止リストの各エントリは候補全体のどれかのハッシュ（候補以外の語は入っていない）。"""
    k = sf.load_packaged_denylist()
    hits = [c for c in set(enumerate_candidates()) if c in k]
    assert len(hits) >= 1
    assert all(c in k for c in hits)  # 一致件数の定義そのもの（構築の妥当性を明示する）
    # PublicKnowledge.from_iterable は表層ごとに「正規化後の表層」のハッシュ 1 個と、空白圧縮後の
    # 表層がそれと異なる場合のみもう 1 個を追加する（knowledge.py 69-79 行）。そのため候補側に
    # 「姓 名」と「姓名」のような別表記が両方あると、両者の正規化後表層のハッシュは別々でも、
    # 空白圧縮後の表層は一致してハッシュ集合上で重なり、一致した候補文字列の数（hits）が
    # ユニークなハッシュ件数（len(k)）よりわずかに多くなりうる。from_iterable と同じ規則で
    # hits から「格納されるはずの表層」の集合を再現すれば、この重なりを含めて件数が厳密に一致する
    # （64bit ハッシュの偶然衝突は無視できるほど小さいため、表層集合の要素数とハッシュ件数は事実上同じ）。
    stored_forms: set[str] = set()
    for c in hits:
        n = normalize_surface(c)
        stored_forms.add(n)
        compact = n.replace(" ", "")
        if compact != n:
            stored_forms.add(compact)
    assert len(stored_forms) == len(k)


def test_manifest_path_uses_splitext_and_trailing_newline():
    from tools.build_surrogate_denylist import manifest_path_for

    assert manifest_path_for("/x/a.pmk.pmk") == "/x/a.pmk.manifest.json"
    raw = open(os.path.join(DATA, "surrogate_denylist.manifest.json"), "rb").read()
    assert raw.endswith(b"\n")

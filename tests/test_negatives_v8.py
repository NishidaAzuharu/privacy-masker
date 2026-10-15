"""負例 v8 の計画と選別（tools/negatives_v8.py）。Drive も NER モデルも読まない。"""
import csv
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import negatives_v8 as nv  # noqa: E402

from privacy_masker.types import Entity  # noqa: E402


# ---------------------------------------------------------------- plan

def test_plan_prefix_keeps_genre_ratio_and_ids_unique():
    rows = nv.build_plan(600, seed=42)
    assert len({r["request_id"] for r in rows}) == 600
    for k in (200, 600):
        c = Counter(r["genre_key"] for r in rows[:k])
        for g in nv.GENRES:
            assert abs(c[g.key] - k * g.weight) <= 1, (k, g.key, c[g.key])


def test_plan_is_deterministic_and_prompt_has_constraints():
    a = nv.build_plan(50, seed=7)
    b = nv.build_plan(50, seed=7)
    assert [r["prompt_sha256"] for r in a] == [r["prompt_sha256"] for r in b]
    for r in a:
        assert r["topic"] in r["prompt"]
        assert "禁止事項" in r["prompt"]
        assert r["format_key"] in nv.FORMATS and r["style"] in nv.STYLES
        assert r["length_key"] in nv.LENGTH_KEYS and r["max_tokens"] == nv.MAX_TOKENS[r["length_key"]]
        assert r["length_min"] < r["length_max"]
    combos = Counter((r["subgenre_key"], r["topic"], r["format_key"], r["style"]) for r in nv.build_plan(2000, seed=1))
    assert max(combos.values()) <= 3


def test_plan_manifest_carries_system_prompt_and_schema():
    rows = nv.build_plan(30, seed=3)
    m = nv.plan_manifest(rows, 3, "deadbeef")
    assert m["system_prompt"] == nv.SYSTEM_PROMPT
    assert m["output_schema"]["required"] == ["lines"]
    assert sum(m["genre_counts"].values()) == 30
    assert "人名を書かない" in m["system_prompt"] and "地名を書かない" in m["system_prompt"]


# ---------------------------------------------------------------- deterministic rules

@pytest.mark.parametrize("text, reason", [
    ("田中さんが担当します。", "surname_with_title"),
    ("森部長に報告した。", "surname_with_title"),
    ("森花子が来た。", "full_name"),
    ("真由美が来た。", "given_name"),
    ("誠さんが来た。", "given_name_with_title"),
    ("担当: 佐藤", "name_after_label"),
    ("operator=鈴木", "name_after_label"),
    ("ジョンと会った。", "katakana_name"),
    ("東京都から通知が届いた。", "place_name"),
    ("仙台では雪が降った。", "place_name"),
    ("北海道産のじゃがいも。", "place_name"),
    ("欧州の市場。", "place_name"),
    ("パリの街を歩いた。", "place_name"),
    ("日本の伝統。", "place_name"),
    ("国連の会議。", "known_org"),
    ("読売の記事。", "known_org"),
    ("ANA の便。", "known_org"),
    ("株式会社の設立手続き。", "corporate_marker"),
    ("総務課へ提出。", "department_name"),
    ("連絡先は 03-1234-5678 です。", "phone"),
    ("〒100-0001 に送付。", "postal_code"),
    ("info@example.com へ。", "email"),
    ("https://example.com を参照。", "url"),
    ("@taro_88 ありがとう。", "mention"),
    ("1-2-3号室に移動。", "address_banchi"),
    ("这个案件を确认します。", "simplified_chinese"),
    ("안녕하세요。", "foreign_script"),
    ("<PERSON>誰か</PERSON>が来た。", "tag_residue"),
    # 盲検の表層はテストに平文で書かない（tests/test_no_eval_surfaces.py）。正本の集合から決定的に 1 件取る
    pytest.param((sorted(nv.EVAL_SURFACES) or ["(no eval surfaces file)"])[0] + ": こんにちは", "eval_surface", id="eval_surface",
                 marks=pytest.mark.skipif(not nv.EVAL_SURFACES, reason="盲検表層ファイル（データフォルダ）が無い環境")),
])
def test_deterministic_reject(text, reason):
    reasons, evidence = nv.deterministic_reject_reasons(text)
    assert reason in reasons, (reasons, evidence)
    assert any(e.startswith(reason + ":") for e in evidence)


@pytest.mark.parametrize("text", [
    "本日の定例会議では在庫の棚卸し方法を確認した。担当者は当番制とする。",
    "status=OK level=INFO owner=admin retry_count=3",
    "担当: 早番、記録者: 夜勤担当、承認: 済",
    "アミメニシキヘビは大型のヘビである。カラスが鳴いた。",
    "鶏皮をパリパリに焼く。ローマ字で入力する。中国語の授業。アメリカザリガニを観察した。イタリアンの店。",
    "フランス料理を食べた。韓国語を学ぶ。アメリカ人と話した。日本語の文章。",
    "価格は100-2000円の範囲。",
    "明治維新の頃の話。イオン結合について。ANALYSIS を実施。Metadata を確認。",
    "三重の対策で、大分よくなった。誠に申し訳ない。ダンジョンに入る。",
    "近所の病院で受診し、担当部署に連絡した。政府の方針と与党の対応。大学の講義。",
    "09:12:03 WARN 再試行 2回目\n09:12:10 INFO 復旧",
])
def test_deterministic_keep(text):
    reasons, evidence = nv.deterministic_reject_reasons(text)
    assert reasons == [], (reasons, evidence)


def test_place_name_katakana_needs_full_run_and_derivations_are_allowed():
    assert nv.place_name_hits("パリパリ ローマ字 アメリカザリガニ") == []
    assert nv.place_name_hits("パリ ローマ アメリカ") == ["パリ", "ローマ", "アメリカ"]
    assert nv.place_name_hits("中国語 中国") == ["中国"]
    # 国内地名には派生語の許容を適用しない
    assert nv.place_name_hits("東京産") == ["東京"]


def test_length_reason():
    assert nv.length_reason("短い") == "too_short"
    assert nv.length_reason("あ" * 901) == "too_long"
    assert nv.length_reason("あ" * 100) is None


# ---------------------------------------------------------------- review rules

def _hit(det, typ, text, surface, score=0.9):
    s = text.index(surface)
    return {"detector": det, "type": typ, "start": s, "end": s + len(surface), "surface": surface, "score": score}


def test_org_like_surface():
    assert nv._org_like("曙製作所") and nv._org_like("青葉総合病院") and nv._org_like("みなと書店")
    assert not nv._org_like("市立病院") and not nv._org_like("総合病院") and not nv._org_like("近所の薬局")
    assert not nv._org_like("当センター") and not nv._org_like("大学") and not nv._org_like("本店")


def test_review_reasons_agreement_title_org_handle_public():
    text = "係の判断で操作者が蒼波へ連絡した。曙製作所の担当者。owner=yuki_t で登録。仙台へ移動。"
    hits = [
        _hit("char8ep", "PERSON", text, "蒼波"), _hit("v7", "PERSON", text, "蒼波"),
        _hit("v7", "ORG", text, "曙製作所"), _hit("v7", "FACILITY", text, "係"),
        _hit("handle", "USERNAME", text, "yuki_t"),
        _hit("char8ep", "LOCATION", text, "仙台"), _hit("char8ep", "LOCATION", text, "係"),
    ]
    reasons, evidence = nv.review_reasons(text, hits, knowledge={"仙台", "係"})
    assert "agreed_person" in reasons and "org_like_surface" in reasons and "handle_like" in reasons
    assert "public_location" in reasons
    assert "public_location:係" not in evidence   # 1 文字の LOCATION は誤検出として残す
    # 敬称つき PERSON
    text2 = "担当の朱鷺野さんに確認した。"
    reasons2, _ = nv.review_reasons(text2, [_hit("v7", "PERSON", text2, "朱鷺野")])
    assert reasons2 == ["person_with_title"]


def test_review_ignores_single_detector_common_noun_and_generic_handle_values():
    text = "市民の意見。owner=admin で登録。"
    hits = [_hit("v7", "ORG", text, "市民"), _hit("handle", "USERNAME", text, "admin")]
    assert nv.review_reasons(text, hits) == ([], [])


def test_review_keeps_prompt_vocab_agreement_and_word_cut_locations():
    # 題材・役割語（粉屋・縄暖簾・カメ・書生）は 2 モデル一致でも残す
    text = "粉屋と書生が縄暖簾の店先でカメを見た。大通り沿いは避けた。山あいの村。"
    hits = []
    for w in ("粉屋", "書生", "縄暖簾", "カメ"):
        hits += [_hit("char8ep", "PERSON", text, w), _hit("v7", "PERSON", text, w)]
    hits += [_hit("char8ep", "LOCATION", text, "大通"), _hit("v7", "LOCATION", text, "あい")]
    knowledge = {"大通", "大通り", "あい", "粉屋"}
    assert nv.review_reasons(text, hits, knowledge) == ([], [])
    assert nv.in_prompt_vocab("若旦") and nv.in_prompt_vocab("川を渡るカメ") and not nv.in_prompt_vocab("蒼波")


def test_katakana_common_nouns_are_not_names():
    assert nv.deterministic_reject_reasons("車のミラーを拭き、ブラウンの靴を履いた。アリが一匹。")[0] == []
    assert "katakana_name" in nv.deterministic_reject_reasons("スミスと会った。")[0]


def test_prompt_asks_for_lines_array():
    row = nv.build_plan(20, seed=1)[0]
    assert '"lines"' in row["prompt"] and '"lines"' in nv.SYSTEM_PROMPT
    m = nv.plan_manifest([row], 1, "x")
    assert m["output_schema"]["required"] == ["lines"]
    assert all("改行" in nv.FORMATS[k] for k in ("bullets", "numbered_steps", "chat_roles", "log_lines", "label_colon"))


def test_parsed_text_joins_lines_and_accepts_legacy_text():
    assert nv.parsed_text({"lines": ["・一", "・二", "", "", "", "段落"]}) == "・一\n・二\n\n段落"
    assert nv.parsed_text({"text": "  本文  "}) == "本文"
    assert nv.parsed_text({"lines": []}) is None and nv.parsed_text(None) is None
    raw = {"neg": {"request_id": "neg", "parsed": {"lines": ["status=OK", "level=INFO", "retry=3", "監視画面で復旧を確認した。"]}}}
    plan = [dict(nv.build_plan(1, seed=5)[0], request_id="neg")]
    res = nv.run_filter(raw, plan, target=5, detectors=None, quota_by_genre=False)
    assert res["records"][0]["text"].count("\n") == 3


def test_review_keeps_hiragana_person_agreement():
    # 文語の役割語「ねえや」は 2 モデルが PERSON で一致しても負例として残す（教えたい誤検出そのもの）
    text = "ねえやが縁側で洗い物をしていた。"
    hits = [_hit("char8ep", "PERSON", text, "ねえや"), _hit("v7", "PERSON", text, "ねえや")]
    assert nv.review_reasons(text, hits) == ([], [])
    text2 = "蒼波が縁側で洗い物をしていた。"
    hits2 = [_hit("char8ep", "PERSON", text2, "蒼波"), _hit("v7", "PERSON", text2, "蒼波")]
    assert nv.review_reasons(text2, hits2)[0] == ["agreed_person"]


# ---------------------------------------------------------------- near duplicates

def test_near_duplicate_pass_drops_paraphrase_and_keeps_distinct():
    a = "今日は雨が降ったので洗濯物を室内に干した。乾きにくいので扇風機を回した。"
    b = "今日は雨が降ったので洗濯物を室内に干しました。乾きにくいので扇風機を回しました。"
    c = "カブトムシの飼育では温度管理が重要で、夏場は直射日光を避ける。"
    dropped, flagged = nv.near_duplicate_pass([a, b, c, a])
    assert set(dropped) == {1, 3}
    assert 2 not in dropped
    # 短文は対象外
    d2, _ = nv.near_duplicate_pass(["短い文。", "短い文！"])
    assert d2 == {}


# ---------------------------------------------------------------- run_filter end to end

class FakeDetector:
    def __init__(self, spans):
        self.spans = spans   # surface -> type

    def detect(self, text):
        out = []
        for surface, typ in self.spans.items():
            i = text.find(surface)
            if i >= 0:
                out.append(Entity(typ, i, i + len(surface), 0.9, "fake"))
        return out


def _raw(rid, text, error=""):
    return {"request_id": rid, "parsed": ({"text": text} if text is not None else None), "structured_error": error}


def _plan_rows(n):
    rows = nv.build_plan(n, seed=11)
    return rows


def test_run_filter_end_to_end(tmp_path):
    plan = _plan_rows(12)
    ids = [p["request_id"] for p in plan]
    base = "在庫の棚卸しは毎月末に行う。担当者は当番制で、記録は所定の表に残す。差異があれば翌日に再確認する。"
    texts = {
        ids[0]: base,
        ids[1]: base.replace("毎月末", "毎月の末日"),            # 近重複 → reject
        ids[2]: "田中さんが担当します。" + base,                   # 規則 reject
        ids[3]: "係の判断で操作者が蒼波へ連絡した。復旧の手順は夜勤の引き継ぎ表に沿って進め、完了後に監視画面で確認する。",  # 2 モデル一致 → review
        ids[4]: "市民の意見を集約した。集会では騒音と駐輪場の不足が多く挙がり、次回までに担当部署が回答案を用意する。",  # 1 モデルの一般語 → keep（ハード負例）
        ids[5]: None,                                              # 構造化失敗
        ids[6]: "短い",                                            # too_short
        ids[7]: "カブトムシの飼育では温度管理が重要で、夏場は直射日光を避ける。土は乾いたら霧吹きで湿らせる。",
        ids[8]: "レビュー戻し用。曙製作所の担当者に確認した。納品は翌週の午前に変更し、検収は受入担当が立ち会う。",  # org_like → review、include で戻す
    }
    raw = {rid: _raw(rid, t) for rid, t in texts.items()}
    raw[ids[5]]["structured_error"] = "json_decode_error"
    det = {"a": FakeDetector({"蒼波": "PERSON", "市民": "ORG", "曙製作所": "ORG"}),
           "b": FakeDetector({"蒼波": "PERSON"})}
    res = nv.run_filter(raw, plan, target=10, detectors=det, include_ids=[ids[8]], quota_by_genre=False)
    v = res["verdicts"]
    assert v[ids[1]].decision == "reject" and v[ids[1]].reasons == ["near_duplicate"]
    assert v[ids[2]].decision == "reject" and "surname_with_title" in v[ids[2]].reasons
    assert v[ids[3]].decision == "review" and "agreed_person" in v[ids[3]].reasons
    assert v[ids[4]].decision == "keep" and v[ids[4]].ner_hits
    assert v[ids[5]].decision == "reject" and v[ids[5]].reasons == ["structured_output_failed"]
    assert v[ids[6]].decision == "reject" and v[ids[6]].reasons == ["too_short"]
    assert v[ids[8]].decision == "keep" and v[ids[8]].reasons == ["included_from_review"]
    assert res["counts"]["not_generated"] == 3 and res["counts"]["generated"] == 9
    assert set(res["selected"]) == {ids[0], ids[4], ids[7], ids[8]}
    for rec in res["records"]:
        assert rec["entities"] == [] and rec["source"] == nv.SOURCE_NAME and rec["variant"].count("/") == 2

    summary = nv.write_filter_outputs(res, plan, str(tmp_path), target=10, model_names=["a", "b"], raw_paths=["x"])
    for name in ("negatives_v8.jsonl", "rejected.csv", "review_excluded.csv", "ner_hits.csv", "report.md", "manifest.json"):
        assert (tmp_path / name).exists(), name
    assert summary["selected"] == 4 and summary["hard_negatives_with_ner_hit"] >= 1
    with open(tmp_path / "review_excluded.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["request_id"] for r in rows] == [ids[3]]
    with open(tmp_path / "negatives_v8.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    assert len(recs) == 4


def test_run_filter_quota_by_genre_and_opening_repeat():
    plan = _plan_rows(60)
    # 全件同じ書き出しの文書（ジャンル別に 5 件を超えると opening_repeat）。近重複にならないよう末尾を変える
    import random
    rng = random.Random(0)
    fillers = ["カブトムシの飼育では温度管理が重要で夏場は直射日光を避ける", "換気扇の掃除は月に一度が目安で油汚れは重曹で落とす",
               "紫陽花の剪定は花が終わった直後に行い翌年の花芽を残す", "ジョギングは週三回二十分から始めて距離より継続を優先する",
               "電気ケトルは沸騰が速く注ぎ口が細いものが使いやすい", "契約期間は一年とし双方から申し出がなければ自動更新する",
               "支払いは検収月の翌月末までに指定の方法で行う", "洗濯物の生乾き臭は脱水後すぐに干し風を当てると防げる",
               "寝かしつけは毎晩同じ手順を繰り返すと寝つきが安定する", "資格勉強は過去問を先に解いて出題の傾向をつかむ"]
    raw = {}
    for i, p in enumerate(plan):
        # 書き出し 20 字を全件でそろえ（opening_signature の長さ）、本文は毎回違う組み合わせにする
        tail = " ".join(rng.sample(fillers, 3)) + f" この記録は{i}番目の例で、{i * 7}分の作業を想定する。"
        raw[p["request_id"]] = _raw(p["request_id"], "点検記録の書き方についての覚え書きです。" + tail)
    res = nv.run_filter(raw, plan, target=20, detectors=None, quota_by_genre=True)
    v = res["verdicts"]
    reasons = Counter(r for x in v.values() for r in x.reasons)
    assert reasons["opening_repeat"] > 0
    assert len(res["selected"]) <= 20
    by_genre = Counter(next(p for p in plan if p["request_id"] == rid)["genre_key"] for rid in res["selected"])
    for g in nv.GENRES:
        assert by_genre[g.key] <= 5   # 書き出し上限がジャンル別に効く

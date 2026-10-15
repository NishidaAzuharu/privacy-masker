"""v11 の構築（tools/build_data_v11.py）。Drive は読まず、小さな行で分割・連結・漏洩検査・出力を確かめる。"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import build_data_v11 as b  # noqa: E402


def _doc(text, ents, source, **kw):
    return {"text": text, "entities": ents, "source": source, **kw}


V10_TRAIN = [_doc("山本健一は奈良県に住む。", [[0, 4, "PERSON"], [5, 8, "LOCATION"]], "out_ner"),
             _doc("株式会社旭の田村です。", [[0, 5, "ORG"], [6, 8, "PERSON"]], "synth_v5c"),
             _doc("記録者：夜勤 朱鷺野", [[7, 10, "PERSON"]], "synth_form"),
             _doc("本日は晴れです。", [], "qwen38_negatives_v8")]
V10_VALID = [_doc("井上美咲は京都府出身。", [[0, 4, "PERSON"], [5, 8, "LOCATION"]], "out_ner")]
GSD = [_doc(f"文{i}は東京の日本人が書いた。", [[3, 5, "LOCATION"]], "gsd_train", neutral=[[6, 9]], variant="gsd/train", id=f"s{i}") for i in range(20)]
SYNTH = [_doc(f"{i} 番目のテンプレート {j}。担当は佐藤花子。", [[len(f'{i} 番目のテンプレート {j}。担当は'), len(f'{i} 番目のテンプレート {j}。担当は') + 4, "PERSON"]],
              "qwen38_synth_v11", template_id=f"t{i}", fill_index=j, variant="business/x/prose") for i in range(10) for j in range(2)]
NEG = [{"text": f"負例 {i} の本文です。", "entities": [], "source": "qwen38_negatives_v8", "variant": "business/a/prose"} for i in range(30)]


def test_split_by_template_id_keeps_fills_of_one_template_together():
    tr, va = b.split_by_template_id(SYNTH, random.Random(1), 0.2)
    assert len(tr) + len(va) == len(SYNTH) and va
    assert not ({d["template_id"] for d in tr} & {d["template_id"] for d in va})


def _long_gsd(i):
    """署名が異なり（数字以外で違う）、2 本つなげば 400 字を超える GSD 風の文。位置は GSD と同じ。"""
    filler = "".join(chr(0x3042 + (i * 7 + k * 3) % 80) for k in range(220))
    return _doc(f"文{i}は東京の日本人が書いた。" + filler, [[3, 5, "LOCATION"]], "gsd_train", neutral=[[6, 9]], variant="gsd/train", id=f"s{i}")


def test_concat_carries_neutral_and_offsets():
    docs = b.make_concat_docs([_long_gsd(i) for i in range(10)], random.Random(2), target=3)   # i が 2 桁だと位置がずれる
    assert 1 <= len(docs) <= 3
    for d in docs:
        assert d["source"] == "concat" and "\n\n" in d["text"]
        for s, e, l in d["entities"]:
            assert d["text"][s:e] == "東京"
        for s, e in d["neutral"]:
            assert d["text"][s:e] == "日本人"


def test_build_assembles_layers_and_variants():
    res = b.build(v10_train=V10_TRAIN, v10_valid=V10_VALID, gsd=GSD, synth_docs=SYNTH, negatives=NEG, hard_texts=set(),
                  excl_person={"井上美咲"}, eval_texts={"文3は東京の日本人が書いた。"}, seed=42, n_negatives=10, concat_ratio=0.0)
    train, plus, valid, info = res["train"], res["train_plus_old"], res["valid"], res["info"]
    srcs = {d["source"] for d in train}
    assert "out_ner" in srcs and "gsd_train" in srcs and "qwen38_synth_v11" in srcs and "qwen38_negatives_v8" in srcs
    assert "synth_v5c" not in srcs and "synth_form" not in srcs                  # v11 単体に旧合成は入らない
    assert {d["source"] for d in plus} >= srcs | {"synth_v5c", "synth_form"}   # plus_old は旧合成を足す
    assert any(d["text"] == V10_TRAIN[0]["text"] for d in train) and any(d["text"] == V10_VALID[0]["text"] for d in valid)
    assert not any(d["text"] == "文3は東京の日本人が書いた。" for d in train + valid)     # 評価本文との一致は落とす
    assert info["leak"]["eval_text_overlap"] == 0 and info["leak"]["eval_text_overlap_dropped"] == 1
    assert info["leak"]["eval_person_surface"] == 0
    assert info["negatives"]["train"] + info["negatives"]["valid"] == 10
    assert all("neutral" in d for d in train if d["source"] == "gsd_train")


def test_write_records_keeps_neutral_and_data_card_has_source_table(tmp_path):
    res = b.build(v10_train=V10_TRAIN, v10_valid=V10_VALID, gsd=GSD, synth_docs=SYNTH, negatives=NEG, hard_texts=set(),
                  excl_person=set(), eval_texts=set(), seed=1, n_negatives=6, concat_ratio=0.0)
    p = tmp_path / "train.jsonl"
    b.write_records(str(p), res["train"])
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert any("neutral" in r for r in rows) and all(set(r) >= {"text", "entities", "source"} for r in rows)
    card = b.data_card(res["info"], res["train"], res["valid"], res["train_plus_old"])
    assert "| 出所 |" in card and "gsd_train" in card and "qwen38_synth_v11" in card


def test_leak_check_ignores_real_text_parts_inside_concat():
    """連結文書のうち GSD 由来の部分に評価セットと同じ姓があっても漏洩に数えない。合成部分なら数える。"""
    gsd = _long_gsd(1)
    gsd["text"] = "山田は" + gsd["text"][3:]          # 先頭 2 字を評価セットと同じ姓に
    gsd["entities"] = [[0, 2, "PERSON"], [3, 5, "LOCATION"]]
    syn = _doc("担当は山田です。" + "本文" * 120, [[3, 5, "PERSON"]], "qwen38_synth_v11", template_id="t9", fill_index=0, variant="business/x/prose")
    docs = b.make_concat_docs([gsd, syn], random.Random(0), target=1)
    assert len(docs) == 1 and docs[0]["parts"] and {p[2] for p in docs[0]["parts"]} == {"gsd_train", "qwen38_synth_v11"}
    assert b.leak_checks(docs, {"山田"}, set())["eval_person_surface"] == 1       # 合成部分の 1 件だけ
    only_gsd = b.make_concat_docs([gsd, _long_gsd(2)], random.Random(0), target=1)
    assert b.leak_checks(only_gsd, {"山田"}, set())["eval_person_surface"] == 0


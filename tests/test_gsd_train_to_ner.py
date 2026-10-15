"""GSD の conllu → 学習形式（tools/gsd_train_to_ner.py）。ネットワークは使わずフィクスチャで検査。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import gsd_train_to_ner as g  # noqa: E402

CONLLU = """# newdoc id = train-s1
# sent_id = train-s1
# text = 田中太郎は東京の日本人だ。
1\t田中\t田中\tPROPN\t名詞-固有名詞-人名-姓\t_\t3\tnsubj\t_\tSpaceAfter=No|NE=B-PERSON
2\t太郎\t太郎\tPROPN\t名詞-固有名詞-人名-名\t_\t1\tflat\t_\tSpaceAfter=No|NE=I-PERSON
3\tは\tは\tADP\t助詞-係助詞\t_\t1\tcase\t_\tSpaceAfter=No
4\t東京\t東京\tPROPN\t名詞-固有名詞-地名-一般\t_\t7\tnmod\t_\tSpaceAfter=No|NE=B-GPE
5\tの\tの\tADP\t助詞-格助詞\t_\t4\tcase\t_\tSpaceAfter=No
6\t日本\t日本\tPROPN\t名詞-固有名詞-地名-国\t_\t7\tcompound\t_\tSpaceAfter=No|NE=B-NORP
7\t人\t人\tNOUN\t名詞-普通名詞-一般\t_\t0\troot\t_\tSpaceAfter=No|NE=I-NORP
8\tだ\tだ\tAUX\t助動詞\t_\t7\tcop\t_\tSpaceAfter=No
9\t。\t。\tPUNCT\t補助記号-句点\t_\t7\tpunct\t_\tSpaceAfter=No

# sent_id = train-s2
# text = ABC 社は 2020 年に設立された。
1-2\tABC 社\t_\t_\t_\t_\t_\t_\t_\t_
1\tABC\tABC\tPROPN\t名詞\t_\t2\tcompound\t_\tNE=B-ORG
2\t社\t社\tNOUN\t名詞\t_\t6\tnsubj\t_\tSpaceAfter=No|NE=I-ORG
3\tは\tは\tADP\t助詞\t_\t2\tcase\t_\t_
4\t2020\t2020\tNUM\t名詞-数詞\t_\t5\tnummod\t_\tNE=B-DATE
5\t年\t年\tNOUN\t名詞\t_\t6\tobl\t_\tSpaceAfter=No|NE=I-DATE
6\tに\tに\tADP\t助詞\t_\t5\tcase\t_\tSpaceAfter=No
7\t設立\t設立\tNOUN\t名詞\t_\t0\troot\t_\tSpaceAfter=No
8\tされた\tされる\tAUX\t助動詞\t_\t7\taux\t_\tSpaceAfter=No
9\t。\t。\tPUNCT\t補助記号\t_\t7\tpunct\t_\tSpaceAfter=No
"""


def test_parse_maps_labels_and_puts_unmapped_types_in_neutral():
    sents, stats = g.parse_conllu(CONLLU)
    assert [s.sent_id for s in sents] == ["train-s1", "train-s2"] and stats["text_mismatch"] == 0
    s1 = sents[0]
    assert s1.text == "田中太郎は東京の日本人だ。"
    assert s1.entities == [[0, 4, "PERSON"], [5, 7, "LOCATION"]]
    assert s1.neutral == [[8, 11]]                        # NORP「日本人」は採点対象外
    s2 = sents[1]
    assert s2.text == "ABC 社は 2020 年に設立された。"          # SpaceAfter に従って空白を復元
    assert s2.entities == [[0, 5, "ORG"]] and s2.neutral == [[7, 13]]


def test_records_and_overlap_drop():
    sents, _ = g.parse_conllu(CONLLU)
    recs = g.to_records(sents, "train")
    assert recs[0]["source"] == "gsd_train" and recs[0]["variant"] == "gsd/train" and "neutral" in recs[0]
    kept, dropped = g.drop_overlaps(recs, {"田中太郎は東京の日本人だ。"})
    assert dropped == 1 and [r["id"] for r in kept] == ["train-s2"]


def test_text_mismatch_is_counted_and_sentence_skipped():
    broken = CONLLU.replace("# text = 田中太郎は東京の日本人だ。", "# text = 別の文。", 1)
    sents, stats = g.parse_conllu(broken)
    assert stats["text_mismatch"] == 1 and [s.sent_id for s in sents] == ["train-s2"]


CONLLU_BILOU = """# sent_id = train-s3
# text = 佐藤氏は日本銀行の藤本博史と宮城県石巻市で会った。
1\t佐藤\t佐藤\tPROPN\t名詞\t_\t0\troot\t_\tSpaceAfter=No|NE=U-PERSON
2\t氏\t氏\tNOUN\t名詞\t_\t1\tflat\t_\tSpaceAfter=No
3\tは\tは\tADP\t助詞\t_\t1\tcase\t_\tSpaceAfter=No
4\t日本\t日本\tPROPN\t名詞\t_\t5\tcompound\t_\tSpaceAfter=No|NE=B-ORG
5\t銀行\t銀行\tNOUN\t名詞\t_\t7\tnmod\t_\tSpaceAfter=No|NE=L-ORG
6\tの\tの\tADP\t助詞\t_\t5\tcase\t_\tSpaceAfter=No
7\t藤本\t藤本\tPROPN\t名詞\t_\t0\tobl\t_\tSpaceAfter=No|NE=B-PERSON
8\t博史\t博史\tPROPN\t名詞\t_\t7\tflat\t_\tSpaceAfter=No|NE=L-PERSON
9\tと\tと\tADP\t助詞\t_\t7\tcase\t_\tSpaceAfter=No
10\t宮城\t宮城\tPROPN\t名詞\t_\t0\tobl\t_\tSpaceAfter=No|NE=B-GPE
11\t県\t県\tNOUN\t名詞\t_\t10\tflat\t_\tSpaceAfter=No|NE=I-GPE
12\t石巻\t石巻\tPROPN\t名詞\t_\t10\tflat\t_\tSpaceAfter=No|NE=I-GPE
13\t市\t市\tNOUN\t名詞\t_\t10\tflat\t_\tSpaceAfter=No|NE=L-GPE
14\tで\tで\tADP\t助詞\t_\t10\tcase\t_\tSpaceAfter=No
15\t会った\t会う\tVERB\t動詞\t_\t0\troot\t_\tSpaceAfter=No|NE=U-DATE
16\t。\t。\tPUNCT\t補助記号\t_\t15\tpunct\t_\tSpaceAfter=No
"""


def test_bilou_unit_and_last_tags_are_read():
    """GSD の NE は BILOU。U（単独）と L（末尾）を落とすと 佐藤・日本 が無ラベル、藤本博史 が 藤本 になる（2026-09-23 の不具合）。"""
    sents, stats = g.parse_conllu(CONLLU_BILOU)
    s = sents[0]
    got = [(s.text[a:b], l) for a, b, l in s.entities]
    assert got == [("佐藤", "PERSON"), ("日本銀行", "ORG"), ("藤本博史", "PERSON"), ("宮城県石巻市", "LOCATION")]
    assert [s.text[a:b] for a, b in s.neutral] == ["会った"]
    assert stats["tag_U"] == 2 and stats["tag_L"] == 3 and stats["tag_B"] == 3 and stats["tag_I"] == 2


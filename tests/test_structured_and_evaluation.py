"""構造化データ（transform_records / transform_dataframe）と評価モジュール（evaluation.evaluate）。"""
import pytest

from privacy_masker import PrivacyEngine, Policy
from privacy_masker.detectors import RegexDetector
from privacy_masker.evaluation import EvaluationReport, evaluate
from privacy_masker.operators import Replace, Tokenize
from privacy_masker.structured import transform_dataframe, transform_records
from privacy_masker.types import Entity


class NameDetector:
    """氏名っぽい 2〜4 文字の漢字列だけ拾う簡易検出器（テスト用）。「太郎」単独は拾わない → 列型推定の効果を見る"""
    name = "fake_ner"
    KNOWN = {"田中太郎", "山田花子", "佐藤健一"}

    def detect(self, text):
        out = []
        for k in self.KNOWN:
            i = text.find(k)
            while i >= 0:
                out.append(Entity("PERSON", i, i + len(k), 0.95, self.name)); i = text.find(k, i + 1)
        return out


def make_engine():
    return PrivacyEngine(detectors=[NameDetector(), RegexDetector.default_ja()],
                         policy=Policy(rules={"PERSON": Replace("<氏名>")}, default=Tokenize()), refiner=None)


RECORDS = [
    {"id": 1, "name": "田中太郎", "memo": "田中太郎さんへ電話 03-1234-5678", "age": 34},
    {"id": 2, "name": "山田花子", "memo": "特記なし", "age": 41},
    {"id": 3, "name": "佐藤健一", "memo": "", "age": 29},
    {"id": 4, "name": "鈴木一郎", "memo": None, "age": 50},      # 検出器が知らない名前 → 列型 PERSON で守られる
]


def test_transform_records_infers_person_column_and_masks_unknown_names():
    out = transform_records(make_engine(), RECORDS)
    assert [r["name"] for r in out] == ["<氏名>"] * 4                 # 4 件目は検出できないが列型で置換
    assert out[0]["memo"] == "<氏名>さんへ電話 <PHONE_NUMBER_001>"    # 自由記述列は通常の transform
    assert out[1]["memo"] == "特記なし" and out[2]["memo"] == "" and out[3]["memo"] is None
    assert [r["age"] for r in out] == [34, 41, 29, 50] and [r["id"] for r in out] == [1, 2, 3, 4]   # 非文字列は触らない
    assert RECORDS[3]["name"] == "鈴木一郎"                           # 入力は変更しない


def test_transform_records_explicit_types_and_fields():
    out = transform_records(make_engine(), RECORDS, fields=["memo"], column_types={})
    assert out[3]["name"] == "鈴木一郎"                               # fields 外は触らない
    out = transform_records(make_engine(), RECORDS, column_types={"name": "PERSON"})
    assert out[3]["name"] == "<氏名>"
    with pytest.raises(Exception):
        transform_records(make_engine(), RECORDS, column_types="most_common")


def test_transform_dataframe_roundtrip():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(RECORDS)
    new = transform_dataframe(make_engine(), df)
    assert list(new["name"]) == ["<氏名>"] * 4 and list(new["age"]) == [34, 41, 29, 50]
    assert list(df["name"]) == ["田中太郎", "山田花子", "佐藤健一", "鈴木一郎"]   # 元は不変
    assert new["memo"][3] is None or new["memo"][3] != new["memo"][3]              # None/NaN は保持


def test_evaluate_reports_coverage_danger_and_fp():
    docs = [
        {"text": "田中太郎と山田花子が来た。", "entities": [[0, 4, "PERSON"], [5, 9, "PERSON"]]},
        {"text": "佐藤健一は東京へ。", "entities": [[0, 4, "PERSON"], [5, 7, "LOCATION"]], "neutral": [[7, 8]]},
    ]
    eng = make_engine()
    rep = evaluate(docs, eng)
    assert isinstance(rep, EvaluationReport) and rep.documents == 2 and rep.gold_entities == 4
    assert rep.by_label["PERSON"].total == 3 and rep.by_label["PERSON"].danger == 0
    assert rep.by_label["LOCATION"].miss == 1 and rep.danger == 1                   # 東京は検出されない
    assert rep.fp_chars == 0
    assert rep.leaks == [{"doc": 1, "label": "LOCATION", "start": 5, "end": 7, "coverage": 0.0}]
    md = rep.to_markdown(); assert "| 2 | 4 |" in md and "LOCATION" in md
    # predictions を直接渡す（過剰マスクの計上: gold 外の PERSON 4 字）
    preds = [[Entity("PERSON", 0, 4, 1.0), Entity("PERSON", 5, 9, 1.0), Entity("PERSON", 10, 14, 1.0)], []]
    rep2 = evaluate(docs, predictions=preds, record_leaks=False)
    assert rep2.fp_chars == 4 and rep2.leaks == [] and rep2.by_label["PERSON"].miss == 1
    with pytest.raises(ValueError):
        evaluate(docs)

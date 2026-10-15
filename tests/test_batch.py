import pytest

from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError, DetectorError
from privacy_masker.types import Entity

from .conftest import FakeDetector


class PositionalDetector:
    """テキスト先頭4文字をPERSONとして返すdetector。"""

    name = "positional"

    def detect(self, text):
        return [Entity("PERSON", 0, 4, 1.0)] if len(text) >= 4 else []


class PositionalBatchDetector(PositionalDetector):
    name = "positional_batch"

    def __init__(self):
        self.batch_calls = 0

    def detect_batch(self, texts):
        self.batch_calls += 1
        return [self.detect(t) for t in texts]


def test_batch_default_is_independent_document_scopes():
    engine = PrivacyEngine(detectors=[PositionalDetector()])
    results = engine.transform_batch(["山田太郎です", "佐藤花子です"])
    assert results[0].text == "<PERSON_001>です"
    assert results[1].text == "<PERSON_001>です"  # 各文書で独立に採番


def test_batch_with_scope_id_shares_tokens():
    engine = PrivacyEngine(detectors=[PositionalDetector()], secret_key=b"k" * 16)
    results = engine.transform_batch(["山田太郎です", "山田太郎です", "佐藤花子です"],
                                     scope_id="ds1")
    assert results[0].text == "<PERSON_001>です"
    assert results[1].text == "<PERSON_001>です"  # 同値は同トークン
    assert results[2].text == "<PERSON_002>です"


def test_batch_uses_detect_batch():
    det = PositionalBatchDetector()
    engine = PrivacyEngine(detectors=[det])
    engine.transform_batch(["山田太郎です"] * 5, batch_size=2)
    assert det.batch_calls == 3  # ceil(5/2)


def test_batch_skip_mode_marks_failures_per_text():
    class Boom:
        name = "boom"

        def detect(self, text):
            raise RuntimeError("crash")

    engine = PrivacyEngine(
        detectors=[Boom(), PositionalDetector()], on_detector_error="skip"
    )
    results = engine.transform_batch(["山田太郎です"])
    assert results[0].failed_detectors == ("boom",)
    assert results[0].text == "<PERSON_001>です"


def test_batch_chunk_partial_failure_rolls_back_entire_chunk():
    class PartiallyBadBatchDetector:
        """チャンク内2文書目だけ不正なEntity（endがlenを超える）を返すdetector。"""

        name = "partial_bad"

        def detect_batch(self, texts):
            return [
                [Entity("PERSON", 0, 4, 1.0)],
                [Entity("PERSON", 0, 999, 1.0)],
            ]

    engine = PrivacyEngine(
        detectors=[PartiallyBadBatchDetector()], on_detector_error="skip"
    )
    results = engine.transform_batch(["山田太郎です", "佐藤花子です"], batch_size=2)
    # チャンク全体がロールバックされ、1文書目にもエンティティが適用されない
    assert results[0].text == "山田太郎です"
    assert results[1].text == "佐藤花子です"
    assert results[0].failed_detectors == ("partial_bad",)
    assert results[1].failed_detectors == ("partial_bad",)


@pytest.mark.parametrize("result_count", [1, 3])
def test_batch_result_count_mismatch_raises_detector_error(result_count):
    class WrongCountBatchDetector:
        name = "wrong_count"

        def detect_batch(self, texts):
            return [[Entity("PERSON", 0, 4, 1.0)] for _ in range(result_count)]

    engine = PrivacyEngine(detectors=[WrongCountBatchDetector()])
    with pytest.raises(
        DetectorError,
        match=rf"returned {result_count} results for 2 texts",
    ):
        engine.transform_batch(["山田太郎です", "佐藤花子です"], batch_size=2)


@pytest.mark.parametrize("result_count", [1, 3])
def test_batch_result_count_mismatch_skip_rolls_back_entire_chunk(result_count):
    class WrongCountBatchDetector:
        name = "wrong_count"

        def detect_batch(self, texts):
            return [[Entity("PERSON", 0, 4, 1.0)] for _ in range(result_count)]

    engine = PrivacyEngine(
        detectors=[WrongCountBatchDetector()], on_detector_error="skip"
    )
    results = engine.transform_batch(
        ["山田太郎です", "佐藤花子です"], batch_size=2
    )

    assert [result.text for result in results] == ["山田太郎です", "佐藤花子です"]
    assert [result.failed_detectors for result in results] == [
        ("wrong_count",),
        ("wrong_count",),
    ]


def test_batch_skips_empty_documents_from_detection():
    class RecordingBatchDetector:
        """受け取ったチャンクを記録した上で必ず失敗するdetector。"""

        name = "recording"

        def __init__(self):
            self.seen = []

        def detect_batch(self, texts):
            self.seen.append(list(texts))
            raise RuntimeError("boom")

    det = RecordingBatchDetector()
    engine = PrivacyEngine(detectors=[det], on_detector_error="skip")
    results = engine.transform_batch(["", "山田太郎です"])
    assert results[0].text == ""
    assert results[0].failed_detectors == ()
    assert results[1].failed_detectors == ("recording",)
    assert det.seen == [["山田太郎です"]]  # 空文書はdetect_batchに渡らない


def test_batch_size_zero_raises_configuration_error():
    engine = PrivacyEngine(detectors=[PositionalDetector()])
    with pytest.raises(ConfigurationError):
        engine.transform_batch(["山田太郎です"], batch_size=0)


def test_batch_size_negative_raises_configuration_error():
    # batch_size=-1だとrange(0, n, -1)が空になり、detect_batch対応detectorの
    # 検出が例外なく静かにスキップされてPIIが素通しする回帰を防ぐ
    det = PositionalBatchDetector()
    engine = PrivacyEngine(detectors=[det])
    with pytest.raises(ConfigurationError):
        engine.transform_batch(["山田太郎です"], batch_size=-1)

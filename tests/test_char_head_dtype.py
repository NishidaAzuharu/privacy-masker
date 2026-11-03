"""推論時の autocast。人工 tokenizer と小さなモデルだけを使い、Hub にはアクセスしない。"""
from __future__ import annotations

from contextlib import contextmanager
import warnings

import pytest

torch = pytest.importorskip("torch")

from privacy_masker import char_head as ch  # noqa: E402
from privacy_masker.detectors.char_head import CharHeadNerDetector  # noqa: E402
from privacy_masker.exceptions import ConfigurationError  # noqa: E402
from privacy_masker.types import Entity  # noqa: E402
from tests.test_char_head_predict import ALPHA, ConstModel, TinyTok  # noqa: E402


class AutocastModel(torch.nn.Module):
    """全文を PERSON にする小さな Linear。forward 中の autocast と出力型を記録する。"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, len(ch.LABEL_LIST), bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(-8.0)
            self.linear.weight[ch.LABEL2ID["B-PERSON"], 0] = 8.0
            self.linear.weight[ch.LABEL2ID["I-PERSON"], 1] = 8.0
        self.calls = []

    def forward(self, **batch):
        mask = batch["char_mask"]
        x = torch.zeros((*mask.shape, 2), device=mask.device)
        x[:, 0, 0] = 1.0
        x[:, 1:, 1] = 1.0
        logits = self.linear(x)
        # 引数なしの is_autocast_enabled() は CUDA 用なので CPU を明示する。
        self.calls.append((torch.is_autocast_enabled(mask.device.type), logits.dtype))
        hidden = torch.ones((*batch["input_ids"].shape, 2), device=mask.device)
        return {"logits": logits, "token_hidden": hidden}


@pytest.mark.parametrize("dtype", ["fp16", "float32", "", torch.float32, torch.float64, 16, [], {}])
def test_invalid_dtype_is_configuration_error_before_loading(dtype, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("不正な dtype でモデルを読み込んではいけない")

    monkeypatch.setattr(ch, "load_char_model", unexpected_load)
    with pytest.raises(ConfigurationError, match="dtype"):
        CharHeadNerDetector(model="unused-model", dtype=dtype)


def test_unsupported_device_is_configuration_error():
    with pytest.raises(ConfigurationError, match="利用できる組み合わせ"):
        CharHeadNerDetector(model=AutocastModel(), tokenizer=TinyTok(ALPHA), device="meta", dtype="float16")


@pytest.mark.parametrize("device,dtype", [("cpu", "float16"), ("mps", "bfloat16"), ("cuda", "bfloat16")])
@pytest.mark.parametrize("failure", ["warning", "disabled", "exception"])
def test_unsupported_autocast_is_configuration_error(device, dtype, failure, monkeypatch):
    """対応状況は torch / デバイスに依るので、非対応の環境を人工的に作る。"""
    @contextmanager
    def unsupported(*args, **kwargs):
        if failure == "warning":
            warnings.warn("unsupported dtype", UserWarning)
        elif failure == "exception":
            raise RuntimeError("unsupported dtype")
        yield

    monkeypatch.setattr(torch, "autocast", unsupported)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: False)
    with pytest.raises(ConfigurationError, match="利用できる組み合わせ"):
        CharHeadNerDetector(model=AutocastModel(), tokenizer=TinyTok(ALPHA), device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [None, "bfloat16", torch.bfloat16, "float16", torch.float16])
def test_detect_and_batch_keep_structure_and_fp32_weights(dtype):
    model = AutocastModel()
    detector = CharHeadNerDetector(model=model, tokenizer=TinyTok(ALPHA), device=torch.device("cpu"), dtype=dtype)
    entities = detector.detect("山田太郎")
    batch = detector.detect_batch(["山田太郎", "", "佐藤"])
    assert isinstance(entities, list) and len(entities) == 1
    assert isinstance(entities[0], Entity)
    assert [(e.entity_type, e.start, e.end) for e in entities] == [("PERSON", 0, 4)]
    assert isinstance(entities[0].score, float)
    assert isinstance(entities[0].vector, bytes) and len(entities[0].vector) == 4
    assert all(isinstance(row, list) for row in batch)
    assert batch[:2] == [entities, []]
    assert [(e.entity_type, e.start, e.end) for e in batch[2]] == [("PERSON", 0, 2)]
    expected = torch.float32 if dtype is None else detector.dtype
    assert model.calls == [(dtype is not None, expected)] * 2
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16])
def test_predict_forward_uses_autocast_and_softmax_uses_fp32(dtype, monkeypatch):
    model = AutocastModel()
    seen = []
    softmax = torch.softmax

    def record_softmax(logits, *args, **kwargs):
        seen.append(logits.dtype)
        return softmax(logits, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", record_softmax)
    # 外側が autocast 中でも dtype=None の forward は fp32 に戻す。
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = ch.predict_char_spans(["山田太郎"], model, TinyTok(ALPHA), autocast_dtype=dtype)
    assert len(result[0]) == 1
    assert model.calls == [(dtype is not None, dtype or torch.float32)]
    assert seen == [torch.float32]


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_remain_fail_closed(dtype, value):
    with pytest.raises(FloatingPointError) as exc:
        ch.predict_char_spans(["山田太郎"], ConstModel(value), TinyTok(ALPHA), autocast_dtype=dtype)
    assert ("dtype を bfloat16 か None（fp32）にして再実行" in str(exc.value)) == (dtype is not None)
    assert ch.predict_char_spans(["山田太郎", "佐藤"], ConstModel(value, pad_only=True),
                                 TinyTok(ALPHA), autocast_dtype=dtype) == [[], []]


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_tiny_bert_detector(dtype):
    """既存テストと同じ小さな BERT + 文字ヘッドでも CPU 推論を通す。"""
    from tests.test_char_head_model import CASE, _tiny_model, _tiny_tokenizer

    tok = _tiny_tokenizer()
    torch.manual_seed(0)
    model = _tiny_model(tok)
    with torch.no_grad():
        model.head.classifier.weight.zero_()
        model.head.classifier.bias.fill_(-8.0)
        model.head.classifier.bias[ch.LABEL2ID["I-PERSON"]] = 8.0
    detector = CharHeadNerDetector(model=model, tokenizer=tok, dtype=dtype)
    entities = detector.detect(CASE)
    assert [(e.entity_type, e.start, e.end) for e in entities] == [("PERSON", 0, len(CASE))]
    assert all(p.dtype == torch.float32 for p in model.parameters())

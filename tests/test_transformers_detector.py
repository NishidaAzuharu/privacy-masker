import pytest

from privacy_masker.detectors import TransformersNerDetector
from privacy_masker.exceptions import ConfigurationError


class FakePipeline:
    """HF token-classification pipeline互換のfake。

    「山田」という文字列の出現位置をPERとして返す。チャンクごとに呼ばれる。
    """

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, chunk: str):
        self.calls.append(chunk)
        preds = []
        start = 0
        while True:
            i = chunk.find("山田", start)
            if i < 0:
                break
            preds.append({"entity_group": "PER", "start": i, "end": i + 2, "score": 0.99})
            start = i + 1
        return preds


def test_requires_model_or_pipeline():
    with pytest.raises(ConfigurationError):
        TransformersNerDetector()


def test_label_mapping():
    det = TransformersNerDetector(pipeline=FakePipeline(), label_map={"PER": "PERSON"})
    ents = det.detect("こちらは山田です")
    assert len(ents) == 1
    assert ents[0].entity_type == "PERSON"
    assert ents[0].start == 4 and ents[0].end == 6
    assert ents[0].source == "transformers_ner"


def test_unmapped_label_passes_through_by_default():
    # スペック§5: label_map未登録はラベル名をそのまま通す（フェイルセーフ）
    det = TransformersNerDetector(pipeline=FakePipeline(), label_map={})
    ents = det.detect("こちらは山田です")
    assert ents[0].entity_type == "PER"


def test_drop_unmapped():
    det = TransformersNerDetector(pipeline=FakePipeline(), label_map={}, drop_unmapped=True)
    assert det.detect("こちらは山田です") == []


def test_long_text_chunking_offsets():
    # チャンク境界を跨いでもグローバルoffsetが正しいこと
    fake = FakePipeline()
    det = TransformersNerDetector(
        pipeline=fake, label_map={"PER": "PERSON"}, chunk_chars=10, chunk_overlap=4
    )
    text = "あ" * 20 + "山田" + "い" * 20
    ents = det.detect(text)
    assert len(fake.calls) > 1  # 実際に分割された
    assert len(ents) == 1  # オーバーラップ重複はdedupされる
    assert (ents[0].start, ents[0].end) == (20, 22)
    assert text[ents[0].start:ents[0].end] == "山田"


def test_detect_batch():
    det = TransformersNerDetector(pipeline=FakePipeline(), label_map={"PER": "PERSON"})
    results = det.detect_batch(["山田です", "無関係", "また山田"])
    assert [len(r) for r in results] == [1, 0, 1]


def test_chunk_config_validation():
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(pipeline=FakePipeline(), chunk_chars=10, chunk_overlap=10)


def test_chunk_validation_precedes_pipeline_construction():
    # 無効なchunk設定はmodel指定時でも（transformersのimportより先に）ConfigurationErrorになる
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(model="dummy-model", chunk_chars=10, chunk_overlap=10)


class FakeBatchPipeline(FakePipeline):
    """HF Pipeline同様にstr/list[str]の両方を受け、batch_size kwargを取るfake。"""

    def __init__(self):
        super().__init__()
        self.batch_calls: list[tuple[tuple[str, ...], int | None]] = []

    def __call__(self, inputs, batch_size=None):
        if isinstance(inputs, str):
            return super().__call__(inputs)
        self.batch_calls.append((tuple(inputs), batch_size))
        return [super(FakeBatchPipeline, self).__call__(c) for c in inputs]


def test_batched_detect_single_call_for_chunks():
    # 長文のチャンク群は1回のバッチ呼び出しに集約される
    fake = FakeBatchPipeline()
    det = TransformersNerDetector(
        pipeline=fake, label_map={"PER": "PERSON"},
        chunk_chars=10, chunk_overlap=4, batch_size=4, pipeline_batching=True,
    )
    text = "あ" * 20 + "山田" + "い" * 20
    ents = det.detect(text)
    assert len(fake.batch_calls) == 1          # チャンク毎の逐次呼び出しではない
    assert fake.batch_calls[0][1] == 4         # batch_sizeがpipelineに伝わる
    assert len(ents) == 1
    assert (ents[0].start, ents[0].end) == (20, 22)


def test_batched_detect_batch_flattens_across_texts():
    # 複数文書のチャンクを全てまとめて1回のバッチ呼び出しにする
    fake = FakeBatchPipeline()
    det = TransformersNerDetector(
        pipeline=fake, label_map={"PER": "PERSON"},
        chunk_chars=10, chunk_overlap=4, batch_size=8, pipeline_batching=True,
    )
    texts = ["山田です", "あ" * 20 + "山田" + "い" * 20, "無関係", "また山田"]
    results = det.detect_batch(texts)
    assert len(fake.batch_calls) == 1
    assert [len(r) for r in results] == [1, 1, 0, 1]
    e = results[1][0]
    assert (e.start, e.end) == (20, 22)
    assert texts[1][e.start:e.end] == "山田"


def test_batched_results_order_restored_after_length_sort():
    # 長さ順に並べ替えて推論しても結果は元の文書順に戻る
    fake = FakeBatchPipeline()
    det = TransformersNerDetector(
        pipeline=fake, label_map={"PER": "PERSON"}, pipeline_batching=True,
    )
    texts = ["こちらは山田と申します", "山田", "無関係な長い長い長い文章です"]
    results = det.detect_batch(texts)
    assert [len(r) for r in results] == [1, 1, 0]
    assert (results[0][0].start, results[0][0].end) == (4, 6)
    assert (results[1][0].start, results[1][0].end) == (0, 2)


def test_non_hf_pipeline_falls_back_to_sequential():
    # str専用のカスタムcallableは自動判定で逐次経路になる（後方互換）
    fake = FakePipeline()
    det = TransformersNerDetector(pipeline=fake, label_map={"PER": "PERSON"})
    results = det.detect_batch(["山田です", "また山田"])
    assert [len(r) for r in results] == [1, 1]
    assert all(isinstance(c, str) for c in fake.calls)


def test_custom_callable_with_tokenizer_attribute_keeps_legacy_contract():
    # ONNX等のwrapperがtokenizerを公開していても、genuine HF pipelineで
    # なければtoken数確認やstride kwargを勝手に要求しない。
    class MustNotBeCalled:
        model_max_length = 4

        def __call__(self, *args, **kwargs):
            raise AssertionError("custom tokenizer must not be inspected")

    fake = FakePipeline()
    fake.tokenizer = MustNotBeCalled()
    det = TransformersNerDetector(pipeline=fake, label_map={"PER": "PERSON"})

    entities = det.detect("こちらは山田です")

    assert [(e.start, e.end) for e in entities] == [(4, 6)]


def test_batched_empty_and_whitespace_texts():
    fake = FakeBatchPipeline()
    det = TransformersNerDetector(
        pipeline=fake, label_map={"PER": "PERSON"}, pipeline_batching=True,
    )
    results = det.detect_batch(["", "山田です", "   "])
    assert [len(r) for r in results] == [0, 1, 0]


def test_device_param_rejected_with_custom_pipeline():
    # deviceは自前構築時のみ有効。カスタムpipelineとの併用は設定ミスとして弾く
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(pipeline=FakePipeline(), device="cpu")


class ListOnlyPipeline:
    """list[str]しか受けないpipeline（ONNX注入等の想定）。strで呼ばれたら失敗する。"""

    def __call__(self, inputs, batch_size=None):
        assert not isinstance(inputs, str), "list-only pipeline was called with str"
        out = []
        for chunk in inputs:
            i = chunk.find("山田")
            out.append([] if i < 0 else
                       [{"entity_group": "PER", "start": i, "end": i + 2, "score": 0.9}])
        return out


def test_explicit_batching_uses_list_even_for_single_chunk():
    # pipeline_batching=True なら単一チャンク（短文）でもlist経路で呼ぶ
    det = TransformersNerDetector(
        pipeline=ListOnlyPipeline(), label_map={"PER": "PERSON"}, pipeline_batching=True,
    )
    ents = det.detect("こちらは山田です")          # 単一チャンク
    assert [(e.start, e.end) for e in ents] == [(4, 6)]
    results = det.detect_batch(["山田です"])       # バッチ側も同様
    assert [(e.start, e.end) for e in results[0]] == [(0, 2)]


def test_detect_batch_accepts_generator():
    det = TransformersNerDetector(
        pipeline=FakeBatchPipeline(), label_map={"PER": "PERSON"}, pipeline_batching=True,
    )
    results = det.detect_batch(t for t in ["山田です", "無関係"])
    assert [len(r) for r in results] == [1, 0]


def test_detect_skips_whitespace_only_text():
    fake = FakeBatchPipeline()
    det = TransformersNerDetector(pipeline=fake, label_map={"PER": "PERSON"})
    assert det.detect("   ") == []
    assert fake.calls == [] and fake.batch_calls == []


def test_min_score_filters_low_confidence():
    class LowScorePipeline:
        def __call__(self, chunk):
            return [{"entity_group": "PER", "start": 0, "end": 2, "score": 0.4},
                    {"entity_group": "PER", "start": 4, "end": 6, "score": 0.9}]

    det = TransformersNerDetector(pipeline=LowScorePipeline(), label_map={"PER": "PERSON"},
                                  min_score=0.6)
    ents = det.detect("山田です鈴木")
    assert [(e.start, e.end) for e in ents] == [(4, 6)]


def test_min_score_validation():
    with pytest.raises(ConfigurationError):
        TransformersNerDetector(pipeline=FakePipeline(), min_score=1.5)


def test_min_score_does_not_break_scoreless_unmapped_preds():
    # scoreキーの無いpredがdrop_unmappedで捨てられる経路はv0.5同様に動く
    class NoScorePipeline:
        def __call__(self, chunk):
            return [{"entity_group": "JUNK", "start": 0, "end": 2}]

    det = TransformersNerDetector(pipeline=NoScorePipeline(), label_map={},
                                  drop_unmapped=True)
    assert det.detect("山田です") == []


def _tiny_hf_pipeline(*, aggregation_strategy: str, tokenizer_max_length: int = 8):
    """Create a local, deterministic HF pipeline without downloading a model."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")

    vocab = {
        "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
        "x": 4, "y": 5, "😀": 6, "山": 7, "田": 8,
    }
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Split(pattern="", behavior="isolated")
    raw.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
        model_max_length=tokenizer_max_length,
    )
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=16,
        max_position_embeddings=8, pad_token_id=3,
        num_labels=2, id2label={0: "O", 1: "B-PER"},
        label2id={"O": 0, "B-PER": 1},
    )
    model = transformers.BertForTokenClassification(config)
    # Make every non-special token a deterministic B-PER. This lets the test
    # observe exactly which character offsets reached the model.
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias[:] = torch.tensor([0.0, 10.0])
    return transformers.pipeline(
        "token-classification", model=model, tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy, device="cpu",
    )


def test_hf_over_limit_chunk_uses_stride_and_preserves_codepoint_offsets():
    # max_length=8 includes two special tokens, so only six text tokens fit per
    # model window. The target and suffix lie beyond the formerly truncated area.
    pipeline = _tiny_hf_pipeline(aggregation_strategy="simple")
    det = TransformersNerDetector(pipeline=pipeline, label_map={"PER": "PERSON"})
    text = "😀" * 8 + "山田" + "y" * 5

    entities = det.detect(text)

    spans = {(e.start, e.end, text[e.start:e.end]) for e in entities}
    assert (8, 9, "山") in spans
    assert (9, 10, "田") in spans
    assert max(e.end for e in entities) == len(text)


def test_hf_over_limit_without_aggregation_fails_closed_but_short_text_works():
    # HF native overflow requires a fast tokenizer and aggregation. Preserve
    # injected-pipeline compatibility for safe short inputs, but never allow its
    # implicit truncation for an over-limit input.
    pipeline = _tiny_hf_pipeline(aggregation_strategy="none")
    det = TransformersNerDetector(pipeline=pipeline)

    assert det.detect("山田")
    with pytest.raises(ConfigurationError, match="cannot safely process"):
        det.detect("😀" * 8 + "山田")


def test_hf_model_limit_smaller_than_tokenizer_limit_fails_closed():
    # The HF stride implementation keys off tokenizer.model_max_length. If that
    # is larger than the model's positional limit, it cannot split soon enough.
    pipeline = _tiny_hf_pipeline(aggregation_strategy="simple", tokenizer_max_length=10)
    det = TransformersNerDetector(pipeline=pipeline)

    assert det.detect("山田")
    with pytest.raises(ConfigurationError, match="cannot safely process"):
        det.detect("x" * 7)  # 7 content + 2 specials: model limit 8 is exceeded

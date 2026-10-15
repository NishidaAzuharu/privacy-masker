"""HuggingFace token classificationモデルのAdapter（スペック§5）。

従来の文字数ベースチャンクは互換性と推論コストの上限として維持する。その上で、
HF TokenClassificationPipelineのtokenizer/model上限を超える場合はHF標準の
token strideを使い、サイレント切り捨てを防ぐ。

v0.5: バッチ推論対応。複数文書・複数チャンクを1回のpipeline呼び出しに集約し、
長さ順ソートでパディングの無駄を減らす（end-to-end実測: CPU約1.9倍・MPS約3.8倍、詳細はREADME性能節）。
HF Pipeline以外のカスタムcallable（str→preds）は自動判定で従来の逐次経路になる。
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, NamedTuple, Sequence

from ..exceptions import ConfigurationError
from ..types import Entity


_HF_LIMIT_SENTINEL = 10**20
_HF_DEFAULT_STRIDE_TOKENS = 50


class _HfOverflowConfig(NamedTuple):
    tokenizer: Any
    effective_limit: int | None
    stride: int | None
    can_stride: bool


def _resolve_dtype(dtype):
    """"bfloat16" / "float16" / "float32" の文字列を torch.dtype に。torch.dtype はそのまま。"""
    if isinstance(dtype, str):
        import torch
        name = dtype.replace("torch.", "")
        if not hasattr(torch, name) or not isinstance(getattr(torch, name), torch.dtype):
            raise ConfigurationError(f"unknown dtype {dtype!r}")
        return getattr(torch, name)
    return dtype


def _is_hf_pipeline(pipeline) -> bool:
    """transformersのPipelineインスタンスか（リスト入力+batch_sizeを受けられるか）。

    transformers未導入環境でカスタムcallableを使うケースがあるためimportは遅延し、
    失敗時はFalse（逐次経路）に倒す。
    """
    try:
        from transformers import Pipeline
        return isinstance(pipeline, Pipeline)
    except Exception:  # pragma: no cover  # import不能・環境破損は逐次経路に倒す
        return False


def _finite_token_limit(value) -> int | None:
    """Normalize real model limits and discard HF's effectively-unbounded sentinel."""
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if limit <= 0 or limit >= _HF_LIMIT_SENTINEL:
        return None
    return limit


def _hf_overflow_config(pipeline) -> _HfOverflowConfig | None:
    """Return token-overflow capabilities for a genuine HF NER pipeline.

    An arbitrary callable may happen to expose ``tokenizer`` or ``model`` attributes;
    it must keep the historical callable contract, so capability detection is guarded
    by the concrete HF token-classification pipeline type.
    """
    try:
        from transformers import TokenClassificationPipeline
    except Exception:  # pragma: no cover - optional dependency unavailable/broken
        return None
    if not isinstance(pipeline, TokenClassificationPipeline):
        return None

    tokenizer = getattr(pipeline, "tokenizer", None)
    tokenizer_limit = _finite_token_limit(getattr(tokenizer, "model_max_length", None))
    model = getattr(pipeline, "model", None)
    config = getattr(model, "config", None)
    model_limits = [
        _finite_token_limit(getattr(config, attr, None))
        for attr in ("max_position_embeddings", "n_positions")
    ]
    known_model_limits = [limit for limit in model_limits if limit is not None]
    model_limit = min(known_model_limits) if known_model_limits else None
    known_limits = [limit for limit in (tokenizer_limit, model_limit) if limit is not None]
    effective_limit = min(known_limits) if known_limits else None

    # HF's native stride chunks at tokenizer.model_max_length. It is unsafe when
    # the tokenizer advertises a larger (or unknown) limit than the model itself.
    native_limit_is_safe = (
        tokenizer_limit is not None
        and effective_limit is not None
        and tokenizer_limit == effective_limit
    )
    aggregation = getattr(pipeline, "_postprocess_params", {}).get(
        "aggregation_strategy"
    )
    aggregation = getattr(aggregation, "value", aggregation)
    has_aggregation = aggregation is not None and str(aggregation).lower() != "none"
    is_fast = bool(getattr(tokenizer, "is_fast", False))

    stride = None
    if native_limit_is_safe:
        try:
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        except (AttributeError, TypeError, ValueError, OverflowError):
            special_tokens = -1
        content_capacity = effective_limit - special_tokens
        # At least one token of overlap and one token of forward progress.
        if special_tokens >= 0 and content_capacity >= 2:
            stride = min(
                _HF_DEFAULT_STRIDE_TOKENS,
                max(1, content_capacity // 4),
                content_capacity - 1,
            )

    return _HfOverflowConfig(
        tokenizer=tokenizer,
        effective_limit=effective_limit,
        stride=stride,
        can_stride=is_fast and has_aggregation and stride is not None,
    )


class TransformersNerDetector:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        pipeline=None,
        label_map: Mapping[str, str] | None = None,
        drop_unmapped: bool = False,
        chunk_chars: int = 400,
        chunk_overlap: int = 50,
        batch_size: int = 16,
        min_score: float = 0.0,
        device=None,
        pipeline_batching: bool | None = None,
        name: str = "transformers_ner",
        dtype=None,
        model_kwargs: Mapping | None = None,
    ) -> None:
        """device: None なら transformers の既定（新しめの版は CUDA/MPS を自動選択、無ければ CPU）。"cpu" / "cuda" / "mps" / 0 で明示。
        dtype: torch.dtype または "bfloat16" 等の文字列（GPU では bfloat16 で約 2 倍速い。CPU は float32 のまま推奨）。
        model_kwargs: from_pretrained に渡す追加引数。例 {"attn_implementation": "sdpa"} / {"attn_implementation": "flash_attention_2"}
          （FlashAttention-2 は flash-attn が導入済みで Ampere 以降の CUDA GPU、かつ dtype が fp16/bf16 のときだけ有効。自動では選ばれない）。"""
        # チャンク設定の検証を最初に実行（副作用のあるパイプライン構築より前）
        if chunk_overlap >= chunk_chars:
            raise ConfigurationError("chunk_overlap must be smaller than chunk_chars")
        if batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if not (0.0 <= min_score <= 1.0):
            raise ConfigurationError("min_score must be in [0, 1]")
        if pipeline is not None and (device is not None or dtype is not None or model_kwargs):
            raise ConfigurationError(
                "device / dtype / model_kwargs are only used when the detector builds its own pipeline; "
                "configure your pipeline instead"
            )
        if pipeline is None:
            if model is None:
                raise ConfigurationError("TransformersNerDetector requires model or pipeline")
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "TransformersNerDetector requires the 'transformers' package. "
                    "Install with: pip install 'privacy-masker[transformers]'"
                ) from e
            kwargs = {} if device is None else {"device": device}
            if dtype is not None:
                kwargs["dtype"] = _resolve_dtype(dtype)
            if model_kwargs:
                kwargs["model_kwargs"] = dict(model_kwargs)
            pipeline = hf_pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer if tokenizer is not None else model,
                aggregation_strategy="simple",
                **kwargs,
            )
        self._pipeline = pipeline
        # 明示指定 > 自動判定（HF Pipelineのみリスト入力+batch_sizeの契約を仮定できる）
        self._batching = (
            pipeline_batching if pipeline_batching is not None else _is_hf_pipeline(pipeline)
        )
        self._hf_overflow = _hf_overflow_config(pipeline)
        self.label_map = dict(label_map or {})
        self.drop_unmapped = drop_unmapped
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.min_score = min_score
        self.name = name

    def _chunks(self, text: str) -> Iterator[tuple[int, str]]:
        if len(text) <= self.chunk_chars:
            yield 0, text
            return
        step = self.chunk_chars - self.chunk_overlap
        offset = 0
        while offset < len(text):
            yield offset, text[offset : offset + self.chunk_chars]
            if offset + self.chunk_chars >= len(text):
                break
            offset += step

    def _run_pipeline(self, chunks: Sequence[str]) -> list[list[dict]]:
        """チャンク列を推論して予測リストの列を返す（入力と同順）。

        バッチ経路では長さ順に並べ替えて推論し、パディング量を揃えてから
        元の順序に戻す。逐次経路（非HF callable）は従来どおり1件ずつ呼ぶ。
        """
        if not chunks:
            return []
        hf_kwargs = self._hf_call_kwargs(chunks)
        if not self._batching:
            return [list(self._pipeline(c, **hf_kwargs)) for c in chunks]
        # 長さ1でもリスト経路に統一する: pipeline側の契約が「list[str]を受ける」で
        # 安定し、list専用のカスタムpipeline（ONNX注入等）が入力長依存で壊れない
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        outs = self._pipeline(
            [chunks[i] for i in order], batch_size=self.batch_size, **hf_kwargs
        )
        results: list[list[dict]] = [[] for _ in chunks]
        for pos, i in enumerate(order):
            results[i] = list(outs[pos])
        return results

    def _hf_call_kwargs(self, chunks: Sequence[str]) -> dict[str, int]:
        """Enable HF overflow windows only when a character chunk exceeds its limit.

        Short inputs retain the exact historical pipeline call. If a genuine HF
        pipeline cannot safely create token windows, fail before it can silently
        truncate part of the input. Custom callables never enter this path.
        """
        overflow = self._hf_overflow
        if overflow is None:
            return {}
        if overflow.effective_limit is None:
            raise ConfigurationError(
                "HF token-classification pipeline cannot safely process input "
                "because its maximum token length is unknown"
            )
        try:
            encoded = overflow.tokenizer(
                list(chunks), add_special_tokens=True, truncation=False
            )
            input_ids = encoded["input_ids"]
            if len(input_ids) != len(chunks):
                raise ValueError("tokenizer result count mismatch")
            needs_overflow = any(
                len(ids) > overflow.effective_limit for ids in input_ids
            )
        except Exception:
            raise ConfigurationError(
                "HF token-classification pipeline cannot safely determine input length"
            ) from None
        if not needs_overflow:
            return {}
        if not overflow.can_stride:
            raise ConfigurationError(
                "HF token-classification pipeline cannot safely process an over-limit "
                "chunk; use a fast tokenizer with aggregation_strategy enabled and "
                "matching tokenizer/model maximum lengths"
            )
        assert overflow.stride is not None
        return {"stride": overflow.stride}

    def _collect(
        self, preds: Sequence[dict], offset: int,
        seen: set[tuple[str, int, int]], entities: list[Entity],
    ) -> None:
        for pred in preds:
            label = pred.get("entity_group") or pred.get("entity")
            if label in self.label_map:
                entity_type = self.label_map[label]
            elif self.drop_unmapped:
                continue
            else:
                entity_type = label  # フェイルセーフ: そのまま通してdefault operatorに委ねる
            if float(pred["score"]) < self.min_score:
                continue  # 低信頼度の検出を捨てる（既定0.0=無効。上げるほど見逃しリスク増）
            start = offset + int(pred["start"])
            end = offset + int(pred["end"])
            key = (entity_type, start, end)
            if key in seen:  # オーバーラップ領域の重複検出をdedup
                continue
            seen.add(key)
            entities.append(
                Entity(entity_type, start, end, float(pred["score"]), self.name)
            )

    def detect(self, text: str) -> list[Entity]:
        if not text.strip():
            return []  # detect_batch・engineと同じ扱い（空白のみはpipelineに投げない）
        chunk_list = list(self._chunks(text))
        preds_list = self._run_pipeline([c for _, c in chunk_list])
        entities: list[Entity] = []
        seen: set[tuple[str, int, int]] = set()
        for (offset, _), preds in zip(chunk_list, preds_list):
            self._collect(preds, offset, seen, entities)
        return entities

    def detect_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        # 全文書の全チャンクをまとめてバッチ推論に集約する
        texts = list(texts)  # 2回走査するためiterator入力を防御
        flat_chunks: list[str] = []
        owners: list[tuple[int, int]] = []  # (文書index, 文書内offset)
        for ti, text in enumerate(texts):
            if not text.strip():
                continue  # 空文書はpipelineに投げない（engine.transform_batchと同じ扱い）
            for offset, chunk in self._chunks(text):
                flat_chunks.append(chunk)
                owners.append((ti, offset))
        preds_list = self._run_pipeline(flat_chunks)
        results: list[list[Entity]] = [[] for _ in texts]
        seens: list[set[tuple[str, int, int]]] = [set() for _ in texts]
        for (ti, offset), preds in zip(owners, preds_list):
            self._collect(preds, offset, seens[ti], results[ti])
        return results

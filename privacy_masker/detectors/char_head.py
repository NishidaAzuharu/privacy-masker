"""文字単位BIOヘッド付きモデルを ``Detector`` として使うためのアダプタ。

``privacy_masker.char_head`` は学習と推論で共有するモジュールで、``CharSpan`` を返す。
``Engine`` は ``Detector`` プロトコル（``name`` と ``detect(text) -> list[Entity]``）しか
受け取らないので、その間を埋めるのがこのクラスの役目である。

``TransformersNerDetector`` と差し替えて使える。チャンク分割・統合・fail-closed の判定は
``char_head`` 側に寄せてあるので、ここでは変換とエラー方針だけを持つ。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..exceptions import ConfigurationError
from ..types import Entity
from .base import BatchDetector  # noqa: F401  （プロトコル準拠を明示するためのimport）


class CharHeadNerDetector:
    """``save_char_model`` で保存したディレクトリを読み、文字スパンを ``Entity`` にする。

    ``model`` はローカルのフォルダか、Hugging Face Hub のリポジトリ ID（``owner/name``）。Hub の場合は
    ``privacy_masker.hub.resolve_model_dir`` が推論に必要なファイルだけを取得してキャッシュする
    （``revision`` で固定、``cache_dir`` で置き場所、``local_files_only=True`` でネットワークに出ない）。
    取得できないときは ``ConfigurationError`` で止まる（空のモデルで続けない）。

    ``min_score`` は既定 0.0（無効）。上げると見逃しが増えるので、匿名化用途では
    安易に上げないこと。``merge_strategy`` / ``merge_adjacent`` は
    ``char_head.predict_char_spans`` にそのまま渡す。

    数値異常（NaN/Inf logits）や未被覆文字は ``char_head`` 側が例外にする。ここでは
    **握りつぶさない**。空結果を正常返却すると PII が丸ごと落ちるため、例外はそのまま
    呼び出し側へ上げる（``Engine`` の ``on_detector_error`` で方針を決められる）。

    ``span_vectors=True``（既定）で ``Entity.vector`` に encoder のスパンベクトルを付ける
    （意味を保つ Surrogate の分類ヘッド用。追加の forward は無い）。特徴量は ``vector_features``
    （読み込んだ分類ヘッドの ``features``、無ければ ``span_mean``）。
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        device: Any = "cpu",
        batch_size: int = 16,
        chunk_chars: int | None = None,
        chunk_overlap: int | None = None,
        max_length: int | None = None,
        min_score: float = 0.0,
        merge_strategy: str = "safe_union",
        merge_adjacent: bool = False,
        label_map: Mapping[str, str] | None = None,
        drop_unmapped: bool = False,
        name: str = "char_head_ner",
        model_kwargs: Mapping | None = None,
        span_vectors: bool = True,
        category_head: bool | str = "auto",
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        from .. import char_head as ch

        if model is None:
            raise ConfigurationError("CharHeadNerDetector requires model (path or CharBioNerModel)")
        if batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if not (0.0 <= min_score <= 1.0):
            raise ConfigurationError("min_score must be in [0, 1]")
        if not (isinstance(category_head, bool) or (isinstance(category_head, str) and category_head == "auto")):
            # 1 / 0 は == では True / False と等しいが受け付けない（bool か "auto" だけ）
            raise ConfigurationError(f"unknown category_head: {category_head!r} (use True, False or 'auto')")

        self._ch = ch
        if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
            from ..hub import resolve_model_dir

            # ローカルのフォルダはそのまま、Hub のリポジトリ ID は取得してローカルのスナップショットに解決する
            path = resolve_model_dir(model, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
            self.model = ch.load_char_model(path, device=device, **(dict(model_kwargs or {})))
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
            self.category_head = self._load_category_head(path, category_head)
        else:
            if model_kwargs:
                raise ConfigurationError("model_kwargs is only used when loading from a path")
            if revision is not None or cache_dir is not None or local_files_only:
                raise ConfigurationError("revision / cache_dir / local_files_only are only used when loading from a path or Hub repo id")
            if category_head is True:
                raise ConfigurationError("category_head=True requires a model path")
            self.model = model.to(device).eval() if hasattr(model, "to") else model
            if tokenizer is None:
                raise ConfigurationError("tokenizer is required when passing a model object")
            self.category_head = None
        if not getattr(tokenizer, "is_fast", False):
            raise ConfigurationError("CharHeadNerDetector requires a fast tokenizer (offset_mapping)")

        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.chunk_chars = ch.CHUNK_CHARS if chunk_chars is None else chunk_chars
        self.chunk_overlap = ch.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.max_length = ch.MAX_LENGTH if max_length is None else max_length
        self.min_score = float(min_score)
        self.merge_strategy = merge_strategy
        self.merge_adjacent = bool(merge_adjacent)
        self.label_map = dict(label_map or {})
        self.drop_unmapped = bool(drop_unmapped)
        self.name = name
        self.span_vectors = bool(span_vectors)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_category_head(path: str, category_head: bool | str):
        """``path`` から分類ヘッド（計画 B、``HeadCategorizer``）を読み込む。

        ``category_head="auto"`` はフォルダに ``category_head.json`` が無ければ ``None``（学習済みの
        分類ヘッドが無い旧モデルでも動く）。bool の ``True`` だけが「必ず読む」で、無ければ
        ``FileNotFoundError`` をそのまま伝える。encoder の SHA-256 不一致（``ConfigurationError``）も
        そのまま伝える（フェイルクローズ: 分類ヘッドの不整合を黙って無視しない）。それ以外
        （``False``）は ``None``。値の検査は ``__init__`` が済ませている。
        """
        import os

        from ..category_head import CATEGORY_HEAD_CONFIG, HeadCategorizer

        if category_head is True:
            return HeadCategorizer.load(path)
        if category_head == "auto" and os.path.exists(os.path.join(path, CATEGORY_HEAD_CONFIG)):
            return HeadCategorizer.load(path)
        return None

    @property
    def vector_features(self) -> str:
        """`Entity.vector` に入れる特徴量。読み込んだ分類ヘッドが要求するもの（`category_head.features`）、無ければ `span_mean`。"""
        head = getattr(self, "category_head", None)
        return getattr(head, "features", None) or "span_mean"

    def _to_entities(self, spans: Sequence[Any], vectors: Sequence[bytes | None] | None = None) -> list[Entity]:
        out: list[Entity] = []
        for i, sp in enumerate(spans):
            label = sp.label
            if self.label_map:
                if label in self.label_map:
                    label = self.label_map[label]
                elif self.drop_unmapped:
                    continue
            if sp.score < self.min_score:
                continue      # 低信頼度を捨てる（既定 0.0 = 無効）
            out.append(
                Entity(
                    label,
                    sp.start,
                    sp.end,
                    float(sp.score),
                    self.name,
                    vector=(vectors[i] if vectors is not None else None),
                )
            )
        return out

    def detect(self, text: str) -> list[Entity]:
        if not text:
            return []
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: Sequence[str]) -> list[list[Entity]]:
        texts = list(texts)
        if not texts:
            return []
        # 空文書は推論に渡さない（predict_char_spans は空文字列を 0 チャンクとして扱うが、
        # 呼び出し側の対応順序を崩さないためここで抜いて後で戻す）
        idx = [i for i, t in enumerate(texts) if t]
        out: list[list[Entity]] = [[] for _ in texts]
        if not idx:
            return out
        vecs: list[list[bytes | None]] | None = [] if self.span_vectors else None
        spans = self._ch.predict_char_spans(
            [texts[i] for i in idx],
            self.model,
            self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            chunk_chars=self.chunk_chars,
            chunk_overlap=self.chunk_overlap,
            max_length=self.max_length,
            merge_adjacent=self.merge_adjacent,
            merge_strategy=self.merge_strategy,
            vectors_out=vecs,
            vector_features=self.vector_features,
        )
        if len(spans) != len(idx):
            raise RuntimeError(
                f"predict_char_spans が {len(spans)} 件返しました（{len(idx)} 件のはず）"
            )
        for j, (i, sp) in enumerate(zip(idx, spans)):
            out[i] = self._to_entities(sp, vecs[j] if vecs is not None else None)
        return out

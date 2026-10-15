"""分類ヘッド（計画 B、設計 §8）: NER encoder のスパンベクトル → カテゴリ。

NER encoder のスパンベクトル（`char_head.encode_span_vectors` / `Entity.vector`）から
ORG / FACILITY / LOCATION のカテゴリを決める、型ごとの線形層 + softmax の学習済みヘッド。
学習は `tools/train_category_head.py`（本モジュールの定数・`encoder_sha256`・`_linear_softmax`
を import して使う。重複を残さないため実装はここに一本化する）。

読み込み・純 Python 推論は `HeadCategorizer` が行う。安全な遅延 import のため、このモジュールは
トップレベルで safetensors / numpy / torch を import しない（base パッケージが `[transformers]`
extra 無しでも import できることを保つため）:

  * `safetensors.numpy.load_file` は `HeadCategorizer.load` の中だけで import する。
  * `char_head.vector_to_floats` は使わない。`char_head` モジュール自体がトップレベルで
    torch を import するため、同じロジック（float16 little-endian のデコード）を
    `_decode_vector` としてここに複製する（バイト互換は `tests/test_category_head.py` で確認）。
  * MLP（`head_type: "mlp"`）は numpy で推論する（`_gelu_tanh` / `_mlp_softmax`、いずれも
    関数の中でだけ import する）。検出器が動く環境には torch と一緒に numpy が必ず入っているため、
    実害はない。線形（`head_type: "linear"`、既定）は従来どおり純 Python のまま
    （速度と挙動を変えないため、numpy は使わない）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Sequence

from .exceptions import ConfigurationError
from .operators.semantic import CATEGORIES, OTHER

CATEGORY_HEAD_WEIGHTS = "category_head.safetensors"
CATEGORY_HEAD_CONFIG = "category_head.json"
# NER encoder の重みファイル名（encoder_sha256 が対象にする）
ENCODER_FILE = "model.safetensors"
# NER encoder の HF config（hidden_size を読む。無い/壊れていれば次元照合はスキップする）
ENCODER_CONFIG_FILE = "config.json"

# char_head.FEATURES と同じ値（char_head はトップレベルで torch を import するので、ここでは import せず複製する。
# 一致は tests/test_category_head.py で確認）
FEATURES: tuple[str, ...] = ("span_mean", "span_ctx")

# ヘッドの種類。linear は従来の 1 層 Linear + softmax、mlp は Linear → GELU(tanh) → Linear → softmax
HEAD_TYPES: tuple[str, ...] = ("linear", "mlp")


def _min_margin_for(kind: str, raw_margin: object) -> float:
    """`category_head.json` の `min_margin`（kind → 数値の辞書、または全 kind 共通の数値）から kind の値を取る。

    null・欠落・数値でない値（文字列・bool）・[0, 1] の外は ``ConfigurationError``（採用値の無いヘッドを
    黙って既定値で動かさない。学習ツールは min_margin の見つからない kind を保存しない）。
    """
    value = raw_margin.get(kind) if isinstance(raw_margin, dict) else raw_margin
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ConfigurationError(
            f"category head has no valid min_margin for {kind!r} (expected a number in [0, 1], got {value!r})"
        )
    return float(value)


def _decode_vector(data: bytes) -> tuple[float, ...]:
    """スパンベクトルのバイト列を float の列に戻す（float16 little-endian）。

    `char_head.vector_to_floats` と同じロジックの複製。`char_head` はトップレベルで torch を
    import するため、torch 無しでも読み込める本モジュールでは呼び出さずにここへ複製する。
    """
    import struct

    n = len(data) // 2
    return struct.unpack(f"<{n}e", data[: n * 2])


def _linear_softmax(W: Sequence[Sequence[float]], b: Sequence[float], x: Sequence[float]) -> list[float]:
    """`nn.Linear` 1 層 + softmax を純 Python で計算する（学習側 `train_category_head.py` と共有）。"""
    logits = [sum(wi * xi for wi, xi in zip(row, x)) + bi for row, bi in zip(W, b)]
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    z = sum(exps)
    return [v / z for v in exps]


def _gelu_tanh(x):
    """GELU の tanh 近似（学習側 `torch.nn.GELU(approximate="tanh")` と同じ式）。numpy 配列を受ける。"""
    import numpy as np

    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _mlp_softmax(W1, b1, W2, b2, x) -> list[float]:
    """入力 → Linear → GELU(tanh) → Linear → softmax を numpy で計算する（学習側 `train_category_head.py` と共有）。"""
    import numpy as np

    h = _gelu_tanh(W1 @ np.asarray(x, dtype=np.float32) + b1)
    logits = W2 @ h + b2
    logits = logits - logits.max()
    e = np.exp(logits)
    return (e / e.sum()).tolist()


def encoder_sha256(model_dir: str) -> str:
    """`model_dir` の `model.safetensors`（NER encoder の重み）のチャンク SHA-256。"""
    h = hashlib.sha256()
    with open(os.path.join(model_dir, ENCODER_FILE), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encoder_hidden_size(model_dir: str) -> int | None:
    """`model_dir` の `config.json`（NER encoder の HF config）から `hidden_size` を読む。

    実在するモデルフォルダには必ずあるが、テスト用のフィクスチャ等で無い・JSON が壊れている・
    `hidden_size` が整数でない場合は ``None``（`_check_input_dim` は次元の厳密な照合を諦め、
    `span_ctx` が 3 の倍数であることだけ確認する）。
    """
    path = os.path.join(model_dir, ENCODER_CONFIG_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    value = cfg.get("hidden_size") if isinstance(cfg, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check_input_dim(kind: str, features: str, input_dim: int, hidden_size: int | None) -> None:
    """ヘッドの入力次元 `input_dim` が `features`（encoder の `hidden_size` から決まる期待次元）と
    合っているか確かめる（設計 §2.1: 「`HeadCategorizer.load` は features と重みの入力次元が合う
    ことを検査し、合わなければ ConfigurationError」）。

    `hidden_size` が分かれば厳密に照合する（`span_mean` は ×1、`span_ctx` は ×3）。分からなければ
    （`config.json` が無い・壊れている等）照合は諦めるが、`span_ctx` は 3 区画を連結した次元のはず
    なので、せめて 3 の倍数であることだけは確認する。
    """
    if hidden_size is not None:
        expected = hidden_size * (3 if features == "span_ctx" else 1)
        if input_dim != expected:
            raise ConfigurationError(
                f"category head for {kind!r} has features {features!r} but input dim {input_dim} "
                f"does not match encoder hidden_size {hidden_size} (expected {expected})"
            )
    elif features == "span_ctx" and input_dim % 3 != 0:
        raise ConfigurationError(
            f"category head for {kind!r} has features 'span_ctx' but input dim {input_dim} "
            "is not divisible by 3 (encoder hidden_size is unknown, so only this weaker check applies)"
        )


class HeadCategorizer:
    """学習済み分類ヘッド（`category_head.safetensors` / `category_head.json`）を読み込み、
    NER encoder のスパンベクトルから型ごとのカテゴリを決める。純 Python 推論で torch に依存しない。
    """

    def __init__(
        self,
        heads: dict[str, tuple],
        categories: dict[str, list[str]],
        min_margin: dict[str, float],
        features: str = "span_mean",
        head_type: str = "linear",
    ) -> None:
        self._heads = heads
        self.categories = categories
        self.min_margin = min_margin
        self.features = features
        self.head_type = head_type

    @classmethod
    def load(cls, model_dir: str, head_dir: str | None = None) -> "HeadCategorizer":
        """`model_dir` から分類ヘッドを読み込む。

        `head_dir` を渡すと `category_head.json` / `category_head.safetensors` はそこから読むが、
        encoder の SHA-256 は常に `model_dir` の `model.safetensors` と照合する（候補フォルダの
        ヘッドを本番 encoder で評価する用。encoder はどちらの場合も `model_dir` のものを使う）。

        `category_head.json` / `category_head.safetensors` が無ければ ``FileNotFoundError``
        （NER encoder の `model.safetensors` が無い場合も、後段の `encoder_sha256` の呼び出しで
        自然に ``FileNotFoundError`` になる）。safetensors が入っていなければ ``ImportError``
        （`[transformers]` extra が要る。ファイルの破損とは区別する）。JSON/safetensors が壊れている・
        JSON の形が違う（dict でない・categories が kind → list の dict でない）・encoder の SHA-256 が
        一致しない・head_type が未知（`HEAD_TYPES` に無い値）・categories が未知・kind の categories が空・重みの形状が合わない
        （weight/w1 が 2 次元・bias/b1 が 1 次元でない、行数が categories と違う等）・features と重みの
        入力次元が合わない（encoder の `hidden_size` から決まる期待次元との不一致。`_check_input_dim`）・
        kind の min_margin が無い（null・欠落・[0, 1] 外）・features が未知（`FEATURES` に無い値）、は
        すべて ``ConfigurationError`` にする。
        """
        src = head_dir or model_dir
        cfg_path = os.path.join(src, CATEGORY_HEAD_CONFIG)
        weights_path = os.path.join(src, CATEGORY_HEAD_WEIGHTS)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(cfg_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(weights_path)

        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            from safetensors.numpy import load_file  # 遅延 import（本体パッケージを torch/safetensors 無しでも import 可能に保つ）

            weights = load_file(weights_path)
        except ImportError:
            raise  # safetensors が無い（extra 未導入）のはファイルの破損ではない。そのまま伝える
        except Exception as exc:  # noqa: BLE001 -- JSON 破損は json.JSONDecodeError(ValueError)、safetensors の破損は
            # safetensors 独自の SafetensorError（Exception 直下、型を絞れない）を投げるため、
            # ここで一括して ConfigurationError に正規化する。
            raise ConfigurationError(f"category head in {src} is corrupt or unreadable: {exc}") from exc

        if not isinstance(cfg, dict):
            raise ConfigurationError(f"{CATEGORY_HEAD_CONFIG} must be a JSON object, got {type(cfg).__name__}")
        raw_categories = cfg.get("categories", {})
        if not isinstance(raw_categories, dict) or not all(isinstance(v, list) for v in raw_categories.values()):
            raise ConfigurationError(f"{CATEGORY_HEAD_CONFIG}: 'categories' must map each kind to a list of names")
        if cfg.get("encoder_sha256") != encoder_sha256(model_dir):
            raise ConfigurationError(
                "category head was trained for a different encoder (model.safetensors SHA-256 mismatch)"
            )
        head_type = cfg.get("head_type", "linear")
        if head_type not in HEAD_TYPES:
            raise ConfigurationError(f"category head has unknown head_type {head_type!r} (expected one of {HEAD_TYPES})")
        features = cfg.get("features", "span_mean")
        if features not in FEATURES:
            raise ConfigurationError(f"category head has unknown features {features!r} (expected one of {FEATURES})")

        hidden_size = _encoder_hidden_size(model_dir)

        heads: dict[str, tuple] = {}
        cats: dict[str, list[str]] = {}
        mm: dict[str, float] = {}
        raw_margin = cfg.get("min_margin")
        for kind, names in raw_categories.items():
            if kind not in CATEGORIES or any(n not in CATEGORIES[kind] for n in names):
                raise ConfigurationError(f"category head has unknown categories for {kind!r}")
            if not names:
                # 0 行の重みは形状検査を素通りし、推論時に黙って OTHER へ落ちる。読み込み時に拒否する（linear / mlp 共通）
                raise ConfigurationError(f"category head has no categories for {kind!r}")
            if head_type == "linear":
                try:
                    W_arr = weights[f"{kind}.weight"]
                    b_arr = weights[f"{kind}.bias"]
                except KeyError as exc:
                    raise ConfigurationError(f"category head is missing weights for {kind!r}") from exc
                if W_arr.ndim != 2 or b_arr.ndim != 1:
                    raise ConfigurationError(
                        f"category head shape mismatch for {kind!r} (weight must be 2-D and bias 1-D, "
                        f"got {W_arr.ndim}-D and {b_arr.ndim}-D)"
                    )
                _check_input_dim(kind, features, W_arr.shape[1], hidden_size)
                W = W_arr.tolist()
                b = b_arr.tolist()
                if len(W) != len(names) or len(b) != len(names):
                    raise ConfigurationError(f"category head shape mismatch for {kind!r}")
                heads[kind] = (W, b)
            else:
                try:
                    W1 = weights[f"{kind}.w1"]
                    b1 = weights[f"{kind}.b1"]
                    W2 = weights[f"{kind}.w2"]
                    b2 = weights[f"{kind}.b2"]
                except KeyError as exc:
                    raise ConfigurationError(f"category head is missing mlp weights for {kind!r}") from exc
                if (
                    W1.ndim != 2
                    or b1.ndim != 1
                    or W2.ndim != 2
                    or b2.ndim != 1
                    or W1.shape[0] != b1.shape[0]
                    or W2.shape[1] != W1.shape[0]
                    or W2.shape[0] != len(names)
                    or b2.shape[0] != len(names)
                ):
                    raise ConfigurationError(f"category head shape mismatch for {kind!r} (mlp)")
                _check_input_dim(kind, features, W1.shape[1], hidden_size)
                heads[kind] = (
                    W1.astype("float32"),
                    b1.astype("float32"),
                    W2.astype("float32"),
                    b2.astype("float32"),
                )
            cats[kind] = list(names)
            mm[kind] = _min_margin_for(kind, raw_margin)
        return cls(heads, cats, mm, features=features, head_type=head_type)

    def _probs(self, head: tuple, x: tuple[float, ...]) -> list[float] | None:
        """`head`（linear なら `(W, b)`、mlp なら `(W1, b1, W2, b2)`）と入力 `x` から確率のリストを返す。
        次元が合わなければ ``None``（呼び出し側でフォールバックする）。"""
        if self.head_type == "linear":
            W, b = head
            if not W or len(W[0]) != len(x):
                return None
            return _linear_softmax(W, b, x)
        W1, b1, W2, b2 = head
        if W1.shape[1] != len(x):
            return None
        return _mlp_softmax(W1, b1, W2, b2, x)

    def __call__(self, kind: str, vector: bytes) -> tuple[str, float]:
        """`vector`（`char_head` が返すスパンベクトルのバイト列）から `(カテゴリ, margin)` を返す。

        kind が学習済みでない・vector が空・次元が合わない場合は `(OTHER[kind], 0.0)` にフォール
        バックする（黙って誤ったカテゴリを出さない）。margin は 1 位と 2 位の確率差。
        """
        head = self._heads.get(kind)
        if head is None or not vector:
            return OTHER.get(kind, "other"), 0.0
        probs = self._probs(head, _decode_vector(vector))
        if probs is None:
            return OTHER.get(kind, "other"), 0.0
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        top = probs[order[0]]
        second = probs[order[1]] if len(order) > 1 else 0.0
        return self.categories[kind][order[0]], top - second

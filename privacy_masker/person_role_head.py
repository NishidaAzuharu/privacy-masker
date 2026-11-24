"""人名の言及文脈から public 確率を返す任意のヘッド。torch は import しない。"""
from __future__ import annotations

import json
import math
import os

from .category_head import (
    HEAD_TYPES,
    _decode_vector,
    _encoder_hidden_size,
    _head_probs,
    _load_head_weights,
    encoder_sha256,
)
from .exceptions import ConfigurationError

PERSON_ROLE_HEAD_CONFIG = "person_role_head.json"
PERSON_ROLE_HEAD_WEIGHTS = "person_role_head.safetensors"


def _validate_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigurationError("person_role threshold must be a number in [0, 1]")
    return float(value)


class PersonRoleHead:
    """スパンの文脈ベクトル → public 確率。公知への昇格には使わない。"""

    def __init__(self, head: tuple, *, head_type: str, threshold: float) -> None:
        self._head = head
        self.head_type = head_type
        self.threshold = _validate_threshold(threshold)
        self.features = "span_ctx"

    @classmethod
    def load(cls, model_dir: str, head_dir: str | None = None) -> PersonRoleHead:
        """ヘッドを読み、model_dir の encoder と照合する。head_dir はヘッドだけの置き場所。

        欠落ファイルは FileNotFoundError、不正な内容・形状・encoder 不一致は
        ConfigurationError。safetensors は読み込み時にだけ import する。
        """
        src = head_dir or model_dir
        cfg_path = os.path.join(src, PERSON_ROLE_HEAD_CONFIG)
        weights_path = os.path.join(src, PERSON_ROLE_HEAD_WEIGHTS)
        for path in (cfg_path, weights_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            from safetensors.numpy import load_file

            weights = load_file(weights_path)
        except ImportError:
            raise
        except Exception as exc:  # safetensors の破損例外も正規化する
            raise ConfigurationError("person role head is corrupt or unreadable") from exc
        if not isinstance(cfg, dict):
            raise ConfigurationError(f"{PERSON_ROLE_HEAD_CONFIG} must be a JSON object")
        if type(cfg.get("format_version")) is not int or cfg["format_version"] != 1:
            raise ConfigurationError("person role head has unknown format_version")
        head_type = cfg.get("head_type")
        if head_type not in HEAD_TYPES:
            raise ConfigurationError("person role head has unknown head_type")
        if cfg.get("features") != "span_ctx":
            raise ConfigurationError("person role head features must be 'span_ctx'")
        if cfg.get("labels") != {"person": ["private", "public"]}:
            raise ConfigurationError("person role head labels must be {'person': ['private', 'public']}")
        threshold = _validate_threshold(cfg.get("threshold"))
        if cfg.get("encoder_sha256") != encoder_sha256(model_dir):
            raise ConfigurationError("person role head encoder SHA-256 mismatch")
        head = _load_head_weights(weights, "person", 2, "span_ctx", _encoder_hidden_size(model_dir),
                                  head_type, description="person role head", linear_float32=True)
        if head_type == "mlp" and "hidden" in cfg:
            if type(cfg["hidden"]) is not int or cfg["hidden"] != head[0].shape[0]:
                raise ConfigurationError("person role head hidden shape mismatch")
        return cls(head, head_type=head_type, threshold=threshold)

    def __call__(self, vector: bytes) -> float | None:
        """float16 little-endian の span_ctx を受ける。空・不正次元は None。"""
        if not vector or len(vector) % 2:
            return None
        x = _decode_vector(vector)
        if not all(math.isfinite(v) for v in x):
            return None
        probs = _head_probs(self._head, self.head_type, x)
        if probs is None or not math.isfinite(probs[1]):
            return None
        return probs[1]

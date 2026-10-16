"""設定ファイル（YAML / JSON / dict）から PrivacyEngine を組み立てる。

    from privacy_masker.config import load_engine, build_engine
    engine = load_engine("privacy_masker.yaml")          # .yaml は PyYAML（pip install pyyaml）が必要。無ければ .json を使う
    engine = build_engine({...})                         # dict から直接

書式（すべて任意。省略時は既定値）:

    detectors:
      - type: char_head             # CharHeadNerDetector（文字単位ヘッド。model はフォルダか Hub の ID、revision / device 等も渡せる）
        model: az225/privacy-masker-ja-pii-ner-30m
      - type: transformers          # TransformersNerDetector
        model: /path/to/best        # または model_env: PM_NER_MODEL
        device: cpu                 # 省略時は transformers の自動選択（CUDA / MPS / CPU）
        batch_size: 16
        dtype: bfloat16             # GPU 向け。省略時 float32
        model_kwargs: {attn_implementation: sdpa}   # flash_attention_2 は flash-attn 導入済みの CUDA GPU でのみ
      - type: regex_ja              # RegexDetector.default_ja(dates=false)
        dates: false
      - type: regex                 # 自前パターン
        patterns:
          - {entity_type: PROJECT, regex: "PJ-\\\\d{4}", score: 0.6, context: [案件, プロジェクト], context_score: 1.0}
      - type: handles               # HandleDetector
    refiner: span                   # span | morphological（auto: fugashi があれば fugashi、無ければ janome）| janome | fugashi | none
                                    # または {type: morphological, backend: fugashi}
    enhancer: default               # default | none | {rules: {PERSON: [氏名]}, boost: 0.3}
    annotators:
      - type: identifiability
        level: standard
        knowledge: /path/jawiki_titles_ns0.pmk     # または knowledge_env
    policy:
      preset: strict                # strict | dataset | standard | minimal（rules/default と併用可: preset を土台に上書き）
      default: {op: tokenize, strategy: sequential}
      rules:
        PERSON: {op: surrogate, kind: person}
        ADDRESS: {op: surrogate, kind: address, keep: city}
        PHONE_NUMBER: {op: mask, keep_suffix: 4}
        EMAIL: {op: hash, algorithm: sha256, length: 12}
        DATE: {op: date_shift, max_days: 365}
        AGE: {op: bucketize, size: 10}
    secret_key_env: PM_SECRET_KEY   # または secret_key（文字列。設定ファイルに鍵を書くのは非推奨）
    options:                        # 既定の呼び出しオプション
      entities: [PERSON, ADDRESS, PHONE_NUMBER]
      score_threshold: {DEFAULT: 0.5}
      allow_list: [株式会社アオバ]
      allow_list_match: exact
      deny_list: {ハヤブサ計画: PROJECT}
    on_detector_error: raise

Custom オペレータ（Python の関数）は設定ファイルからは作れない。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from . import operators as ops
from .enhancer import ContextEnhancer
from .engine import PrivacyEngine
from .exceptions import ConfigurationError
from .policy import Policy
from .refiner import SpanRefiner

_OPERATORS = {
    "tokenize": ops.Tokenize, "surrogate": ops.Surrogate, "mask": ops.Mask, "replace": ops.Replace,
    "redact": ops.Redact, "keep": ops.Keep, "hash": ops.Hash, "encrypt": ops.Encrypt,
    "date_shift": ops.DateShift, "bucketize": ops.Bucketize, "format_preserving": ops.FormatPreserving,
}


def _env(spec: Mapping[str, Any], key: str) -> Any:
    """`key` か `key_env`（環境変数名）のどちらかから値を取る。"""
    if key in spec:
        return spec[key]
    env = spec.get(f"{key}_env")
    if env:
        v = os.environ.get(env)
        if not v:
            raise ConfigurationError(f"environment variable {env!r} (for {key}) is not set")
        return v
    return None


def _build_operator(spec: Mapping[str, Any]):
    if isinstance(spec, str):
        spec = {"op": spec}
    kind = spec.get("op")
    if kind not in _OPERATORS:
        raise ConfigurationError(f"unknown operator {kind!r}; choose from {sorted(_OPERATORS)}")
    kwargs = {k: v for k, v in spec.items() if k != "op"}
    for k in ("key", "salt"):
        if isinstance(kwargs.get(k), str):
            kwargs[k] = kwargs[k].encode("utf-8")
    if kind in ("encrypt",) and "key" not in kwargs and "key_env" in kwargs:
        kwargs["key"] = os.environ.get(kwargs.pop("key_env"), "") or None
        if kwargs["key"] is None:
            raise ConfigurationError("encrypt: key_env is not set")
    try:
        return _OPERATORS[kind](**kwargs)
    except TypeError as e:
        raise ConfigurationError(f"operator {kind!r}: {e}") from None


def _build_detector(spec: Mapping[str, Any]):
    kind = spec.get("type")
    if kind == "char_head":
        from .detectors import CharHeadNerDetector
        model = _env(spec, "model")
        if not model:
            raise ConfigurationError("char_head detector requires model or model_env")
        kw = {k: v for k, v in spec.items() if k not in ("type", "model", "model_env")}
        return CharHeadNerDetector(model=model, **kw)
    if kind == "transformers":
        from .detectors import TransformersNerDetector
        model = _env(spec, "model")
        if not model:
            raise ConfigurationError("transformers detector requires model or model_env")
        kw = {k: v for k, v in spec.items() if k not in ("type", "model", "model_env")}
        return TransformersNerDetector(model=model, **kw)
    if kind in ("regex_ja", "regex_default_ja"):
        from .detectors import RegexDetector
        return RegexDetector.default_ja(dates=bool(spec.get("dates", False)))
    if kind == "regex":
        from .detectors import RegexDetector, RegexPattern
        pats = []
        for p in spec.get("patterns", []):
            try:
                rx = re.compile(p["regex"])
            except (KeyError, re.error) as e:
                raise ConfigurationError(f"regex pattern invalid: {e}") from None
            pats.append(RegexPattern(p["entity_type"], rx, None, float(p.get("score", 1.0)), tuple(p.get("context", ())),
                                     float(p.get("context_score", 1.0)), int(p.get("context_window", 30)), bool(p.get("require_context", False))))
        if not pats:
            raise ConfigurationError("regex detector requires patterns")
        return RegexDetector(pats, name=spec.get("name", "regex"))
    if kind == "handles":
        from .detectors import HandleDetector
        return HandleDetector()
    raise ConfigurationError(f"unknown detector type {kind!r}")


def _build_refiner(spec):
    if spec in (None, "span", {"type": "span"}):
        return SpanRefiner()
    if spec in ("none", False, {"type": "none"}):
        return None
    if spec in ("morphological", "janome", "fugashi") or (isinstance(spec, Mapping) and spec.get("type") == "morphological"):
        from .refiner_janome import MorphologicalRefiner
        if isinstance(spec, Mapping):
            kw = {k: v for k, v in spec.items() if k != "type"}
            return MorphologicalRefiner(**kw)
        return MorphologicalRefiner(backend="auto" if spec == "morphological" else spec)
    raise ConfigurationError(f"unknown refiner {spec!r}")


def _build_enhancer(spec):
    if spec in (None, "none", False):
        return None
    if spec in ("default", True, {}):
        return ContextEnhancer()
    if isinstance(spec, Mapping):
        return ContextEnhancer(spec.get("rules"), boost=float(spec.get("boost", 0.3)), max_score=float(spec.get("max_score", 1.0)),
                               window=int(spec.get("window", 20)), extend_default=bool(spec.get("extend_default", True)))
    raise ConfigurationError(f"unknown enhancer {spec!r}")


def _build_annotator(spec: Mapping[str, Any]):
    if spec.get("type") != "identifiability":
        raise ConfigurationError(f"unknown annotator type {spec.get('type')!r}")
    from .annotators import IdentifiabilityAnnotator, PublicKnowledge
    knowledge = None
    path = _env(spec, "knowledge")
    if path:
        knowledge = PublicKnowledge.load(path)
    return IdentifiabilityAnnotator(level=spec.get("level", "standard"), knowledge=knowledge,
                                    window=int(spec.get("window", 40)))


def _build_policy(spec: Mapping[str, Any] | None) -> Policy:
    if not spec:
        return Policy.preset("strict")
    base = Policy.preset(spec["preset"]) if spec.get("preset") else Policy()
    rules = dict(base.rules)
    for t, op in (spec.get("rules") or {}).items():
        rules[t] = _build_operator(op)
    default = _build_operator(spec["default"]) if spec.get("default") else base.default
    return Policy(rules=rules, default=default)


def build_engine(config: Mapping[str, Any]) -> PrivacyEngine:
    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    dets = [_build_detector(d) for d in config.get("detectors") or []]
    if not dets:
        raise ConfigurationError("config needs at least one detector")
    opts = dict(config.get("options") or {})
    secret_key = config.get("secret_key")
    kw: dict[str, Any] = {}
    if secret_key is not None:
        kw["secret_key"] = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    elif config.get("secret_key_env"):
        kw["secret_key_env"] = config["secret_key_env"]
    return PrivacyEngine(
        detectors=dets,
        policy=_build_policy(config.get("policy")),
        refiner=_build_refiner(config.get("refiner")),
        enhancer=_build_enhancer(config.get("enhancer")),
        annotators=[_build_annotator(a) for a in config.get("annotators") or []],
        on_detector_error=config.get("on_detector_error", "raise"),
        entities=opts.get("entities"), score_threshold=opts.get("score_threshold"),
        allow_list=opts.get("allow_list"), allow_list_match=opts.get("allow_list_match", "exact"),
        deny_list=opts.get("deny_list"), **kw,
    )


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("YAML config requires PyYAML: pip install pyyaml (or use JSON)") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigurationError("config file must contain a mapping at top level")
    return data


def load_engine(path: str) -> PrivacyEngine:
    return build_engine(load_config(path))

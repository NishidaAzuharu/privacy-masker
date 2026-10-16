"""PrivacyEngine（スペック§3）。"""
from __future__ import annotations

import re
import dataclasses
from typing import Iterable, Mapping, Sequence

from .annotators.base import Annotator
from .crypto.base import CryptoProvider
from .detectors.base import Detector
from .encoding import decode_fields
from .enhancer import ContextEnhancer
from .exceptions import (
    AnnotatorError, ConfigurationError, DetectorError, PrivacyMaskerError, RestoreError,
)
from .operators import Surrogate
from .policy import Policy
from .refiner import SpanRefiner
from .resolver import EntityResolver
from .scopes import DocumentContext, StoreContext
from .secrets_types import SecretBytes
from .stores import MemoryStore
from .stores.base import PseudonymStore
from .types import Entity, ExplainedEntity, RestoreResult, TransformedEntity, TransformResult

_ERROR_MODES = ("raise", "skip")
_ALLOW_LIST_MATCH = ("exact", "regex")
_DENY_LIST_DEFAULT_TYPE = "CUSTOM"
_PUBLIC_SUFFIX = "_PUBLIC"


class _Options:
    """呼び出し時オプション。engine の既定値を呼び出し引数で上書きする。"""

    __slots__ = ("entities", "score_threshold", "allow_list", "allow_list_match", "deny_list", "_allow_res")

    def __init__(self, entities, score_threshold, allow_list, allow_list_match, deny_list):
        self.entities = frozenset(entities) if entities is not None else None
        self.score_threshold = score_threshold
        self.allow_list = tuple(allow_list) if allow_list else ()
        self.allow_list_match = allow_list_match
        if isinstance(deny_list, Mapping):
            self.deny_list = {str(k): str(v) for k, v in deny_list.items()}
        elif deny_list:
            self.deny_list = {str(k): _DENY_LIST_DEFAULT_TYPE for k in deny_list}
        else:
            self.deny_list = {}
        if allow_list_match not in _ALLOW_LIST_MATCH:
            raise ConfigurationError(f"allow_list_match must be one of {_ALLOW_LIST_MATCH}")
        for term in list(self.allow_list) + list(self.deny_list):
            if not term or not term.strip():
                raise ConfigurationError("allow_list / deny_list terms must be non-empty")
        self._allow_res = tuple(re.compile(pat) for pat in self.allow_list) if allow_list_match == "regex" else ()

    def threshold_for(self, entity_type: str) -> float:
        t = self.score_threshold
        if t is None:
            return 0.0
        if isinstance(t, Mapping):
            return float(t.get(entity_type, t.get("DEFAULT", 0.0)))
        return float(t)

    def allowed(self, surface: str) -> bool:
        surface = surface.strip()
        if self.allow_list_match == "exact":
            return surface in self.allow_list
        return any(r.fullmatch(surface) for r in self._allow_res)
_DEFAULT_REFINER = SpanRefiner()  # 既定で有効。無効化は PrivacyEngine(refiner=None)


class PrivacyEngine:
    def __init__(
        self,
        detectors: Sequence[Detector],
        policy: Policy | None = None,
        *,
        secret_key: bytes | None = None,
        secret_key_env: str | None = None,
        store: PseudonymStore | None = None,
        crypto: CryptoProvider | None = None,
        resolver: EntityResolver | None = None,
        refiner: SpanRefiner | None = _DEFAULT_REFINER,
        annotators: Sequence[Annotator] = (),
        on_detector_error: str = "raise",
        enhancer: ContextEnhancer | None = None,
        entities: Iterable[str] | None = None,
        score_threshold: float | Mapping[str, float] | None = None,
        allow_list: Iterable[str] | None = None,
        allow_list_match: str = "exact",
        deny_list: Mapping[str, str] | Iterable[str] | None = None,
    ) -> None:
        if not detectors:
            raise ConfigurationError("at least one detector is required")
        if on_detector_error not in _ERROR_MODES:
            raise ConfigurationError(f"on_detector_error must be one of {_ERROR_MODES}")
        if secret_key is not None and secret_key_env is not None:
            raise ConfigurationError("secret_key and secret_key_env are mutually exclusive")
        self.detectors = list(detectors)
        self.policy = policy if policy is not None else Policy()
        self.store = store if store is not None else MemoryStore()
        self.crypto = crypto
        self.resolver = resolver if resolver is not None else EntityResolver()
        self.refiner = refiner
        # Annotator（識別性ベース匿名化）: resolve後・Policy前に型をサブタイプへ書き換える。
        # 既定は空 = 従来どおり全エンティティが Policy に渡る（strict 相当）
        self.annotators = list(annotators)
        for ann in self.annotators:
            if not callable(getattr(ann, "annotate", None)):
                raise ConfigurationError(
                    "annotators must provide an annotate(text, entities) method"
                )
        self.on_detector_error = on_detector_error
        # 文脈語によるスコア補正（閾値の前に適用）。None なら無効
        self.enhancer = enhancer
        # 既定の呼び出しオプション。transform()/detect()/transform_batch() の引数で上書きできる
        self._defaults = _Options(entities, score_threshold, allow_list, allow_list_match, deny_list)
        self._secret: SecretBytes | None
        if secret_key is not None:
            self._secret = SecretBytes(secret_key)
        elif secret_key_env is not None:
            self._secret = SecretBytes.from_env(secret_key_env)
        else:
            self._secret = None
        self._validate_construction()

    def _validate_construction(self) -> None:
        if self.policy.has_strategy("vault"):
            if self.crypto is None:
                raise ConfigurationError("vault strategy requires a CryptoProvider")
            if self._secret is None:
                raise ConfigurationError("vault strategy requires secret_key")
        for op in self.policy.operators():
            # Tokenize / Surrogate 共通: strategy属性とkey属性を持つoperatorが対象
            if getattr(op, "strategy", None) == "hmac":
                if getattr(op, "key", None) is None and self._secret is None:
                    raise ConfigurationError(
                        "hmac strategy requires a key (operator key/key_env or engine secret_key)"
                    )

    # ---- 検出 ----

    def detect(
        self, text: str, *,
        entities: Iterable[str] | None = None,
        score_threshold: float | Mapping[str, float] | None = None,
        allow_list: Iterable[str] | None = None,
        allow_list_match: str | None = None,
        deny_list: Mapping[str, str] | Iterable[str] | None = None,
    ) -> list[Entity]:
        opts = self._options(entities, score_threshold, allow_list, allow_list_match, deny_list)
        raw, _ = self._run_detectors(text)
        final, _ = self._finalize(text, raw, opts)
        return final

    def explain(
        self, text: str, *,
        entities: Iterable[str] | None = None,
        score_threshold: float | Mapping[str, float] | None = None,
        allow_list: Iterable[str] | None = None,
        allow_list_match: str | None = None,
        deny_list: Mapping[str, str] | Iterable[str] | None = None,
    ) -> list[ExplainedEntity]:
        """detect() と同じ最終エンティティに、判定の経緯（検出器・スコア・Refiner・重複解決・Annotator）を付けて返す。
        原文の文字は含めない。"""
        opts = self._options(entities, score_threshold, allow_list, allow_list_match, deny_list)
        raw, _ = self._run_detectors(text)
        final, kept = self._finalize(text, raw, opts)
        return [ExplainedEntity(e, self._explain(text, e, kept, {})) for e in final]

    def _options(self, entities, score_threshold, allow_list, allow_list_match, deny_list) -> _Options:
        d = self._defaults
        if all(x is None for x in (entities, score_threshold, allow_list, allow_list_match, deny_list)):
            return d
        return _Options(
            d.entities if entities is None else entities,
            d.score_threshold if score_threshold is None else score_threshold,
            d.allow_list if allow_list is None else allow_list,
            d.allow_list_match if allow_list_match is None else allow_list_match,
            d.deny_list if deny_list is None else deny_list,
        )

    def _finalize(self, text: str, raw: list[Entity], opts: _Options) -> tuple[list[Entity], list[Entity]]:
        """raw 検出 → スコア閾値・型フィルタ → Refiner → deny list 追加 → allow list 除外 → 重複解決 → Annotator。
        (最終エンティティ, 閾値・型フィルタ後の検出) を返す（後者は explain 用）。"""
        if self.enhancer is not None:
            raw = self.enhancer.enhance(text, raw)
        kept = [e for e in raw if e.score >= opts.threshold_for(e.entity_type)
                and (opts.entities is None or e.entity_type in opts.entities)]
        ents = self.refiner.refine(text, kept) if self.refiner is not None else list(kept)
        for term, etype in opts.deny_list.items():
            if opts.entities is not None and etype not in opts.entities:
                continue
            pos = text.find(term)
            while pos >= 0:
                ents.append(Entity(etype, pos, pos + len(term), 1.0, "deny_list"))
                pos = text.find(term, pos + len(term))
        if opts.allow_list:
            ents = [e for e in ents if not opts.allowed(text[e.start:e.end])]
        resolved = self.resolver.resolve(ents)
        if opts.allow_list:   # deny_list で足した項が Refiner 後に残るように、allow は resolve 後にも適用
            resolved = [e for e in resolved if not opts.allowed(text[e.start:e.end])]
        return self._annotate(text, resolved), kept

    @staticmethod
    def _explain(text: str, final: Entity, raw_kept: list[Entity], resolved_types: dict) -> str:
        """1 エンティティの経緯。原文の文字列は含めない（位置と型・スコア・検出器名のみ）。"""
        overl = [r for r in raw_kept if r.start < final.end and r.end > final.start]
        base_type = final.entity_type[:-len(_PUBLIC_SUFFIX)] if final.entity_type.endswith(_PUBLIC_SUFFIX) else final.entity_type
        same = [r for r in overl if r.entity_type == base_type]
        parts = []
        if final.source == "deny_list":
            parts.append("deny_list")
        elif same:
            best = max(same, key=lambda r: r.score)
            parts.append(f"detector={best.source or '?'}({best.score:.2f})")
            if (best.start, best.end) != (final.start, final.end):
                parts.append(f"refined {best.start}-{best.end}→{final.start}-{final.end}")
        else:
            parts.append(f"detector={final.source or '?'}({final.score:.2f})")
        others = sorted({r.entity_type for r in overl if r.entity_type != base_type})
        if others:
            parts.append("resolved over " + ",".join(others))
        if final.entity_type != base_type:
            parts.append(f"annotator {base_type}→{final.entity_type}")
        return "; ".join(parts)

    def _run_detectors(self, text: str) -> tuple[list[Entity], tuple[str, ...]]:
        all_entities: list[Entity] = []
        failed: list[str] = []
        for det in self.detectors:
            try:
                found = det.detect(text)
                for e in found:
                    self._validate_entity(e, text, det.name)
            except Exception as exc:
                if self.on_detector_error == "raise":
                    if isinstance(exc, DetectorError):
                        raise
                    raise DetectorError(
                        f"detector {det.name!r} failed ({type(exc).__name__})"
                    ) from None
                failed.append(det.name)
                continue
            all_entities.extend(found)
        return all_entities, tuple(failed)

    @staticmethod
    def _validate_entity(e: Entity, text: str, detector_name: str) -> None:
        if not (0 <= e.start < e.end <= len(text)) or not (0.0 <= e.score <= 1.0):
            raise DetectorError(f"detector {detector_name!r} returned an invalid entity")

    # ---- 注釈 ----

    def _annotate(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Annotatorを順に適用する。位置の改変・範囲外・空の型は AnnotatorError（原文は載せない）。"""
        for ann in self.annotators:
            name = getattr(ann, "name", type(ann).__name__)
            try:
                out = list(ann.annotate(text, entities))
            except PrivacyMaskerError:
                raise
            except Exception as exc:
                raise AnnotatorError(
                    f"annotator {name!r} failed ({type(exc).__name__})"
                ) from None
            for e in out:
                if (
                    not isinstance(e, Entity)
                    or not (0 <= e.start < e.end <= len(text))
                    or not isinstance(e.entity_type, str)
                    or not e.entity_type
                ):
                    raise AnnotatorError(f"annotator {name!r} returned an invalid entity")
            entities = out
        return entities

    # ---- 変換 ----

    def transform(
        self, text: str, *, scope_id: str | None = None,
        entities: Iterable[str] | None = None,
        score_threshold: float | Mapping[str, float] | None = None,
        allow_list: Iterable[str] | None = None,
        allow_list_match: str | None = None,
        deny_list: Mapping[str, str] | Iterable[str] | None = None,
        explain: bool = False,
    ) -> TransformResult:
        self._check_call(scope_id)
        if not text.strip():
            self._forget_documents()
            return TransformResult(text=text, entities=(), scope_id=scope_id)
        opts = self._options(entities, score_threshold, allow_list, allow_list_match, deny_list)
        raw, failed = self._run_detectors(text)
        resolved, kept = self._finalize(text, raw, opts)
        ctx = self._make_context(scope_id, text)
        try:
            return self._apply(text, resolved, ctx, scope_id, failed, kept if explain else None)
        finally:
            # _apply() が operator の例外（StoreError・ConfigurationError 等）で
            # 途中終了しても、文書全文のメモ（SurrogateFilter._text_cache）を必ず捨てる
            self._forget_documents()

    def _check_call(self, scope_id: str | None) -> None:
        if scope_id is None and self.policy.has_strategy("vault"):
            raise ConfigurationError("vault strategy requires scope_id")
        if scope_id is not None and self._secret is None:
            raise ConfigurationError("scope_id requires secret_key (for value_hash)")

    def _make_context(self, scope_id: str | None, text: str | None = None):
        if scope_id is None:
            return DocumentContext(self._secret, text)
        assert self._secret is not None
        return StoreContext(scope_id, self.store, self.crypto, self._secret, text)

    def _apply(
        self, text: str, resolved: list[Entity], ctx, scope_id: str | None,
        failed: tuple[str, ...], raw_kept: list[Entity] | None = None,
    ) -> TransformResult:
        out = text
        transformed: list[TransformedEntity] = []
        for e in sorted(resolved, key=lambda x: x.start, reverse=True):
            op = self.policy.operator_for(e.entity_type)
            replacement = op.apply(text[e.start : e.end], e, ctx)
            out = out[: e.start] + replacement + out[e.end :]
            explanation = None
            if raw_kept is not None:
                explanation = self._explain(text, e, raw_kept, {}) + f"; operator={op.name}"
            transformed.append(
                TransformedEntity(
                    entity_type=e.entity_type,
                    original_start=e.start,
                    original_end=e.end,
                    replacement=replacement,
                    score=e.score,
                    operator=op.name,
                    explanation=explanation,
                )
            )
        transformed.reverse()
        return TransformResult(
            text=out, entities=tuple(transformed), scope_id=scope_id, failed_detectors=failed
        )

    def _forget_documents(self) -> None:
        """文書の処理が終わったら、長命な operator（Surrogate）が持つ全文のメモを捨てる。

        transform() / transform_batch() の各呼び出しの最後に 1 回呼ぶ。operator が
        forget_document() を持たない（Surrogate 以外の）場合は何もしない。
        """
        for op in self.policy.operators():
            forget = getattr(op, "forget_document", None)
            if callable(forget):
                forget()

    # ---- バッチ ----

    def transform_batch(
        self, texts: Sequence[str], *, scope_id: str | None = None, batch_size: int = 32,
        entities: Iterable[str] | None = None,
        score_threshold: float | Mapping[str, float] | None = None,
        allow_list: Iterable[str] | None = None,
        allow_list_match: str | None = None,
        deny_list: Mapping[str, str] | Iterable[str] | None = None,
        explain: bool = False,
    ) -> list[TransformResult]:
        self._check_call(scope_id)
        if batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        opts = self._options(entities, score_threshold, allow_list, allow_list_match, deny_list)
        texts = list(texts)
        # 空文字列・空白のみの文書はtransform()と同様に検出対象から除外する
        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        per_text_entities: list[list[Entity]] = [[] for _ in texts]
        per_text_failed: list[list[str]] = [[] for _ in texts]
        for det in self.detectors:
            if hasattr(det, "detect_batch"):
                for k in range(0, len(non_empty_idx), batch_size):
                    idx_chunk = non_empty_idx[k : k + batch_size]
                    chunk = [texts[i] for i in idx_chunk]
                    try:
                        results = det.detect_batch(chunk)
                        if len(results) != len(chunk):
                            raise DetectorError(
                                f"detector {det.name!r} returned {len(results)} "
                                f"results for {len(chunk)} texts"
                            )
                        # 検証を先に全て済ませ、通ってから一括コミットする
                        # （チャンク内の一部だけ検証失敗した場合に他の文書分だけが
                        # 部分的にコミットされてしまう矛盾を防ぐ）
                        for j, found in enumerate(results):
                            for e in found:
                                self._validate_entity(e, chunk[j], det.name)
                        for j, found in enumerate(results):
                            per_text_entities[idx_chunk[j]].extend(found)
                    except Exception as exc:
                        if self.on_detector_error == "raise":
                            if isinstance(exc, DetectorError):
                                raise
                            raise DetectorError(
                                f"detector {det.name!r} failed ({type(exc).__name__})"
                            ) from None
                        for i in idx_chunk:
                            per_text_failed[i].append(det.name)
            else:
                for i in non_empty_idx:
                    text = texts[i]
                    try:
                        found = det.detect(text)
                        for e in found:
                            self._validate_entity(e, text, det.name)
                        per_text_entities[i].extend(found)
                    except Exception as exc:
                        if self.on_detector_error == "raise":
                            if isinstance(exc, DetectorError):
                                raise
                            raise DetectorError(
                                f"detector {det.name!r} failed ({type(exc).__name__})"
                            ) from None
                        per_text_failed[i].append(det.name)

        shared_ctx = None if scope_id is None else self._make_context(scope_id)
        out: list[TransformResult] = []
        try:
            for i, text in enumerate(texts):
                if not text.strip():
                    out.append(TransformResult(text=text, entities=(), scope_id=scope_id))
                    continue
                resolved, kept = self._finalize(text, per_text_entities[i], opts)
                ctx = shared_ctx if shared_ctx is not None else self._make_context(None, text)
                ctx.text = text          # StoreContext を共有するときも文書ごとの全文を持たせる
                out.append(self._apply(text, resolved, ctx, scope_id, tuple(per_text_failed[i]), kept if explain else None))
            return out
        finally:
            # ある文書の operator が例外を送出しても（例: Surrogate が候補を出し尽くして
            # StoreError）、直前の文書で SurrogateFilter._text_cache に残った全文のメモを必ず捨てる
            self._forget_documents()

    # ---- 復元 ----

    def _require_crypto(self) -> None:
        if self.crypto is None:
            raise ConfigurationError("restore requires a CryptoProvider")

    def restore(self, text: str, *, scope_id: str) -> RestoreResult:
        self._require_crypto()
        return self._restore_with(text, scope_id, self._restore_pattern(scope_id))

    def _surrogate_tokens(self, scope_id: str) -> list[str]:
        """Surrogate(strategy="vault") が scope に保存した偽名（2 字以上、値付き）。長い順（同長は辞書順）。

        フォールバックトークン（<TYPE_xxxxxxxx>、SurrogateFilter が全候補を拒否したときの型レベルの
        退避先）も strategy="vault" では original_value 付きで保存され（surrogate.py の _apply_full が
        _draw の戻り値をそのまま save_token に渡すため）、これが元の値へ戻る唯一の正規の経路になる。
        よってトークンの見た目（"<" で始まるか）では絞り込まず、実際に値が保存されているかどうかだけで判定する。
        """
        if not any(isinstance(op, Surrogate) and op.strategy == "vault" for op in self.policy.operators()):
            return []
        iter_tokens = getattr(self.store, "iter_tokens", None)
        if not callable(iter_tokens):
            raise ConfigurationError(
                "Surrogate(strategy='vault') restore requires a PseudonymStore implementing iter_tokens(scope_id) (v0.9)"
            )
        toks = {t for _, t in iter_tokens(scope_id) if len(t) >= 2}
        toks = {t for t in toks if self.store.get_value(scope_id, t) is not None}
        return sorted(toks, key=lambda t: (-len(t), t))

    def restore_batch(self, texts: Sequence[str], *, scope_id: str) -> list[RestoreResult]:
        """複数文書をまとめて復元。同じ scope_id の対応表を使う。"""
        self._require_crypto()
        pattern = self._restore_pattern(scope_id)      # scope の対応表（トークン列挙）は1回だけ読む
        return [self._restore_with(t, scope_id, pattern) for t in texts]

    def _restore_pattern(self, scope_id: str) -> "re.Pattern[str] | None":
        """scope の対応表から復元用の正規表現を1本組み立てる（対応するものが無ければ None）。"""
        templates = sorted(self.policy.tokenize_templates())
        # 偽名（Surrogate vault）は書式を持たないので、scope に保存された値付きトークンを列挙して長い順に置く。
        # Tokenize の書式より**前**に置くこと（Important I2）: 偽名が型トークンで始まる形になった場合でも、
        # alternation は先に書いた枝を優先するため、短い型トークンに一致して偽名が途中で切られることがない
        patterns = [re.escape(t) for t in self._surrogate_tokens(scope_id)]
        patterns += [self._pattern_for(t).pattern for t in templates]
        if not patterns:
            return None
        return re.compile("|".join(f"(?:{p})" for p in patterns))

    def _restore_with(self, text: str, scope_id: str, pattern: "re.Pattern[str] | None") -> RestoreResult:
        """組み立て済みのパターンで1文書を復元する（restore() / restore_batch() 共通の本体）。"""
        if pattern is None:
            return RestoreResult(text=text, unresolved=())
        unresolved: list[str] = []
        restored = 0

        def _sub(m: "re.Match[str]") -> str:
            nonlocal restored
            token = m.group(0)
            blob = self.store.get_value(scope_id, token)
            if blob is None:
                unresolved.append(token)
                return token
            data = self.crypto.decrypt(blob)  # 失敗時はRestoreError（CryptoProviderの責務）
            try:
                fields = decode_fields(data)
            except ValueError as e:
                raise RestoreError("malformed vault entry") from e
            if len(fields) != 3 or fields[0] != scope_id or fields[1] != token:
                raise RestoreError("context mismatch in vault entry")
            restored += 1
            return fields[2]

        # 全パターンを1本のalternationに結合済みのため、元のtextに対して1回だけsub()する。
        # 順次書き換えると、先に復号した平文が後続パターンへ再マッチして別エンティティの
        # 復号値で上書きされ得る
        out = pattern.sub(_sub, text)
        return RestoreResult(text=out, unresolved=tuple(unresolved), restored=restored)

    @staticmethod
    def _pattern_for(template: str) -> "re.Pattern[str]":
        escaped = re.escape(template)
        escaped = escaped.replace(re.escape("{type}"), r"[A-Z][A-Z0-9_]*")
        escaped = escaped.replace(re.escape("{id}"), r"[A-Za-z0-9]+")
        return re.compile(escaped)

"""構造化データ（dict のレコード列・pandas DataFrame）の匿名化（presidio-structured に相当）。

- `transform_records(engine, records, fields=None, column_types="auto")`
  各レコードの文字列フィールドを `engine.transform_batch()` でまとめて変換する。
  `column_types="auto"` のとき、列（フィールド）ごとに標本を検出して「その列の値がほぼ全体として 1 つの型に一致する」場合は
  列全体をその型の値とみなし、検出に頼らず列の全値に Policy の operator を適用する（氏名列・住所列で NER が短い値を落とす事故を防ぐ）。
  presidio-structured の most_common 戦略に相当。`column_types={"name": "PERSON"}` で明示指定もできる。
  型が決まった列は `TransformContext.text=None`（文書全文を渡さない）で operator を呼ぶため、
  `Surrogate` の安全フィルタの `present_in_document`（設計 §7）は効かない。偽名がレコード内の
  他の列の値と偶然衝突する可能性は残る（列単位の処理であり文書横断の既出チェックではないため）。
- `transform_dataframe(engine, df, columns=None, ...)` は DataFrame 版（pandas は任意依存。文字列列だけを対象にする）。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .exceptions import ConfigurationError
from .types import Entity

_AUTO = "auto"


def _infer_column_types(engine, records: Sequence[Mapping[str, Any]], fields: Sequence[str], *,
                        sample: int = 200, min_ratio: float = 0.6) -> dict[str, str]:
    """列ごとに標本を検出し、値全体（空白除く）が単一型で覆われる割合が min_ratio 以上なら列型とする。"""
    out: dict[str, str] = {}
    for f in fields:
        values = [r[f] for r in records[:sample] if isinstance(r.get(f), str) and r[f].strip()]
        if len(values) < 3:
            continue
        votes: Counter = Counter()
        for v, ents in zip(values, [engine.detect(v) for v in values]):
            stripped = v.strip()
            lo, hi = v.index(stripped), v.index(stripped) + len(stripped)
            full = [e for e in ents if e.start <= lo and e.end >= hi]
            if full:
                votes[full[0].entity_type] += 1
        if votes:
            etype, n = votes.most_common(1)[0]
            if n / len(values) >= min_ratio:
                out[f] = etype
    return out


def transform_records(
    engine, records: Sequence[Mapping[str, Any]], *, fields: Sequence[str] | None = None,
    column_types: Mapping[str, str] | str | None = _AUTO, scope_id: str | None = None, batch_size: int = 32, **options,
) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return []
    if fields is None:
        fields = sorted({k for r in records for k, v in r.items() if isinstance(v, str)})
    fields = list(fields)
    if isinstance(column_types, str):
        if column_types != _AUTO:
            raise ConfigurationError("column_types must be a mapping, 'auto', or None")
        col_types = _infer_column_types(engine, records, fields)
    else:
        col_types = dict(column_types or {})
    # 列型が決まった列: 値全体をその型の 1 エンティティとして Policy に通す（検出はしない）
    # それ以外の列: 文字列を集めて transform_batch
    out = [dict(r) for r in records]
    plain_jobs: list[tuple[int, str]] = []
    typed_jobs: list[tuple[int, str, str]] = []
    for i, r in enumerate(records):
        for f in fields:
            v = r.get(f)
            if not isinstance(v, str) or not v.strip():
                continue
            if f in col_types:
                typed_jobs.append((i, f, col_types[f]))
            else:
                plain_jobs.append((i, f))
    if plain_jobs:
        results = engine.transform_batch([records[i][f] for i, f in plain_jobs], scope_id=scope_id, batch_size=batch_size, **options)
        for (i, f), res in zip(plain_jobs, results):
            out[i][f] = res.text
    if typed_jobs:
        ctx = engine._make_context(scope_id, None) if scope_id is not None else None
        for i, f, etype in typed_jobs:
            v = records[i][f]
            stripped = v.strip(); lo = v.index(stripped); hi = lo + len(stripped)
            ent = Entity(etype, lo, hi, 1.0, "column")
            c = ctx if ctx is not None else engine._make_context(None, None)
            op = engine.policy.operator_for(etype)
            out[i][f] = v[:lo] + op.apply(stripped, ent, c) + v[hi:]
    return out


def transform_dataframe(engine, df, *, columns: Sequence[str] | None = None, **kwargs):
    """pandas DataFrame の文字列列を匿名化して新しい DataFrame を返す（元は変更しない）。"""
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("transform_dataframe requires pandas") from e
    if columns is None:
        # pandas 3 は文字列列が object ではなく str dtype になるので両方見る
        from pandas.api.types import is_object_dtype, is_string_dtype
        columns = [c for c in df.columns if is_string_dtype(df[c]) or is_object_dtype(df[c])]
    records = df[list(columns)].where(df[list(columns)].notna(), None).to_dict("records")
    fixed = transform_records(engine, records, fields=list(columns), **kwargs)
    new = df.copy()
    for c in columns:
        new[c] = [r.get(c) for r in fixed]
    return new

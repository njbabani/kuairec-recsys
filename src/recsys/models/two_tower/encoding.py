"""Turn feature tables into dense integer and float arrays a neural network can index.

Every entity (user or video) gets a row number starting at 1; row 0 is the *unknown* entity
used for ids never seen before. Within each categorical column, value 0 likewise means
"unknown or missing", so the network always has an embedding to fall back on.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self

import numpy as np
import polars as pl

UNKNOWN = 0


@dataclass(frozen=True)
class Vocabulary:
    """Distinct non-null values in sorted order; value ``values[i]`` encodes as ``i + 1``."""

    values: tuple[Any, ...]

    @classmethod
    def fit(cls, values: pl.Series) -> Self:
        return cls(tuple(values.drop_nulls().unique().sort().to_list()))

    def __len__(self) -> int:
        return len(self.values) + 1  # plus the unknown slot

    @cached_property
    def index(self) -> dict[Any, int]:
        return {value: position for position, value in enumerate(self.values, start=1)}

    def encode(self, values: pl.Series) -> np.ndarray:
        return (
            values.replace_strict(self.index, default=UNKNOWN, return_dtype=pl.Int64)
            .fill_null(UNKNOWN)
            .to_numpy()
        )


@dataclass(frozen=True)
class EntityFeatures:
    """Per-entity feature arrays, all with ``size`` rows (row 0 = the unknown entity)."""

    ids: Vocabulary
    categorical: np.ndarray  # int64 (size, n_categorical); 0 = unknown value
    cardinalities: tuple[int, ...]  # vocabulary size of each categorical column
    numeric: np.ndarray  # float32 (size, n_numeric), standardised; missing = 0 (the mean)
    multi_valued: np.ndarray  # int64 (size, longest list), padded with 0
    multi_valued_cardinality: int  # 0 when there is no multi-valued column

    @property
    def size(self) -> int:
        return len(self.ids)


def encode_entities(
    table: pl.DataFrame,
    key: str,
    extra_ids: pl.Series,
    *,
    categorical: Sequence[str] = (),
    numeric: Sequence[str] = (),
    multi_valued: str | None = None,
) -> EntityFeatures:
    """Encode ``table`` (one row per ``key``) plus any ``extra_ids`` that lack a row in it."""
    if table[key].is_duplicated().any():
        raise ValueError(f"{key} must be unique in the feature table")
    key_dtype = table[key].dtype
    ids = Vocabulary.fit(pl.concat([table[key], extra_ids.cast(key_dtype)]))
    # One row per encoded entity, in index order; the leading null key is the unknown entity.
    rows = pl.DataFrame({key: [None, *ids.values]}, schema={key: key_dtype}).join(
        table, on=key, how="left", maintain_order="left"
    )

    vocabularies = [Vocabulary.fit(table[column]) for column in categorical]
    codes = [
        vocab.encode(rows[column]) for vocab, column in zip(vocabularies, categorical, strict=True)
    ]
    multi, multi_cardinality = _encode_lists(rows, table, multi_valued)
    return EntityFeatures(
        ids=ids,
        categorical=np.column_stack(codes) if codes else np.zeros((rows.height, 0), np.int64),
        cardinalities=tuple(len(vocab) for vocab in vocabularies),
        numeric=_standardise(rows, table, numeric),
        multi_valued=multi,
        multi_valued_cardinality=multi_cardinality,
    )


def _standardise(rows: pl.DataFrame, table: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    standardised = np.zeros((rows.height, len(columns)), dtype=np.float32)
    for position, column in enumerate(columns):
        known = _finite_or_null(table[column])  # NaN or +-inf would poison the whole column
        mean, std = known.mean(), known.std(ddof=0)
        scale = std or 1.0  # a constant column is only centred
        values = (_finite_or_null(rows[column]) - (mean or 0.0)) / scale
        standardised[:, position] = values.fill_null(0.0)
    return standardised


def _finite_or_null(values: pl.Series) -> pl.Series:
    values = values.cast(pl.Float64)
    return values.set(~values.is_finite().fill_null(False), None)


def _encode_lists(
    rows: pl.DataFrame, table: pl.DataFrame, column: str | None
) -> tuple[np.ndarray, int]:
    if column is None:
        return np.zeros((rows.height, 0), dtype=np.int64), 0
    index = Vocabulary.fit(table[column].explode(empty_as_null=True)).index
    lists = [
        [index.get(value, UNKNOWN) for value in values or []] for values in rows[column].to_list()
    ]
    padded = np.zeros((rows.height, max(map(len, lists))), dtype=np.int64)
    for row, codes in enumerate(lists):
        padded[row, : len(codes)] = codes
    return padded, len(index) + 1

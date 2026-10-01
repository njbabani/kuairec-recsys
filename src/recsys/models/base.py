"""The interface every recommender implements, so experiments can treat them uniformly."""

from typing import Protocol, Self

import polars as pl


class Recommender(Protocol):
    name: str

    def fit(self, train: pl.DataFrame) -> Self:
        """Learn from labeled training interactions (``user_id``, ``video_id``, ``is_positive``)."""
        ...

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        """One score per row of ``pairs`` (``user_id``, ``video_id``), in row order; higher ranks
        first."""
        ...

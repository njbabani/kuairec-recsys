"""The interface every recommender implements, so experiments can treat them uniformly."""

from typing import Protocol, Self

import polars as pl


class Scorer(Protocol):
    """Anything that scores (user, video) pairs: a fitted model, or a pipeline of them."""

    name: str

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        """One score per row of ``pairs`` (``user_id``, ``video_id``), in row order; higher ranks
        first."""
        ...


class Recommender(Scorer, Protocol):
    """A scorer that learns from training interactions."""

    def fit(self, train: pl.DataFrame) -> Self:
        """Learn from labeled training interactions (``user_id``, ``video_id``, ``is_positive``)."""
        ...

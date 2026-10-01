"""Non-personalized baselines: random order and two kinds of popularity.

The EDA found that *how often* the old recommender showed a video is a poor signal of appeal,
while *how well it was received when shown* transfers well. The two popularity variants test
exactly that.
"""

from typing import Self

import numpy as np
import polars as pl

POPULARITY_SIGNALS = ("views", "engagement")


class RandomRecommender:
    """A floor every real model must beat: an independent random score per pair."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, train: pl.DataFrame) -> Self:
        return self

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        return pl.Series("score", np.random.default_rng(self.seed).random(pairs.height))


class PopularityRecommender:
    """Scores every user's candidates identically, by a per-video statistic from training.

    * ``by="views"``: how often the video appeared in the logged feed (its exposure).
    * ``by="engagement"``: its positive rate, shrunk towards the global rate by adding
      ``prior_strength`` pseudo-views (empirical Bayes), so 3 positives out of 3 views does not
      outrank 900 out of 1,000.
    """

    def __init__(self, by: str, prior_strength: float = 0.0) -> None:
        if by not in POPULARITY_SIGNALS:
            raise ValueError(f"by must be one of {POPULARITY_SIGNALS}, got {by!r}")
        if prior_strength < 0:
            raise ValueError(f"prior_strength must be non-negative, got {prior_strength}")
        self.by = by
        self.prior_strength = prior_strength
        self.name = f"popularity_{by}"
        self._video_scores: pl.DataFrame | None = None
        self._unseen_score = 0.0

    def fit(self, train: pl.DataFrame) -> Self:
        stats = train.group_by("video_id").agg(
            views=pl.len(), positives=pl.col("is_positive").sum()
        )
        if self.by == "views":
            self._video_scores = stats.select("video_id", score=pl.col("views").cast(pl.Float64))
            self._unseen_score = 0.0
        else:
            global_rate = float(train["is_positive"].mean())
            pseudo_positives = self.prior_strength * global_rate
            self._video_scores = stats.select(
                "video_id",
                score=(pl.col("positives") + pseudo_positives)
                / (pl.col("views") + self.prior_strength),
            )
            self._unseen_score = global_rate
        return self

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        if self._video_scores is None:
            raise RuntimeError("Call fit() before score()")
        return (
            pairs.select("video_id")
            .join(self._video_scores, on="video_id", how="left", maintain_order="left")["score"]
            .fill_null(self._unseen_score)
        )

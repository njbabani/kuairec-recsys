"""A transparent personalised baseline: popular videos, re-weighted by the user's category taste.

score(user, video) = base score of the video x the user's average *lift* over its categories,
where lift = how often this user liked videos of the category / how often everyone did.
A lift of 1.5 for comedy means "50% more likely than average to like comedy". Rates are shrunk
towards the category's rate with ``prior_strength`` pseudo-views, so two lucky views do not
make a user a fan. Users with no history get a lift of 1 (the plain base ranking), and videos
unseen in training still get their category lift, so this model handles cold-start videos.
"""

from typing import Self

import polars as pl

from recsys.models.base import Recommender

NEUTRAL_LIFT = 1.0


class CategoryAffinityRecommender:
    name = "category_affinity"

    def __init__(
        self, video_categories: pl.DataFrame, prior_strength: float, base: Recommender
    ) -> None:
        # With no pseudo-views, a category nobody liked has rate 0 and its lift would be 0 / 0.
        if prior_strength <= 0:
            raise ValueError(f"prior_strength must be positive, got {prior_strength}")
        self.video_categories = video_categories.select("video_id", "category_ids")
        self.prior_strength = prior_strength
        self.base = base
        self._lifts: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame) -> Self:
        if not train["is_positive"].any():
            raise ValueError("Training data has no positive views to learn tastes from")
        self.base.fit(train)

        global_rate = float(train["is_positive"].mean())
        by_category = (
            train.join(self.video_categories, on="video_id", how="inner")
            .explode("category_ids", empty_as_null=False)
            .drop_nulls("category_ids")
        )
        category_rates = self._shrunk_rates(by_category, ["category_ids"], toward=global_rate)
        user_rates = (
            by_category.group_by("user_id", "category_ids")
            .agg(views=pl.len(), positives=pl.col("is_positive").sum())
            .join(category_rates, on="category_ids")
        )
        self._lifts = user_rates.select(
            "user_id",
            "category_ids",
            lift=(pl.col("positives") + self.prior_strength * pl.col("rate"))
            / (pl.col("views") + self.prior_strength)
            / pl.col("rate"),
        )
        return self

    def _shrunk_rates(self, views: pl.DataFrame, keys: list[str], toward: float) -> pl.DataFrame:
        return (
            views.group_by(keys)
            .agg(views=pl.len(), positives=pl.col("is_positive").sum())
            .select(
                *keys,
                rate=(pl.col("positives") + self.prior_strength * toward)
                / (pl.col("views") + self.prior_strength),
            )
        )

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        if self._lifts is None:
            raise RuntimeError("Call fit() before score()")
        mean_lift = (
            pairs.select("user_id", "video_id")
            .with_row_index("row")
            .join(self.video_categories, on="video_id", how="left")
            # Unknown or empty categories become one null row, which gets the neutral lift.
            .explode("category_ids", empty_as_null=True)
            .join(self._lifts, on=["user_id", "category_ids"], how="left")
            .group_by("row")
            .agg(lift=pl.col("lift").fill_null(NEUTRAL_LIFT).mean())
            .sort("row")["lift"]
        )
        return (self.base.score(pairs) * mean_lift).alias("score")

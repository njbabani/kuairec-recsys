"""Features for the re-ranker: one row per (user, video) pair, from data before the cutoff only.

Every feature comes from the training period (logged views before ``cutoff``), platform-wide
daily statistics from before the cutoff, or catalogue and profile data. The re-ranker learns
from the validation week's labels and is evaluated on the fully observed users, so nothing here
looks at either: :meth:`RankingFeatures.transform` only ever sees user and video ids.

* stage-one scores: the retrieval model and the earlier recommenders, used as features
* video: views and length in training, platform engagement rates, age at the cutoff
* user: activity and positive rate in training, profile activity level
* user x video: the user's positive rate for videos of this length, and the share of the
  user's views that went to this video's category
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Self

import polars as pl

from recsys.models.base import Scorer

FEATURE_COLUMNS = (
    "two_tower_score",
    "als_score",
    "engagement_rate",
    "category_lift",
    "log_video_views",
    "log_duration_s",
    "first_category",
    "log_platform_plays",
    "platform_like_rate",
    "platform_complete_rate",
    "platform_share_rate",
    "platform_comment_rate",
    "platform_follow_rate",
    "video_age_days",
    "log_user_views",
    "user_positive_rate",
    "user_active_degree",
    "user_length_rate",
    "user_category_share",
)
CATEGORICAL_COLUMNS = ("first_category", "user_active_degree")
# Platform-wide counts per play, from the daily statistics.
PLATFORM_RATES = {
    "platform_like_rate": "like_cnt",
    "platform_complete_rate": "complete_play_cnt",
    "platform_share_rate": "share_cnt",
    "platform_comment_rate": "comment_cnt",
    "platform_follow_rate": "follow_cnt",
}


@dataclass(frozen=True)
class StageOneScorers:
    """Fitted models whose scores become features (each fitted on training data only)."""

    two_tower: Scorer
    als: Scorer
    engagement: Scorer
    category_affinity: Scorer


@dataclass(frozen=True)
class FeatureSources:
    train: pl.DataFrame  # user_id, video_id, is_positive, duration_bucket (before the cutoff)
    user_features: pl.DataFrame
    video_features: pl.DataFrame
    daily_stats: pl.DataFrame
    cutoff: date


def platform_stats(daily_stats: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Platform-wide engagement per play and video age, from the days before ``cutoff``."""
    totals = (
        daily_stats.filter(pl.col("date") < cutoff)
        .group_by("video_id")
        .agg(
            pl.col("play_cnt").sum(),
            *(pl.col(count).sum() for count in PLATFORM_RATES.values()),
            uploaded=pl.col("upload_dt").str.to_date("%Y-%m-%d", strict=False).min(),
        )
    )
    plays = pl.col("play_cnt")
    return totals.select(
        "video_id",
        log_platform_plays=plays.cast(pl.Float64).log1p(),
        **{
            name: pl.when(plays > 0).then(pl.col(count) / plays)
            for name, count in PLATFORM_RATES.items()
        },
        video_age_days=(pl.lit(cutoff) - pl.col("uploaded")).dt.total_days(),
    )


def video_train_stats(train: pl.DataFrame) -> pl.DataFrame:
    return train.group_by("video_id").agg(
        log_video_views=pl.len().cast(pl.Float64).log1p(),
        duration_bucket=pl.col("duration_bucket").mode().min(),
    )


def user_train_stats(train: pl.DataFrame) -> pl.DataFrame:
    return train.group_by("user_id").agg(
        log_user_views=pl.len().cast(pl.Float64).log1p(),
        user_positive_rate=pl.col("is_positive").mean(),
    )


def user_length_rates(train: pl.DataFrame, prior_strength: float) -> pl.DataFrame:
    """Each user's positive rate per length bucket, shrunk towards their overall rate."""
    overall = train.group_by("user_id").agg(user_rate=pl.col("is_positive").mean())
    return (
        train.group_by("user_id", "duration_bucket")
        .agg(views=pl.len(), positives=pl.col("is_positive").sum())
        .join(overall, on="user_id")
        .select(
            "user_id",
            "duration_bucket",
            user_length_rate=(pl.col("positives") + prior_strength * pl.col("user_rate"))
            / (pl.col("views") + prior_strength),
        )
    )


def first_categories(video_features: pl.DataFrame) -> pl.DataFrame:
    return video_features.select("video_id", first_category=pl.col("category_ids").list.first())


def user_category_shares(train: pl.DataFrame, video_features: pl.DataFrame) -> pl.DataFrame:
    """Share of each user's training views that went to each (first-level) category."""
    viewed = train.join(first_categories(video_features), on="video_id", how="inner")
    return (
        viewed.group_by("user_id", "first_category")
        .agg(views=pl.len())
        .select(
            "user_id",
            "first_category",
            user_category_share=pl.col("views") / pl.col("views").sum().over("user_id"),
        )
    )


def _codes(values: pl.Series) -> dict:
    """Small non-negative integer codes for a categorical column (what LightGBM expects)."""
    return {value: code for code, value in enumerate(values.drop_nulls().unique().sort())}


@dataclass(frozen=True)
class _Tables:
    videos: pl.DataFrame
    users: pl.DataFrame
    length_rates: pl.DataFrame
    category_shares: pl.DataFrame
    codes: dict[str, dict[Any, int]]


class RankingFeatures:
    def __init__(self, scorers: StageOneScorers, prior_strength: float) -> None:
        self.scorers = scorers
        self.prior_strength = prior_strength
        self._tables: _Tables | None = None

    def fit(self, sources: FeatureSources) -> Self:
        """Precompute every per-video and per-user statistic from the training period."""
        for name, table, key in (
            ("user", sources.user_features, "user_id"),
            ("video", sources.video_features, "video_id"),
        ):
            if table[key].is_duplicated().any():
                raise ValueError(f"{key} must be unique in the {name} feature table")
        train = sources.train
        videos = (
            sources.video_features.select("video_id", "log_duration_s")
            .join(first_categories(sources.video_features), on="video_id", how="left")
            .join(video_train_stats(train), on="video_id", how="left")
            .join(platform_stats(sources.daily_stats, sources.cutoff), on="video_id", how="left")
        )
        users = sources.user_features.select("user_id", "user_active_degree").join(
            user_train_stats(train), on="user_id", how="full", coalesce=True
        )
        self._tables = _Tables(
            videos=videos,
            users=users,
            length_rates=user_length_rates(train, self.prior_strength),
            category_shares=user_category_shares(train, sources.video_features),
            codes={
                "first_category": _codes(videos["first_category"]),
                "user_active_degree": _codes(users["user_active_degree"]),
            },
        )
        return self

    def transform(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """``user_id``, ``video_id`` and every feature column, one row per pair, in order."""
        if self._tables is None:
            raise RuntimeError("Call fit() before transform()")
        tables = self._tables
        frame = self._stage_one_scores(pairs.select("user_id", "video_id"))
        frame = (
            frame.join(tables.videos, on="video_id", how="left", maintain_order="left")
            .join(tables.users, on="user_id", how="left", maintain_order="left")
            .join(
                tables.length_rates,
                on=["user_id", "duration_bucket"],
                how="left",
                maintain_order="left",
            )
            .join(
                tables.category_shares,
                on=["user_id", "first_category"],
                how="left",
                maintain_order="left",
            )
        )
        if frame.height != pairs.height:
            raise RuntimeError("A feature join changed the number of rows; check table keys")
        known_user = pl.col("log_user_views").is_not_null()
        return frame.with_columns(
            user_length_rate=pl.col("user_length_rate").fill_null(pl.col("user_positive_rate")),
            user_category_share=pl.when(known_user).then(
                pl.col("user_category_share").fill_null(0.0)
            ),
            **{
                column: pl.col(column).replace_strict(mapping, default=None, return_dtype=pl.Int32)
                for column, mapping in tables.codes.items()
            },
        ).select("user_id", "video_id", *FEATURE_COLUMNS)

    def categorical_codes(self) -> dict[str, dict[Any, int]]:
        """The integer code of every categorical value, needed to use a saved ranker."""
        if self._tables is None:
            raise RuntimeError("Call fit() before categorical_codes()")
        return {column: dict(codes) for column, codes in self._tables.codes.items()}

    def _stage_one_scores(self, pairs: pl.DataFrame) -> pl.DataFrame:
        engagement = self.scorers.engagement.score(pairs)
        affinity = self.scorers.category_affinity.score(pairs)
        als = self.scorers.als.score(pairs)
        return pairs.with_columns(
            two_tower_score=self.scorers.two_tower.score(pairs),
            # ALS gives -inf to videos it never saw; to the ranker that is "missing".
            als_score=pl.when(als.is_finite()).then(als),
            engagement_rate=engagement,
            category_lift=pl.when(engagement > 0).then(affinity / engagement),
        )

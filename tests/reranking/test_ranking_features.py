import math
from datetime import date

import polars as pl
import pytest

from recsys.reranking.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FeatureSources,
    RankingFeatures,
    StageOneScorers,
    platform_stats,
    user_category_shares,
    user_length_rates,
)

CUTOFF = date(2020, 8, 30)


class FixedScorer:
    """Scores a pair as user_id * 10 + video_id; remembers which columns it was shown."""

    name = "fixed"

    def __init__(self, offset: float = 0.0, unknown_video: int | None = None) -> None:
        self.offset = offset
        self.unknown_video = unknown_video
        self.seen_columns: list[str] = []

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        self.seen_columns = pairs.columns
        values = pairs["user_id"] * 10 + pairs["video_id"] + self.offset
        if self.unknown_video is not None:
            values = pl.Series(
                [
                    -math.inf if video == self.unknown_video else value
                    for video, value in zip(pairs["video_id"], values, strict=True)
                ]
            )
        return values.cast(pl.Float64).alias("score")


def train() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "video_id": [10, 10, 11, 12, 10, 11],
            "is_positive": [True, False, True, False, False, False],
            "duration_bucket": [0, 0, 1, 1, 0, 1],
        }
    )


def sources() -> FeatureSources:
    return FeatureSources(
        train=train(),
        user_features=pl.DataFrame(
            {"user_id": [1, 2], "user_active_degree": ["high_active", "low_active"]}
        ),
        video_features=pl.DataFrame(
            {
                "video_id": [10, 11, 12, 13],
                "category_ids": [[5], [7, 5], [5], [-124]],
                "log_duration_s": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        daily_stats=pl.DataFrame(
            {
                "video_id": [10, 10, 11],
                "date": [date(2020, 8, 1), date(2020, 8, 30), date(2020, 8, 2)],
                "upload_dt": ["2020-07-31", "2020-07-31", "2020-08-20"],
                "play_cnt": [100, 900, 50],
                "like_cnt": [10, 900, 5],
                "complete_play_cnt": [40, 0, 10],
                "share_cnt": [1, 0, 0],
                "comment_cnt": [2, 0, 1],
                "follow_cnt": [3, 0, 0],
            }
        ),
        cutoff=CUTOFF,
    )


def scorers(als: FixedScorer | None = None) -> StageOneScorers:
    return StageOneScorers(
        two_tower=FixedScorer(),
        als=als or FixedScorer(offset=1.0),
        engagement=FixedScorer(offset=2.0),
        category_affinity=FixedScorer(offset=4.0),
    )


def test_platform_stats_only_count_days_before_the_cutoff():
    stats = platform_stats(sources().daily_stats, CUTOFF).sort("video_id")

    video_10 = stats.row(0, named=True)
    assert video_10["platform_like_rate"] == pytest.approx(0.10)  # the Aug 30 day is excluded
    assert video_10["platform_complete_rate"] == pytest.approx(0.40)
    assert video_10["log_platform_plays"] == pytest.approx(math.log1p(100))
    assert video_10["video_age_days"] == 30


def test_user_length_rates_are_shrunk_towards_the_users_overall_rate():
    rates = user_length_rates(train(), prior_strength=2.0)

    user_1_bucket_0 = rates.filter((pl.col("user_id") == 1) & (pl.col("duration_bucket") == 0))
    # user 1: 2 positives in 4 views overall (0.5); bucket 0 has 1 positive in 2 views
    assert user_1_bucket_0["user_length_rate"].item() == pytest.approx((1 + 2 * 0.5) / (2 + 2))


def test_user_category_shares_describe_where_each_users_views_went():
    shares = user_category_shares(train(), sources().video_features)

    per_user = shares.group_by("user_id").agg(pl.col("user_category_share").sum())
    assert per_user["user_category_share"].to_list() == pytest.approx([1.0, 1.0])


def test_features_cover_every_column_in_the_order_of_the_pairs():
    pairs = pl.DataFrame({"user_id": [2, 1, 1], "video_id": [11, 12, 10]})

    features = RankingFeatures(scorers(), prior_strength=2.0).fit(sources()).transform(pairs)

    assert features.columns == ["user_id", "video_id", *FEATURE_COLUMNS]
    assert features.select("user_id", "video_id").equals(pairs)
    assert features["two_tower_score"].to_list() == [31.0, 22.0, 20.0]
    lift = features["category_lift"] * features["engagement_rate"]
    assert lift.to_list() == pytest.approx([35.0, 26.0, 24.0])  # category_affinity score


def test_features_never_see_the_label_and_tolerate_unknown_users_and_videos():
    spy = scorers()
    pairs = pl.DataFrame({"user_id": [1, 99], "video_id": [99, 10], "is_positive": [True, False]})

    features = RankingFeatures(spy, prior_strength=2.0).fit(sources()).transform(pairs)

    assert spy.two_tower.seen_columns == ["user_id", "video_id"]
    assert features.height == 2
    assert features["log_video_views"][0] is None  # video 99 never seen in training
    assert features["log_user_views"][1] is None  # user 99 never seen in training


def test_missing_als_scores_become_nulls_for_the_ranker():
    pairs = pl.DataFrame({"user_id": [1, 1], "video_id": [10, 13]})
    features = (
        RankingFeatures(scorers(als=FixedScorer(unknown_video=13)), prior_strength=2.0)
        .fit(sources())
        .transform(pairs)
    )

    assert features["als_score"][1] is None


def test_categorical_features_are_small_non_negative_codes():
    pairs = pl.DataFrame({"user_id": [1, 2, 1], "video_id": [10, 11, 13]})

    features = RankingFeatures(scorers(), prior_strength=2.0).fit(sources()).transform(pairs)

    for column in CATEGORICAL_COLUMNS:
        codes = features[column].drop_nulls()
        assert codes.min() >= 0, column
        assert codes.dtype == pl.Int32, column


def test_transform_before_fit_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        RankingFeatures(scorers(), prior_strength=2.0).transform(
            pl.DataFrame({"user_id": [1], "video_id": [10]})
        )


@pytest.mark.parametrize("table", ["user_features", "video_features"])
def test_duplicate_ids_in_a_feature_table_are_rejected(table):
    clean = sources()
    duplicated = getattr(clean, table)
    broken = FeatureSources(**{**clean.__dict__, table: pl.concat([duplicated, duplicated])})

    with pytest.raises(ValueError, match="unique"):
        RankingFeatures(scorers(), prior_strength=2.0).fit(broken)


def test_categorical_codes_are_available_to_describe_a_saved_model():
    fitted = RankingFeatures(scorers(), prior_strength=2.0).fit(sources())

    codes = fitted.categorical_codes()

    assert codes["first_category"] == {-124: 0, 5: 1, 7: 2}
    assert set(codes["user_active_degree"]) == {"high_active", "low_active"}

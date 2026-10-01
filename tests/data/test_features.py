import math

import pandera.polars as pa
import polars as pl
import pytest

from recsys.data.features import (
    USER_CATEGORICAL_COLUMNS,
    USER_COUNT_COLUMNS,
    VIDEO_FEATURES_SCHEMA,
    build_user_features,
    build_video_features,
    run_features,
)


def make_users(**overrides: list) -> pl.DataFrame:
    """A users table with every feature column the stage reads (two users by default)."""
    columns: dict[str, list] = {"user_id": [0, 1]}
    columns |= {name: ["a", "b"] for name in USER_CATEGORICAL_COLUMNS}
    columns |= {name: [0, 99] for name in USER_COUNT_COLUMNS}
    return pl.DataFrame(columns | overrides)


def write_processed(directory, users: pl.DataFrame, make_interactions) -> None:
    directory.mkdir()
    users.write_parquet(directory / "users.parquet")
    pl.DataFrame({"video_id": [10, 11], "category_ids": [[8], [9]]}).write_parquet(
        directory / "video_categories.parquet"
    )
    make_interactions(video_id=[10]).write_parquet(directory / "big_matrix.parquet")
    make_interactions(video_id=[11]).write_parquet(directory / "small_matrix.parquet")


def test_user_features_keep_categories_as_text_and_log_transform_counts():
    users = make_users(is_live_streamer=[0, 1], follow_user_num=[0, 99])

    features = build_user_features(users)

    assert features["is_live_streamer"].to_list() == ["0", "1"]
    assert features["log_follow_user_num"].to_list() == pytest.approx([0.0, math.log(100)])
    assert "follow_user_num" not in features.columns


def test_user_features_keep_missing_values_missing():
    users = make_users(user_active_degree=["high_active", None], register_days=[None, 5])

    features = build_user_features(users)

    assert features["user_active_degree"].to_list() == ["high_active", None]
    assert features["log_register_days"][0] is None


def test_video_features_use_the_median_length_seen_across_every_table(make_interactions):
    categories = pl.DataFrame({"video_id": [12, 10, 11], "category_ids": [[3], [8], [27, 9]]})
    big = make_interactions(video_id=[10, 10, 10], video_duration=[9000, 10000, 60000])
    small = make_interactions(video_id=[11], video_duration=[6000])

    features = build_video_features(categories, [big, small])

    assert features["video_id"].to_list() == [10, 11, 12]
    assert features["category_ids"].to_list() == [[8], [27, 9], [3]]
    lengths = features["log_duration_s"].to_list()
    assert lengths[:2] == pytest.approx([math.log(10.0), math.log(6.0)])
    assert lengths[2] is None  # never watched anywhere: length unknown


@pytest.mark.integration
def test_run_features_writes_validated_tables(tmp_path, make_interactions):
    write_processed(tmp_path / "processed", make_users(), make_interactions)

    run_features(tmp_path / "processed", tmp_path / "features")

    users = pl.read_parquet(tmp_path / "features" / "users.parquet")
    videos = pl.read_parquet(tmp_path / "features" / "videos.parquet")
    assert users["user_id"].to_list() == [0, 1]
    assert videos["video_id"].to_list() == [10, 11]


def test_run_features_rejects_duplicate_users_and_writes_nothing(tmp_path, make_interactions):
    write_processed(tmp_path / "processed", make_users(user_id=[0, 0]), make_interactions)

    with pytest.raises(pa.errors.SchemaErrors):
        run_features(tmp_path / "processed", tmp_path / "features")

    assert not (tmp_path / "features").exists()


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_feature_contract_rejects_non_finite_numbers(bad_value):
    videos = pl.DataFrame(
        {"video_id": [10, 11], "category_ids": [[8], [9]], "log_duration_s": [1.0, bad_value]}
    )

    with pytest.raises(pa.errors.SchemaErrors):
        VIDEO_FEATURES_SCHEMA.validate(videos, lazy=True)

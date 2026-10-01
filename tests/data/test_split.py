import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandera.polars as pa
import polars as pl
import pytest

from recsys.config import LabelParams, SplitParams
from recsys.data.labels import DurationThresholds, label_interactions
from recsys.data.split import (
    build_splits,
    local_midnight,
    run_split,
    split_users,
    summarize_splits,
    temporal_split,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
VALID_START = date(2020, 8, 30)
CUTOFF = datetime(2020, 8, 30, tzinfo=SHANGHAI)
ONE_BUCKET = DurationThresholds((), (0.4,), quantile=0.8)
LABEL = LabelParams(bucket_width_ratio=1.25, min_views_per_bucket=1, positive_quantile=0.8)
SPLIT = SplitParams(valid_start=VALID_START, tune_user_share=0.5, assignment_salt="test-salt")
MANY_USERS = list(range(20))


def at(day: int, hour: int = 12) -> datetime:
    return datetime(2020, 8, day, hour, tzinfo=SHANGHAI)


# --- temporal split ----------------------------------------------------------------------


def test_local_midnight_uses_the_platform_timezone():
    assert local_midnight(VALID_START) == CUTOFF


def test_temporal_split_sends_events_from_the_cutoff_onwards_to_validation(make_interactions):
    df = make_interactions(event_time=[at(29, 23), at(30, 0), at(31)])

    train, valid = temporal_split(df, CUTOFF)

    assert train["event_time"].to_list() == [at(29, 23)]
    assert valid["event_time"].to_list() == [at(30, 0), at(31)]
    assert train.height + valid.height == df.height


@pytest.mark.parametrize(
    "cutoff",
    [datetime(2020, 8, 1, tzinfo=SHANGHAI), datetime(2020, 9, 1, tzinfo=SHANGHAI)],
    ids=["before-all-events", "after-all-events"],
)
def test_temporal_split_rejects_a_cutoff_that_leaves_a_side_empty(make_interactions, cutoff):
    df = make_interactions(event_time=[at(29), at(31)])

    with pytest.raises(ValueError, match="empty"):
        temporal_split(df, cutoff)


def test_temporal_split_refuses_rows_without_a_time_instead_of_dropping_them(make_interactions):
    df = make_interactions(event_time=[at(29), None, at(31)])

    with pytest.raises(ValueError, match="event_time"):
        temporal_split(df, CUTOFF)


# --- user split of the fully observed matrix ---------------------------------------------


def test_split_users_puts_every_user_on_exactly_one_side(make_interactions):
    df = make_interactions(user_id=MANY_USERS * 2, video_id=[1] * 20 + [2] * 20)

    tune, test = split_users(df, tune_share=0.5, salt="s")

    assert set(tune["user_id"]).isdisjoint(test["user_id"])
    assert tune.height + test.height == df.height
    assert tune.group_by("user_id").len()["len"].to_list() == [2] * tune["user_id"].n_unique()


def test_split_users_is_deterministic_and_depends_on_the_salt(make_interactions):
    df = make_interactions(user_id=MANY_USERS)

    first, _ = split_users(df, tune_share=0.5, salt="a")
    again, _ = split_users(df, tune_share=0.5, salt="a")
    other, _ = split_users(df, tune_share=0.5, salt="b")

    assert first["user_id"].to_list() == again["user_id"].to_list()
    assert first["user_id"].to_list() != other["user_id"].to_list()


def test_split_users_rejects_a_share_that_leaves_a_side_empty(make_interactions):
    with pytest.raises(ValueError, match="empty"):
        split_users(make_interactions(user_id=[0]), tune_share=0.5, salt="s")


# --- building and summarizing ------------------------------------------------------------


def test_build_splits_fits_label_cutoffs_on_training_data_only(make_interactions):
    # The extreme validation view would raise the cutoff if it leaked into fitting.
    big = make_interactions(
        watch_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 50.0], event_time=[at(20)] * 5 + [at(31)]
    )
    small = make_interactions(user_id=MANY_USERS, watch_ratio=[0.45] * 20)

    splits, thresholds = build_splits(big, small, LABEL, SPLIT)

    assert thresholds.watch_ratio_thresholds == pytest.approx((0.42,))
    observed = pl.concat([splits["tune"], splits["test"]])
    assert observed["is_positive"].all()


def test_build_splits_labels_all_four_splits(make_interactions):
    big = make_interactions(event_time=[at(20), at(31)])
    small = make_interactions(user_id=MANY_USERS)

    splits, _ = build_splits(big, small, LABEL, SPLIT)

    assert set(splits) == {"train", "valid", "tune", "test"}
    for frame in splits.values():
        assert {"duration_bucket", "is_positive"} <= set(frame.columns)


def test_summarize_splits_reports_cold_start_repeats_and_test_coverage(make_interactions):
    train = make_interactions(video_id=[10, 11], event_time=[at(20), at(21)])
    # One re-watch of a training pair, then two views of a video never seen in training.
    valid = make_interactions(video_id=[10, 12, 12], event_time=[at(30), at(30), at(31)])
    tune = make_interactions(user_id=[0], video_id=[10])
    test = make_interactions(
        user_id=[0, 5], video_id=[10, 13], watch_ratio=[0.5, 0.1], event_time=[at(5), None]
    )
    splits = {
        name: label_interactions(frame, ONE_BUCKET)
        for name, frame in {"train": train, "valid": valid, "tune": tune, "test": test}.items()
    }

    summary = summarize_splits(splits, ONE_BUCKET, SPLIT)

    assert summary["valid_start"] == "2020-08-30"
    assert summary["valid"]["cold_start_videos"] == 1
    assert summary["valid"]["cold_start_video_row_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["valid"]["repeat_pair_row_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["test"]["users_in_train_share"] == pytest.approx(0.5)
    assert summary["test"]["videos_in_train_share"] == pytest.approx(0.5)
    assert summary["test"]["null_event_time_share"] == pytest.approx(0.5)
    assert summary["test"]["first_event"] == at(5).isoformat()
    assert summary["test"]["positive_rate"] == pytest.approx(0.5)
    assert set(summary["test"]["length_balance"]) == {
        "slices",
        "min_positive_rate",
        "max_positive_rate",
    }
    assert summary["label"] == ONE_BUCKET.to_dict()


# --- integration: full stage -------------------------------------------------------------


@pytest.mark.integration
def test_run_split_writes_validated_splits_and_summary(make_interactions, tmp_path):
    processed, out_dir = tmp_path / "processed", tmp_path / "splits"
    processed.mkdir()
    make_interactions(
        video_id=[10, 11, 12], watch_ratio=[0.2, 0.9, 0.5], event_time=[at(20), at(21), at(31)]
    ).write_parquet(processed / "big_matrix.parquet")
    make_interactions(user_id=MANY_USERS, video_id=[13] * 20).write_parquet(
        processed / "small_matrix.parquet"
    )
    summary_path = tmp_path / "reports" / "split_summary.json"

    run_split(processed, out_dir, summary_path, LABEL, SPLIT)

    assert sorted(p.stem for p in out_dir.glob("*.parquet")) == ["test", "train", "tune", "valid"]
    assert pl.read_parquet(out_dir / "train.parquet").schema["is_positive"] == pl.Boolean
    summary = json.loads(summary_path.read_text())
    assert (summary["train"]["rows"], summary["valid"]["rows"]) == (2, 1)
    assert summary["tune"]["users"] + summary["test"]["users"] == 20


@pytest.mark.integration
def test_run_split_validates_its_inputs_before_writing_anything(make_interactions, tmp_path):
    processed, out_dir = tmp_path / "processed", tmp_path / "splits"
    processed.mkdir()
    make_interactions(watch_ratio=[-1.0, 0.5], event_time=[at(20), at(31)]).write_parquet(
        processed / "big_matrix.parquet"
    )
    make_interactions(user_id=MANY_USERS).write_parquet(processed / "small_matrix.parquet")

    with pytest.raises(pa.errors.SchemaErrors, match="watch_ratio"):
        run_split(processed, out_dir, tmp_path / "s.json", LABEL, SPLIT)

    assert not out_dir.exists()

import math

import polars as pl
import pytest

from recsys.data.labels import (
    DurationThresholds,
    fit_duration_thresholds,
    label_interactions,
    length_buckets,
    positive_rate_by_length,
)

NATURAL_WATCH_MS = 7500  # people watch ~7.5 s whatever the length (the duration bias)


# --- length buckets ----------------------------------------------------------------------


def test_length_buckets_are_log_spaced_when_every_range_is_well_populated():
    durations = pl.Series([1000] * 5 + [2000] * 5 + [3000] * 5)

    edges = length_buckets(durations, width_ratio=1.5, min_views=5)

    assert edges == pytest.approx((1500.0, 2250.0))


def test_length_buckets_merge_sparse_ranges_until_each_holds_min_views():
    # One lonely 1.5 s view between two dense lengths: its range is merged into the next bucket.
    durations = pl.Series([1000] * 10 + [1500] + [5000] * 10)

    edges = length_buckets(durations, width_ratio=1.25, min_views=5)

    assert edges == pytest.approx((1250.0,))


def test_length_buckets_collapse_to_one_bucket_when_there_is_too_little_data():
    assert length_buckets(pl.Series([1000, 9000]), width_ratio=1.25, min_views=5) == ()


@pytest.mark.parametrize(("width_ratio", "min_views"), [(1.0, 5), (0.5, 5), (1.25, 0)])
def test_length_buckets_reject_invalid_arguments(width_ratio, min_views):
    with pytest.raises(ValueError, match=r"width_ratio|min_views"):
        length_buckets(pl.Series([1000, 2000]), width_ratio=width_ratio, min_views=min_views)


# --- fitting -----------------------------------------------------------------------------


def test_fit_learns_a_watch_ratio_cutoff_per_length_bucket(make_interactions):
    train = make_interactions(
        video_duration=[1000] * 5 + [60000] * 5,
        watch_ratio=[1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    thresholds = fit_duration_thresholds(
        train, width_ratio=2.0, min_views_per_bucket=5, quantile=0.8
    )

    assert thresholds.bucket_edges_ms == pytest.approx((2000.0,))
    assert thresholds.watch_ratio_thresholds == pytest.approx((4.2, 0.42))


@pytest.mark.parametrize(
    ("width_ratio", "min_views", "quantile"),
    [(1.0, 5, 0.8), (1.25, 0, 0.8), (1.25, 5, 0.0), (1.25, 5, 1.0)],
)
def test_fit_rejects_invalid_arguments(make_interactions, width_ratio, min_views, quantile):
    with pytest.raises(ValueError, match=r"width_ratio|min_views|quantile"):
        fit_duration_thresholds(
            make_interactions(),
            width_ratio=width_ratio,
            min_views_per_bucket=min_views,
            quantile=quantile,
        )


def test_fit_rejects_empty_training_data(make_interactions):
    with pytest.raises(ValueError, match="empty"):
        fit_duration_thresholds(
            make_interactions().clear(), width_ratio=1.25, min_views_per_bucket=5, quantile=0.8
        )


# --- labeling ----------------------------------------------------------------------------


def test_label_marks_views_strictly_above_their_bucket_cutoff(make_interactions):
    thresholds = DurationThresholds((1000,), (2.0, 0.5), quantile=0.8)
    df = make_interactions(
        video_duration=[1000, 1000, 60000, 60000], watch_ratio=[2.0, 2.1, 0.5, 0.6]
    )

    labeled = label_interactions(df, thresholds)

    assert labeled["duration_bucket"].to_list() == [0, 0, 1, 1]
    assert labeled["is_positive"].to_list() == [False, True, False, True]


def test_label_removes_duration_bias_on_heavy_tailed_lengths(make_interactions):
    # Lengths from 5 s to 160 s, identical *relative* engagement at every length: the raw
    # watch ratio is 32x higher for the shortest videos, the label rate must not be.
    lengths_ms = [5000, 10000, 20000, 40000, 80000, 160000]
    engagement = [i / 100 for i in range(1, 101)]
    train = make_interactions(
        video_duration=[length for length in lengths_ms for _ in engagement],
        watch_ratio=[NATURAL_WATCH_MS / length * e for length in lengths_ms for e in engagement],
    )
    thresholds = fit_duration_thresholds(
        train, width_ratio=1.25, min_views_per_bucket=50, quantile=0.8
    )

    rates = positive_rate_by_length(
        label_interactions(train, thresholds), width_ratio=1.25, min_views=50
    )

    assert rates["positive_rate"].to_list() == pytest.approx([0.2] * len(lengths_ms))


def test_label_assigns_unseen_durations_to_the_outermost_buckets(make_interactions):
    thresholds = DurationThresholds((1000, 5000), (3.0, 2.0, 1.0), quantile=0.8)
    df = make_interactions(video_duration=[1, 999_999], watch_ratio=[3.5, 0.5])

    labeled = label_interactions(df, thresholds)

    assert labeled["duration_bucket"].to_list() == [0, 2]
    assert labeled["is_positive"].to_list() == [True, False]


def test_label_works_with_a_single_bucket(make_interactions):
    df = make_interactions(watch_ratio=[0.3, 0.5])

    labeled = label_interactions(df, DurationThresholds((), (0.4,), quantile=0.8))

    assert labeled["is_positive"].to_list() == [False, True]


def test_label_returns_a_new_frame_and_leaves_its_input_unchanged(make_interactions):
    df = make_interactions(watch_ratio=[0.3, 0.5])
    snapshot = df.clone()

    label_interactions(df, DurationThresholds((), (0.4,), quantile=0.8))

    assert df.equals(snapshot)


# --- residual-bias check -----------------------------------------------------------------


def test_positive_rate_by_length_reports_each_slice_with_its_length_range(make_interactions):
    labeled = make_interactions(
        video_duration=[1000, 1000, 4000, 4000], watch_ratio=[0.1, 0.9, 0.1, 0.1]
    ).with_columns(is_positive=pl.col("watch_ratio") > 0.5)

    rates = positive_rate_by_length(labeled, width_ratio=2.0, min_views=2)

    assert rates.columns == ["length_min_s", "length_max_s", "views", "positive_rate"]
    assert rates.rows() == [(1.0, 1.0, 2, 0.5), (4.0, 4.0, 2, 0.0)]


# --- thresholds value object -------------------------------------------------------------


def test_thresholds_reject_mismatched_edges_and_cutoffs():
    with pytest.raises(ValueError, match="one more"):
        DurationThresholds((1000,), (2.0,), quantile=0.8)


def test_thresholds_reject_unsorted_edges():
    with pytest.raises(ValueError, match="increasing"):
        DurationThresholds((5000, 1000), (1.0, 2.0, 3.0), quantile=0.8)


@pytest.mark.parametrize("cutoff", [math.inf, math.nan])
def test_thresholds_reject_non_finite_cutoffs(cutoff):
    with pytest.raises(ValueError, match="finite"):
        DurationThresholds((), (cutoff,), quantile=0.8)


def test_thresholds_reject_a_quantile_outside_the_unit_interval():
    with pytest.raises(ValueError, match="quantile"):
        DurationThresholds((), (0.4,), quantile=1.5)


def test_thresholds_serialize_to_plain_json_types():
    thresholds = DurationThresholds((1000,), (2.0, 0.5), quantile=0.8)

    assert thresholds.to_dict() == {
        "quantile": 0.8,
        "bucket_edges_ms": [1000],
        "watch_ratio_thresholds": [2.0, 0.5],
    }

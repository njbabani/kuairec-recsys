import math

import polars as pl
import pytest

from recsys.evaluation.ranking import (
    bootstrap_mean,
    evaluate,
    paired_bootstrap_difference,
    paired_win_rate,
    per_user_metrics,
    popularity_percentiles,
    rank_candidates,
    top_k_concentration,
)


def scored_frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "video_id", "score", "is_positive"], orient="row")


# User 1: positives at ranks 1 and 3. User 2: no positives. User 3: both candidates positive.
HAND_COMPUTED = scored_frame(
    [
        (1, 10, 0.9, True),
        (1, 11, 0.8, False),
        (1, 12, 0.7, True),
        (1, 13, 0.1, False),
        (2, 10, 0.5, False),
        (2, 11, 0.4, False),
        (3, 10, 0.3, True),
        (3, 11, 0.2, True),
    ]
)


def test_rank_candidates_orders_each_user_by_score_and_breaks_ties_by_video_id():
    scored = scored_frame([(1, 12, 0.5, True), (1, 10, 0.5, False), (1, 11, 0.9, False)])

    ranked = rank_candidates(scored)

    assert ranked.select("video_id", "rank").rows() == [(11, 1), (10, 2), (12, 3)]


def test_per_user_metrics_match_hand_computed_values():
    metrics = per_user_metrics(HAND_COMPUTED, ks=(2,)).sort("user_id")

    assert metrics["user_id"].to_list() == [1, 3]  # user 2 has nothing relevant to find
    user_1, user_3 = metrics.rows(named=True)
    assert user_1["precision_at_2"] == pytest.approx(0.5)
    assert user_1["recall_at_2"] == pytest.approx(0.5)
    assert user_1["ndcg_at_2"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert user_1["map_at_2"] == pytest.approx(0.5)
    for name in ("precision_at_2", "recall_at_2", "ndcg_at_2", "map_at_2"):
        assert user_3[name] == pytest.approx(1.0)


def test_metrics_when_a_user_has_more_positives_than_k_at_several_ks():
    # Positives at ranks 1, 3 and 4 out of 5 candidates (the usual case: ~500 positives vs k=10).
    scored = scored_frame(
        [
            (1, 10, 0.9, True),
            (1, 11, 0.8, False),
            (1, 12, 0.7, True),
            (1, 13, 0.6, True),
            (1, 14, 0.1, False),
        ]
    )

    user = per_user_metrics(scored, ks=(2, 4)).row(0, named=True)

    log2 = math.log2
    assert user["precision_at_2"] == pytest.approx(1 / 2)
    assert user["recall_at_2"] == pytest.approx(1 / 3)
    assert user["ndcg_at_2"] == pytest.approx(1 / (1 + 1 / log2(3)))  # ideal: 2 hits in top 2
    assert user["map_at_2"] == pytest.approx(1 / 2)  # normalised by min(3 positives, k=2)
    assert user["precision_at_4"] == pytest.approx(3 / 4)
    assert user["recall_at_4"] == pytest.approx(1.0)
    assert user["ndcg_at_4"] == pytest.approx(
        (1 + 1 / log2(4) + 1 / log2(5)) / (1 + 1 / log2(3) + 1 / log2(4))
    )
    assert user["map_at_4"] == pytest.approx((1 / 1 + 2 / 3 + 3 / 4) / 3)


def test_metrics_do_not_depend_on_input_row_order():
    shuffled = HAND_COMPUTED.sample(fraction=1.0, shuffle=True, seed=3)

    assert per_user_metrics(shuffled, ks=(1, 2)).equals(per_user_metrics(HAND_COMPUTED, ks=(1, 2)))


@pytest.mark.parametrize("bad_score", [None, float("nan")], ids=["null", "nan"])
def test_evaluate_refuses_missing_or_nan_scores_instead_of_ranking_them_first(bad_score):
    scored = HAND_COMPUTED.with_columns(
        score=pl.when(pl.col("video_id") == 11)
        .then(pl.lit(bad_score, dtype=pl.Float64))
        .otherwise(pl.col("score"))
    )

    with pytest.raises(ValueError, match="null or NaN"):
        evaluate(scored, ks=(2,), popularity=None, n_bootstrap=10, seed=0)


def test_minus_infinity_is_a_valid_score_that_ranks_last():
    scored = scored_frame([(1, 10, -math.inf, True), (1, 11, 0.1, False)])

    assert rank_candidates(scored)["video_id"].to_list() == [11, 10]


def test_paired_win_rate_counts_ties_as_half():
    model_a = pl.DataFrame({"user_id": [1, 2, 3, 4], "ndcg_at_10": [0.5, 0.2, 0.3, 0.9]})
    model_b = pl.DataFrame({"user_id": [1, 2, 3, 4], "ndcg_at_10": [0.4, 0.4, 0.3, 0.1]})

    assert paired_win_rate(model_a, model_b, "ndcg_at_10") == pytest.approx((2 + 0.5) / 4)


def test_precision_uses_the_candidate_count_when_k_exceeds_it():
    metrics = per_user_metrics(scored_frame([(1, 10, 0.9, True), (1, 11, 0.1, False)]), ks=(5,))

    assert metrics["precision_at_5"].to_list() == pytest.approx([0.5])


def test_bootstrap_mean_is_deterministic_and_brackets_the_mean():
    values = pl.Series([0.1, 0.4, 0.5, 0.9, 0.3])

    first = bootstrap_mean(values, n_samples=500, seed=7)
    again = bootstrap_mean(values, n_samples=500, seed=7)

    assert first == again
    mean, low, high = first
    assert mean == pytest.approx(0.44)
    assert low <= mean <= high


def test_bootstrap_of_constant_values_has_zero_width():
    assert bootstrap_mean(pl.Series([0.3, 0.3, 0.3]), n_samples=100, seed=0) == pytest.approx(
        (0.3, 0.3, 0.3)
    )


def test_paired_difference_compares_the_same_users_and_ignores_unshared_ones():
    model_a = pl.DataFrame({"user_id": [1, 2, 9], "ndcg_at_10": [0.5, 0.7, 1.0]})
    model_b = pl.DataFrame({"user_id": [1, 2, 8], "ndcg_at_10": [0.4, 0.4, 0.0]})

    mean, low, high = paired_bootstrap_difference(
        model_a, model_b, "ndcg_at_10", n_samples=500, seed=0
    )

    assert mean == pytest.approx(0.2)  # users 1 and 2 only: (0.1 + 0.3) / 2
    tolerance = 1e-9  # 0.5 - 0.4 is 0.0999... in floating point
    assert 0.1 - tolerance <= low <= mean <= high <= 0.3 + tolerance


def test_paired_difference_of_a_model_with_itself_is_exactly_zero():
    per_user = pl.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.2, 0.9, 0.4]})

    assert paired_bootstrap_difference(
        per_user, per_user, "ndcg_at_10", n_samples=100, seed=0
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_popularity_percentiles_rank_candidates_by_training_views():
    train = pl.DataFrame({"video_id": [1, 1, 1, 2, 2, 3]})  # 3, 2 and 1 views; video 4 unseen

    percentiles = popularity_percentiles(train, candidate_videos=pl.Series([1, 2, 3, 4]))

    assert dict(percentiles.rows()) == pytest.approx({1: 1.0, 2: 2 / 3, 3: 1 / 3, 4: 0.0})


def test_top_k_concentration_measures_coverage_gini_and_popularity():
    ranked = rank_candidates(
        scored_frame([(1, 1, 0.9, True), (1, 2, 0.1, False), (2, 1, 0.8, True), (2, 3, 0.2, False)])
    )
    popularity = pl.DataFrame({"video_id": [1, 2, 3], "popularity_pct": [1.0, 0.5, 0.0]})

    result = top_k_concentration(ranked, k=1, popularity=popularity)

    # Both users get video 1: one of three candidates covered, all exposure on one video.
    assert result == pytest.approx(
        {"coverage_at_1": 1 / 3, "gini_at_1": 2 / 3, "popularity_pct_at_1": 1.0}
    )


def test_evaluate_returns_a_flat_json_ready_summary_with_confidence_intervals():
    result = evaluate(HAND_COMPUTED, ks=(2,), popularity=None, n_bootstrap=200, seed=0)

    summary = result.summary
    assert summary["users_evaluated"] == 2
    assert summary["users_without_positives"] == 1
    assert summary["precision_at_2"] == pytest.approx((0.5 + 1.0) / 2)
    assert summary["ndcg_at_2_ci_low"] <= summary["ndcg_at_2"] <= summary["ndcg_at_2_ci_high"]
    assert {"coverage_at_2", "gini_at_2"} <= set(summary)
    assert all(isinstance(value, int | float) for value in summary.values())
    assert result.per_user.height == 2


def test_evaluate_rejects_frames_without_scores():
    with pytest.raises(ValueError, match="score"):
        evaluate(HAND_COMPUTED.drop("score"), ks=(2,), popularity=None, n_bootstrap=10, seed=0)

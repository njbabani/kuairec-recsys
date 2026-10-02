import polars as pl
import pytest

from recsys.reranking.candidates import PrecomputedScorer, ranker_training_rows

TABLE = pl.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 11, 10], "score": [0.5, -1.0, 2.0]})


def test_precomputed_scores_are_looked_up_in_the_order_of_the_pairs():
    pairs = pl.DataFrame({"user_id": [2, 1, 1], "video_id": [10, 11, 10]})

    scores = PrecomputedScorer(TABLE, name="two_tower").score(pairs)

    assert scores.to_list() == [2.0, -1.0, 0.5]


def test_a_pair_without_a_precomputed_score_is_an_error_not_a_guess():
    pairs = pl.DataFrame({"user_id": [1, 3], "video_id": [10, 10]})

    with pytest.raises(ValueError, match="1 pair"):
        PrecomputedScorer(TABLE, name="two_tower").score(pairs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_precomputed_scores_are_rejected(bad):
    table = TABLE.with_columns(score=pl.Series([0.5, bad, 2.0]))

    with pytest.raises(ValueError, match="finite"):
        PrecomputedScorer(table, name="two_tower")


def test_duplicate_precomputed_pairs_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        PrecomputedScorer(pl.concat([TABLE, TABLE]), name="two_tower")


def test_ranker_training_rows_look_like_what_evaluation_will_ask():
    train = pl.DataFrame({"user_id": [1, 2, 2, 2], "video_id": [10, 10, 11, 12]})
    valid = pl.DataFrame(
        [
            (1, 10, True, 1),  # re-watch of a training pair: the evaluation has none, dropped
            (1, 11, True, 2),
            (1, 12, False, 3),
            (1, 99, True, 4),  # video never seen in training: the evaluation has none, dropped
            (3, 11, False, 5),  # user 3 only skipped: nothing to order, dropped
            (3, 12, False, 6),
            (4, 11, True, 7),  # user 4 liked everything: nothing to order, dropped
            (4, 12, True, 8),
        ],
        schema=["user_id", "video_id", "is_positive", "event_time"],
        orient="row",
    )

    rows = ranker_training_rows(valid, train)

    assert rows.columns == ["user_id", "video_id", "is_positive"]
    assert rows.sort("video_id").rows() == [(1, 11, True), (1, 12, False)]


def test_repeated_views_in_the_week_keep_only_the_reaction_to_the_first_one():
    # The fully observed matrix has one reaction per pair, so each pair keeps its first view.
    train = pl.DataFrame({"user_id": [9, 9], "video_id": [10, 11]})
    valid = pl.DataFrame(
        [
            (1, 11, True, 5),  # second view of (1, 11): dropped
            (1, 11, False, 1),  # first view of (1, 11): kept, even though it disagrees
            (1, 10, True, 3),
        ],
        schema=["user_id", "video_id", "is_positive", "event_time"],
        orient="row",
    )

    rows = ranker_training_rows(valid, train)

    assert rows.sort("video_id").rows() == [(1, 10, True), (1, 11, False)]

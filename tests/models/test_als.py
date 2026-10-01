import math

import polars as pl
import pytest

from recsys.models.als import ALSRecommender

SETTINGS = {"factors": 4, "regularization": 0.01, "alpha": 10.0, "iterations": 30, "seed": 0}


def two_taste_groups() -> pl.DataFrame:
    """Users 0-4 like videos 0-4, users 5-9 like videos 5-9; pair (0, 0) is held out."""
    rows = [
        (user, video, True)
        for user in range(10)
        for video in (range(5) if user < 5 else range(5, 10))
        if (user, video) != (0, 0)
    ]
    return pl.DataFrame(rows, schema=["user_id", "video_id", "is_positive"], orient="row")


def test_als_ranks_an_unseen_in_taste_video_above_other_groups_videos():
    model = ALSRecommender(**SETTINGS).fit(two_taste_groups())
    pairs = pl.DataFrame({"user_id": [0] * 6, "video_id": [0, 5, 6, 7, 8, 9]})

    scores = model.score(pairs).to_list()

    assert scores[0] > max(scores[1:])


def test_als_gives_videos_never_seen_in_training_the_lowest_possible_score():
    model = ALSRecommender(**SETTINGS).fit(two_taste_groups())

    scores = model.score(pl.DataFrame({"user_id": [0, 0], "video_id": [1, 99]}))

    assert math.isfinite(scores[0])
    assert scores[1] == -math.inf


@pytest.mark.parametrize("user_id", [42, 99], ids=["no-positive-history", "never-seen"])
def test_als_ranks_users_without_positive_history_by_the_average_taste(user_id):
    # User 42 only has non-positive views; user 99 is not in training at all. Both get the
    # average user vector instead of a zero vector that would rank everything equally.
    train = pl.concat(
        [
            two_taste_groups(),
            pl.DataFrame({"user_id": [42], "video_id": [3], "is_positive": [False]}),
        ]
    )
    model = ALSRecommender(**SETTINGS).fit(train)

    scores = model.score(pl.DataFrame({"user_id": [user_id] * 10, "video_id": list(range(10))}))

    assert all(math.isfinite(score) for score in scores)
    assert len({round(score, 9) for score in scores}) > 1


def test_als_accepts_zero_alpha_as_unweighted_positives():
    model = ALSRecommender(**{**SETTINGS, "alpha": 0.0}).fit(two_taste_groups())

    scores = model.score(pl.DataFrame({"user_id": [0, 0], "video_id": [0, 5]})).to_list()

    assert scores[0] > scores[1]


def test_als_is_reproducible_for_a_seed():
    pairs = pl.DataFrame({"user_id": [0, 3, 7], "video_id": [0, 2, 8]})

    first = ALSRecommender(**SETTINGS).fit(two_taste_groups()).score(pairs)
    again = ALSRecommender(**SETTINGS).fit(two_taste_groups()).score(pairs)

    assert first.to_list() == pytest.approx(again.to_list())


def test_als_refuses_training_data_without_positives():
    train = two_taste_groups().with_columns(is_positive=pl.lit(False))

    with pytest.raises(ValueError, match="positive"):
        ALSRecommender(**SETTINGS).fit(train)


def test_als_scoring_before_fitting_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        ALSRecommender(**SETTINGS).score(pl.DataFrame({"user_id": [0], "video_id": [0]}))

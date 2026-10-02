import math

import polars as pl
import pytest

from recsys.abtest.outcomes import session_outcomes, sessions

SCORES = pl.DataFrame(
    {
        "user_id": [1, 1, 1, 2, 2, 2],
        "video_id": [10, 11, 12, 10, 11, 12],
        "score": [0.9, 0.1, 0.5, 0.2, 0.2, 0.7],
    }
)
REACTIONS = pl.DataFrame(
    {
        "user_id": [1, 1, 1, 2, 2, 2],
        "video_id": [10, 11, 12, 10, 11, 12],
        "is_positive": [True, True, False, False, True, True],
        "play_duration": [5000, 9000, 1000, 2000, 3000, 4000],
    }
)


def test_a_session_is_each_users_top_videos_by_score():
    session = sessions(SCORES, "score", length=2)

    assert session.sort("user_id", "position").rows() == [
        (1, 10, 1),
        (1, 12, 2),
        (2, 12, 1),
        (2, 10, 2),  # tie at 0.2 broken by the lower video id
    ]


def with_score_of_video_12(score: float | None) -> pl.DataFrame:
    return SCORES.with_columns(
        score=pl.when(pl.col("video_id") == 12).then(score).otherwise(pl.col("score"))
    )


@pytest.mark.parametrize("bad_score", [math.nan, None], ids=["nan", "null"])
def test_a_session_refuses_scores_that_cannot_be_ranked(bad_score):
    with pytest.raises(ValueError, match="null or NaN"):
        sessions(with_score_of_video_12(bad_score), "score", length=2)


def test_a_minus_infinite_score_ranks_a_video_last():
    session = sessions(with_score_of_video_12(-math.inf), "score", length=3)

    assert session.filter(pl.col("video_id") == 12)["position"].to_list() == [3, 3]


def test_session_outcomes_read_off_what_each_user_actually_did():
    session = sessions(SCORES, "score", length=2)  # user 1: videos 10, 12; user 2: 12, 10

    outcomes = session_outcomes(session, REACTIONS).sort("user_id")

    assert outcomes["liked"].to_list() == [1, 1]
    assert outcomes["watch_time_s"].to_list() == pytest.approx([6.0, 6.0])
    top_one = session_outcomes(sessions(SCORES, "score", length=1), REACTIONS).sort("user_id")
    assert top_one["liked"].to_list() == [1, 1]  # user 1 liked video 10, user 2 video 12
    assert top_one["watch_time_s"].to_list() == pytest.approx([5.0, 4.0])


@pytest.mark.parametrize("column", ["is_positive", "play_duration"])
def test_a_session_video_without_a_complete_reaction_is_an_error(column):
    reactions = REACTIONS.with_columns(
        pl.when((pl.col("user_id") == 1) & (pl.col("video_id") == 12))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match="reaction"):
        session_outcomes(sessions(SCORES, "score", length=2), reactions)


def test_a_session_video_missing_from_the_reactions_is_an_error():
    reactions = REACTIONS.filter(~((pl.col("user_id") == 1) & (pl.col("video_id") == 12)))

    with pytest.raises(ValueError, match="reaction"):
        session_outcomes(sessions(SCORES, "score", length=2), reactions)

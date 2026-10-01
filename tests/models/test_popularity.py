import polars as pl
import pytest

from recsys.models.popularity import PopularityRecommender, RandomRecommender

# Views per video: 10 -> 3 (1 positive), 11 -> 2 (2 positives), 12 -> 1 (1 positive).
TRAIN = pl.DataFrame(
    {
        "user_id": [0, 1, 2, 0, 1, 0],
        "video_id": [10, 10, 10, 11, 11, 12],
        "is_positive": [False, False, True, True, True, True],
    }
)
GLOBAL_RATE = 4 / 6
PAIRS = pl.DataFrame({"user_id": [5, 5, 5, 5], "video_id": [12, 10, 11, 99]})


def test_view_popularity_scores_videos_by_training_views_in_input_order():
    scores = PopularityRecommender(by="views").fit(TRAIN).score(PAIRS)

    assert scores.to_list() == [1.0, 3.0, 2.0, 0.0]


def test_engagement_popularity_without_a_prior_is_the_raw_positive_rate():
    scores = PopularityRecommender(by="engagement", prior_strength=0).fit(TRAIN).score(PAIRS)

    assert scores.to_list() == pytest.approx([1.0, 1 / 3, 1.0, GLOBAL_RATE])


def test_engagement_popularity_shrinks_small_samples_towards_the_global_rate():
    scores = PopularityRecommender(by="engagement", prior_strength=2).fit(TRAIN).score(PAIRS)

    prior = 2 * GLOBAL_RATE
    assert scores.to_list() == pytest.approx(
        [(1 + prior) / (1 + 2), (1 + prior) / (3 + 2), (2 + prior) / (2 + 2), GLOBAL_RATE]
    )


def test_scoring_before_fitting_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        PopularityRecommender(by="views").score(PAIRS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"by": "clicks"}, "by"), ({"by": "engagement", "prior_strength": -1}, "prior_strength")],
)
def test_popularity_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PopularityRecommender(**kwargs)


def test_random_recommender_is_reproducible_for_a_seed():
    first = RandomRecommender(seed=1).fit(TRAIN).score(PAIRS)
    again = RandomRecommender(seed=1).fit(TRAIN).score(PAIRS)
    other = RandomRecommender(seed=2).fit(TRAIN).score(PAIRS)

    assert first.to_list() == again.to_list()
    assert first.to_list() != other.to_list()
    assert first.len() == PAIRS.height

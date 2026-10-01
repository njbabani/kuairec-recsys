import polars as pl
import pytest

from recsys.models.category_affinity import CategoryAffinityRecommender
from recsys.models.popularity import PopularityRecommender

COMEDY, DANCE = 9, 1
# Videos 1-2 are scored; 3-4 carry the users' history; 5-7 are never seen in training.
CATEGORIES = pl.DataFrame(
    {
        "video_id": [1, 2, 3, 4, 5, 6, 7],
        "category_ids": [[COMEDY], [DANCE], [COMEDY], [DANCE], [COMEDY], [DANCE], [COMEDY, DANCE]],
    }
)


def views(rows: list[tuple[int, int, bool]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "video_id", "is_positive"], orient="row")


def history() -> pl.DataFrame:
    """User 0 likes comedy and skips dance, user 1 the reverse; videos 1 and 2 are liked equally."""
    taste = [(0, 3, True), (0, 4, False), (1, 3, False), (1, 4, True)] * 5
    shared = [(9, 1, True), (9, 1, False), (9, 2, True), (9, 2, False)]
    return views(taste + shared)


def model(prior_strength: float = 2.0) -> CategoryAffinityRecommender:
    return CategoryAffinityRecommender(
        CATEGORIES,
        prior_strength=prior_strength,
        base=PopularityRecommender(by="engagement", prior_strength=2.0),
    )


def scores(recommender, user: int, videos: list[int]) -> list[float]:
    pairs = pl.DataFrame({"user_id": [user] * len(videos), "video_id": videos})
    return recommender.score(pairs).to_list()


def test_each_user_sees_their_preferred_category_first():
    fitted = model().fit(history())

    comedy_fan = scores(fitted, user=0, videos=[1, 2])
    dance_fan = scores(fitted, user=1, videos=[1, 2])

    assert comedy_fan[0] > comedy_fan[1]
    assert dance_fan[1] > dance_fan[0]


def test_taste_also_ranks_videos_never_seen_in_training():
    fitted = model().fit(history())

    cold_comedy, cold_dance = scores(fitted, user=0, videos=[5, 6])

    assert cold_comedy > cold_dance


def test_a_user_without_history_gets_the_base_popularity_scores():
    train = history()
    base = PopularityRecommender(by="engagement", prior_strength=2.0).fit(train)
    pairs = pl.DataFrame({"user_id": [42, 42, 42], "video_id": [1, 2, 99]})

    affinity = model().fit(train).score(pairs)

    assert affinity.to_list() == pytest.approx(base.score(pairs).to_list())


def test_videos_in_several_categories_average_the_users_lifts():
    fitted = model().fit(history())

    comedy, dance, both = scores(fitted, user=0, videos=[5, 6, 7])

    assert dance < both < comedy


def test_scores_keep_the_row_order_of_the_pairs():
    fitted = model().fit(history())
    pairs = pl.DataFrame({"user_id": [1, 0, 1, 0], "video_id": [1, 2, 2, 1]})

    shuffled = fitted.score(pairs).to_list()

    assert shuffled == pytest.approx([scores(fitted, u, [v])[0] for u, v in pairs.iter_rows()])


def test_a_huge_prior_washes_out_taste_and_leaves_the_base_order():
    fitted = model(prior_strength=1e9).fit(history())

    comedy_fan = scores(fitted, user=0, videos=[1, 2])

    assert comedy_fan[0] == pytest.approx(comedy_fan[1])


def test_category_affinity_refuses_training_data_without_positives():
    train = history().with_columns(is_positive=pl.lit(False))

    with pytest.raises(ValueError, match="positive"):
        model().fit(train)


def test_scoring_before_fitting_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        model().score(pl.DataFrame({"user_id": [0], "video_id": [1]}))


@pytest.mark.parametrize("prior_strength", [0.0, -1.0])
def test_category_affinity_needs_a_positive_prior_to_avoid_dividing_by_zero(prior_strength):
    with pytest.raises(ValueError, match="prior_strength"):
        model(prior_strength=prior_strength)

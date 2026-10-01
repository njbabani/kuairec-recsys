import numpy as np
import polars as pl
import pytest

from recsys.evaluation.selection import selection_optimism

N_USERS = 300


def trial_scores(true_means: list[float], noise: float, seed: int = 0) -> pl.DataFrame:
    """Per-user scores (rows) for several trials (columns) with the given true means."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {f"trial_{i}": rng.normal(mean, noise, size=N_USERS) for i, mean in enumerate(true_means)}
    )


def test_picking_the_best_of_many_equal_trials_looks_better_than_it_is():
    scores = trial_scores([0.5] * 20, noise=0.2)

    assert selection_optimism(scores, n_samples=200, seed=0) > 0.01


def test_there_is_almost_no_optimism_when_one_trial_clearly_wins():
    scores = trial_scores([0.5, 0.5, 0.9, 0.5], noise=0.05)

    assert abs(selection_optimism(scores, n_samples=200, seed=0)) < 0.005


@pytest.mark.parametrize(
    "scores",
    [
        pl.DataFrame({"trial_0": [0.5]}),
        pl.DataFrame({"trial_0": [0.5, None]}, schema={"trial_0": pl.Float64}),
    ],
    ids=["single-user", "missing-score"],
)
def test_selection_optimism_rejects_unusable_score_tables(scores):
    with pytest.raises(ValueError, match=r"users|missing"):
        selection_optimism(scores, n_samples=10, seed=0)

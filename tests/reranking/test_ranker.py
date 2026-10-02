import numpy as np
import polars as pl
import pytest

from recsys.reranking.features import FEATURE_COLUMNS
from recsys.reranking.ranker import (
    RankerConfig,
    contributions,
    load_ranker,
    openmp_threads,
    predict,
    ranking_data,
    save_ranker,
    train_fixed_rounds,
    train_ranker,
)

CONFIG = RankerConfig(
    num_leaves=7,
    min_child_samples=5,
    learning_rate=0.1,
    feature_fraction=1.0,
    max_rounds=200,
    early_stopping_rounds=20,
    eval_at=5,
    seed=0,
)


def synthetic(n_users: int, seed: int) -> tuple[pl.DataFrame, pl.Series]:
    """20 candidates per user; a candidate is liked when two_tower_score is high."""
    rng = np.random.default_rng(seed)
    rows = n_users * 20
    features = pl.DataFrame(
        {
            "user_id": np.repeat(np.arange(n_users), 20),
            "video_id": np.tile(np.arange(20), n_users),
            **{column: rng.normal(size=rows) for column in FEATURE_COLUMNS},
        }
    ).with_columns(first_category=pl.lit(1, pl.Int32), user_active_degree=pl.lit(0, pl.Int32))
    noise = rng.normal(scale=0.3, size=rows)
    labels = pl.Series("label", (features["two_tower_score"].to_numpy() + noise > 0.8).astype(int))
    return features, labels


@pytest.fixture(scope="module")
def trained():
    train = ranking_data(*synthetic(60, seed=1))
    holdout = ranking_data(*synthetic(20, seed=2))
    return train_ranker(train, holdout, CONFIG)


def test_ranking_data_groups_each_users_rows_together():
    features = pl.DataFrame(
        {"user_id": [2, 1, 2, 1, 1], "video_id": [1, 2, 3, 4, 5]}
        | {column: [0.0] * 5 for column in FEATURE_COLUMNS}
    )

    data = ranking_data(features, pl.Series([1, 0, 0, 1, 0]))

    assert data.groups.tolist() == [3, 2]  # user 1 first, then user 2
    assert data.labels.tolist() == [0, 1, 0, 1, 0]
    assert data.features.columns == list(FEATURE_COLUMNS)


def test_ranker_learns_the_signal_and_stops_early(trained):
    assert trained.holdout_score > 0.9
    assert 1 <= trained.best_iteration < CONFIG.max_rounds
    assert len(trained.curve) >= trained.best_iteration


def test_predictions_put_the_signal_first(trained):
    features, labels = synthetic(10, seed=3)

    scores = predict(trained.booster, features.select(FEATURE_COLUMNS))

    liked = labels.to_numpy() == 1
    assert scores[liked].mean() > scores[~liked].mean()


def test_saved_ranker_reloads_with_identical_predictions(trained, tmp_path):
    features, _ = synthetic(5, seed=4)
    path = tmp_path / "reranker" / "model.txt"

    save_ranker(trained.booster, path)
    reloaded = load_ranker(path)

    assert [p.name for p in path.parent.iterdir()] == ["model.txt"]
    np.testing.assert_allclose(
        predict(reloaded, features.select(FEATURE_COLUMNS)),
        predict(trained.booster, features.select(FEATURE_COLUMNS)),
    )


def test_shap_contributions_add_up_to_the_prediction(trained):
    features, _ = synthetic(3, seed=5)
    matrix = features.select(FEATURE_COLUMNS)

    shap = contributions(trained.booster, matrix)

    assert shap.columns == [*FEATURE_COLUMNS, "bias"]
    np.testing.assert_allclose(
        shap.sum_horizontal().to_numpy(), predict(trained.booster, matrix), atol=1e-6
    )
    assert shap["two_tower_score"].abs().mean() > shap["log_user_views"].abs().mean()


def test_training_is_reproducible_for_a_seed():
    train = ranking_data(*synthetic(30, seed=6))
    holdout = ranking_data(*synthetic(10, seed=7))

    first = train_ranker(train, holdout, CONFIG)
    again = train_ranker(train, holdout, CONFIG)

    assert first.curve == again.curve


@pytest.mark.parametrize(
    ("platform", "torch_loaded", "expected"),
    [("darwin", True, 1), ("darwin", False, 0), ("linux", True, 0)],
    ids=["mac-with-torch", "mac-without-torch", "linux"],
)
def test_lightgbm_runs_single_threaded_next_to_pytorch_on_macos(platform, torch_loaded, expected):
    # PyTorch and LightGBM each bring an OpenMP runtime; on macOS two of them in one process
    # crash as soon as both start thread pools, so LightGBM falls back to one thread.
    modules = {"torch": object()} if torch_loaded else {}

    assert openmp_threads(platform, modules) == expected


def test_a_fixed_number_of_rounds_trains_without_a_holdout():
    data = ranking_data(*synthetic(20, seed=8))

    booster = train_fixed_rounds(data, CONFIG, rounds=7)

    assert booster.num_trees() == 7

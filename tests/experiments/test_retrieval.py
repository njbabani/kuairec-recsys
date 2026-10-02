import polars as pl
import pytest

from recsys.experiments.retrieval import SCORED_SPLITS, run_retrieval
from recsys.models.two_tower.ensemble import TwoTowerEnsemble
from recsys.models.two_tower.recommender import TwoTowerRecommender

SETTINGS = {
    "loss": "bce",
    "embedding_dim": 4,
    "hidden_dim": 8,
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.05,
    "weight_decay": 0.0,
    "device": "cpu",
}


@pytest.fixture
def saved_ensemble(labeled_splits, feature_tables, tmp_path):
    users = pl.read_parquet(feature_tables / "users.parquet")
    videos = pl.read_parquet(feature_tables / "videos.parquet")
    train = pl.read_parquet(labeled_splits / "train.parquet")
    members = [TwoTowerRecommender(users, videos, **SETTINGS, seed=seed) for seed in (0, 1)]
    ensemble = TwoTowerEnsemble(members).fit(train)
    ensemble.save(tmp_path / "models" / "two_tower")
    return ensemble


@pytest.fixture
def all_splits(labeled_splits):
    """Add valid and test files (copies of tune with other users) next to train and tune."""
    tune = pl.read_parquet(labeled_splits / "tune.parquet")
    tune.with_columns(user_id=pl.col("user_id") + 1).write_parquet(labeled_splits / "valid.parquet")
    tune.with_columns(user_id=pl.col("user_id") + 2).write_parquet(labeled_splits / "test.parquet")
    return labeled_splits


@pytest.mark.integration
def test_retrieval_scores_every_unique_pair_of_the_scored_splits(
    all_splits, saved_ensemble, tmp_path
):
    output = tmp_path / "retrieval" / "two_tower_scores.parquet"

    scores = run_retrieval(all_splits, tmp_path / "models" / "two_tower", output, device="cpu")

    expected_pairs = (
        pl.concat([pl.read_parquet(all_splits / f"{name}.parquet") for name in SCORED_SPLITS])
        .select("user_id", "video_id")
        .unique()
    )
    assert scores.height == expected_pairs.height
    assert scores.select("user_id", "video_id").is_duplicated().sum() == 0
    assert pl.read_parquet(output).equals(scores)
    direct = saved_ensemble.score(scores.select("user_id", "video_id"))
    assert scores["score"].to_list() == pytest.approx(direct.to_list())

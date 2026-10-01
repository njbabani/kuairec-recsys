import json

import pytest

from recsys.config import (
    ALSParams,
    CategoryAffinityParams,
    EngagementPopularityParams,
    ModelParams,
    RandomParams,
    TwoTowerEnsembleParams,
)
from recsys.experiments.leaderboard import REFERENCE_MODELS, run_leaderboard
from recsys.experiments.runner import load_split
from recsys.models.two_tower.ensemble import TwoTowerEnsemble
from recsys.tracking import InMemoryTracker

MODELS = ModelParams(
    random=RandomParams(seed=0),
    popularity_engagement=EngagementPopularityParams(prior_strength=1.0),
    category_affinity=CategoryAffinityParams(prior_strength=1.0),
    als=ALSParams(factors=2, regularization=0.01, alpha=10.0, iterations=20, seed=0),
    two_tower=TwoTowerEnsembleParams(
        loss="bce",
        embedding_dim=4,
        hidden_dim=8,
        epochs=30,
        batch_size=8,
        learning_rate=0.05,
        weight_decay=0.0,
        seeds=(0, 1),
        device="cpu",
    ),
)
NAMES = [
    "random",
    "popularity_views",
    "popularity_engagement",
    "category_affinity",
    "als",
    "two_tower",
]


@pytest.fixture
def report(labeled_splits, feature_tables, evaluation_params, tmp_path):
    tracker = InMemoryTracker()
    metrics_path = tmp_path / "reports" / "leaderboard.json"
    result = run_leaderboard(
        labeled_splits,
        feature_tables,
        metrics_path,
        MODELS,
        evaluation_params,
        tracker,
        models_dir=tmp_path / "models",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    return {"result": result, "tracker": tracker, "path": metrics_path, "root": tmp_path}


@pytest.mark.integration
def test_leaderboard_evaluates_every_model_on_tune_and_tracks_every_run(report):
    result, tracker = report["result"], report["tracker"]

    assert list(result["models"]) == NAMES
    assert [run["name"] for run in tracker.runs] == [f"leaderboard/{name}" for name in NAMES]
    assert all(run["status"] == "FINISHED" for run in tracker.runs)
    assert all("fit_seconds" in run["metrics"] for run in tracker.runs)
    assert all("fit_seconds" not in metrics for metrics in result["models"].values())
    assert json.loads(report["path"].read_text()) == result


@pytest.mark.integration
def test_leaderboard_compares_every_model_with_each_reference_on_the_same_users(report):
    paired = report["result"]["paired"]

    assert list(paired) == list(REFERENCE_MODELS) == ["popularity_engagement", "als"]
    for reference, comparison in paired.items():
        assert comparison["metric"] == "ndcg_at_2"
        assert set(comparison["differences"]) == set(NAMES) - {reference}
        for difference in comparison["differences"].values():
            assert difference["ci_low"] <= difference["mean"] <= difference["ci_high"]
            assert 0.0 <= difference["win_rate"] <= 1.0


@pytest.mark.integration
def test_personalised_models_learn_the_taste_groups_that_view_popularity_cannot(report):
    models = report["result"]["models"]

    for name in ("category_affinity", "als", "two_tower"):
        assert models[name]["ndcg_at_2"] > models["popularity_views"]["ndcg_at_2"], name


@pytest.mark.integration
def test_two_tower_is_a_seed_ensemble_whose_member_spread_is_reported(report, evaluation_params):
    spread = report["result"]["seed_spread"]["two_tower"]

    assert spread["seeds"] == [0, 1]
    assert len(spread["ndcg_at_2"]) == 2
    assert spread["mean"] == pytest.approx(sum(spread["ndcg_at_2"]) / 2)
    assert spread["sd"] >= 0


@pytest.mark.integration
def test_trained_two_tower_models_are_saved_and_reload_with_the_same_scores(report, labeled_splits):
    saved = report["root"] / "models" / "two_tower"
    tune = load_split(labeled_splits, "tune").select("user_id", "video_id")

    loaded = TwoTowerEnsemble.load(saved, device="cpu")

    assert sorted(path.name for path in saved.iterdir()) == ["seed_0.pt", "seed_1.pt"]
    assert [member.seed for member in loaded.members] == [0, 1]
    assert loaded.score(tune).is_finite().all()
    assert not (report["root"] / "checkpoints").exists()  # resume state cleaned up

import json

import pytest

from recsys.config import (
    ALSParams,
    BaselineParams,
    EngagementPopularityParams,
    RandomParams,
)
from recsys.experiments.baselines import run_baselines
from recsys.tracking import InMemoryTracker

BASELINES = BaselineParams(
    random=RandomParams(seed=0),
    popularity_engagement=EngagementPopularityParams(prior_strength=1.0),
    als=ALSParams(factors=2, regularization=0.01, alpha=10.0, iterations=20, seed=0),
)


@pytest.mark.integration
def test_run_baselines_evaluates_each_baseline_on_tune_and_tracks_every_run(
    labeled_splits, evaluation_params, tmp_path
):
    tracker = InMemoryTracker()
    metrics_path = tmp_path / "reports" / "baselines.json"

    report = run_baselines(labeled_splits, metrics_path, BASELINES, evaluation_params, tracker)

    names = ["random", "popularity_views", "popularity_engagement", "als"]
    assert list(report["models"]) == names
    assert [run["name"] for run in tracker.runs] == [f"baseline/{name}" for name in names]
    assert all(run["status"] == "FINISHED" for run in tracker.runs)
    assert json.loads(metrics_path.read_text()) == report


@pytest.mark.integration
def test_run_baselines_reports_paired_differences_against_the_reference_model(
    labeled_splits, evaluation_params, tmp_path
):
    report = run_baselines(
        labeled_splits, tmp_path / "m.json", BASELINES, evaluation_params, InMemoryTracker()
    )

    comparison = report["paired_vs_reference"]
    assert comparison["reference"] == "popularity_engagement"
    assert comparison["metric"] == "ndcg_at_2"
    assert set(comparison["differences"]) == {"random", "popularity_views", "als"}
    for difference in comparison["differences"].values():
        assert difference["ci_low"] <= difference["mean"] <= difference["ci_high"]
        assert 0.0 <= difference["win_rate"] <= 1.0
    assert all("fit_seconds" not in metrics for metrics in report["models"].values())


@pytest.mark.integration
def test_als_learns_the_taste_groups_that_view_popularity_cannot(
    labeled_splits, evaluation_params, tmp_path
):
    models = run_baselines(
        labeled_splits, tmp_path / "m.json", BASELINES, evaluation_params, InMemoryTracker()
    )["models"]

    assert models["als"]["ndcg_at_2"] > models["popularity_views"]["ndcg_at_2"]

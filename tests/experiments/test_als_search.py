import json

import pytest
from pydantic import ValidationError

from recsys.config import ALSSearchParams
from recsys.experiments.als_search import (
    als_grid,
    params_from_sweep_config,
    run_search,
    sweep_trial,
)
from recsys.tracking import InMemoryTracker

SWEEP_CONFIG = {"factors": 32, "regularization": 0.05, "alpha": 20, "iterations": 10, "seed": 1}


def test_als_grid_is_the_cartesian_product_of_the_search_space():
    search = ALSSearchParams(
        factors=(2, 4), regularization=(0.01,), alpha=(5.0, 10.0), iterations=5, seed=0
    )

    grid = als_grid(search)

    assert {(p.factors, p.alpha) for p in grid} == {(2, 5.0), (2, 10.0), (4, 5.0), (4, 10.0)}
    assert all(p.iterations == 5 and p.seed == 0 for p in grid)


def test_params_from_sweep_config_validates_the_sampled_values():
    assert params_from_sweep_config(SWEEP_CONFIG).factors == 32

    with pytest.raises(ValidationError):
        params_from_sweep_config({**SWEEP_CONFIG, "factors": 0})


@pytest.mark.integration
def test_run_search_tracks_every_trial_and_reports_the_best(
    labeled_splits, evaluation_params, tmp_path
):
    search = ALSSearchParams(
        factors=(1, 2), regularization=(0.01,), alpha=(10.0,), iterations=20, seed=0
    )
    tracker = InMemoryTracker()
    report_path = tmp_path / "als_search.json"

    report = run_search(labeled_splits, report_path, search, evaluation_params, tracker)

    assert len(report["trials"]) == len(tracker.runs) == 2
    best = max(report["trials"], key=lambda trial: trial["metrics"]["ndcg_at_2"])
    assert report["best"] == best
    assert report["select_metric"] == "ndcg_at_2"
    corrected = report["best_score_optimism_corrected"]
    assert corrected == pytest.approx(best["metrics"]["ndcg_at_2"] - report["selection_optimism"])
    assert all("fit_seconds" not in trial["metrics"] for trial in report["trials"])
    assert all("fit_seconds" in run["metrics"] for run in tracker.runs)
    assert json.loads(report_path.read_text()) == report


class _FakeSweepRun:
    """Stands in for the run `wandb agent` hands to a sweep trial."""

    def __init__(self) -> None:
        self.config = dict(SWEEP_CONFIG, factors=2, iterations=20, seed=0)
        self.logged: dict = {}

    def __enter__(self) -> "_FakeSweepRun":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def log(self, metrics: dict) -> None:
        self.logged |= metrics


@pytest.mark.integration
def test_sweep_trial_fits_the_sampled_config_and_logs_its_metrics(
    labeled_splits, evaluation_params
):
    run = _FakeSweepRun()

    metrics = sweep_trial(labeled_splits, evaluation_params, init=lambda: run)

    assert run.logged == metrics
    assert {"ndcg_at_2", "fit_seconds"} <= set(metrics)

import json
import subprocess
import sys

import polars as pl
import pytest

from recsys.experiments import reranker
from recsys.experiments.reranker import RerankerPaths, run_reranker
from recsys.reranking.features import FEATURE_COLUMNS
from recsys.reranking.ranker import load_ranker
from recsys.tracking import InMemoryTracker


class SimulatedCrashError(RuntimeError):
    pass


@pytest.fixture
def paths(reranker_world) -> RerankerPaths:
    return reranker_world


@pytest.fixture
def run(paths, evaluation_params, toy_world):
    def run_stage(tracker=None):
        return run_reranker(
            paths,
            toy_world.models,
            toy_world.reranker_params,
            evaluation_params,
            toy_world.cutoff,
            tracker or InMemoryTracker(),
        )

    return run_stage


@pytest.mark.integration
def test_reranker_trains_on_validation_and_evaluates_the_two_stage_pipeline(paths, run, toy_world):
    tracker = InMemoryTracker()

    report = run(tracker)

    assert set(report["models"]) == {"two_stage", "two_stage_all_candidates", "two_tower", "als"}
    assert set(report["paired"]) == {"two_tower", "als"}
    assert 0.0 <= report["retrieval"]["recall_at_n"] <= 1.0
    assert len(report["ranker"]["trials"]) == 2
    assert report["ranker"]["best"]["holdout_score"] == max(
        trial["holdout_score"] for trial in report["ranker"]["trials"]
    )
    assert set(report["feature_importance"]) == set(FEATURE_COLUMNS)
    assert {"mean_abs_shap", "mean_abs_within_user_shap", "gain"} == set(
        report["feature_importance"]["two_tower_score"]
    )
    users = report["ranker"]["users"]
    assert users["evaluation_users_excluded"] == len(toy_world.evaluation_users)
    assert users["fit"] + users["holdout"] + users["report"] == len(toy_world.valid_users)
    # The same comparison on held-out validation users, i.e. data like the ranker's own.
    in_distribution = report["in_distribution"]
    assert set(in_distribution["models"]) == {"reranker", "two_tower"}
    assert in_distribution["users"] == users["report"]  # an untouched slice, not the holdout
    difference = in_distribution["paired"]["differences"]["reranker"]
    assert difference["ci_low"] <= difference["mean"] <= difference["ci_high"]
    # Upper bound: rankers cross-fitted on the fully observed tune users themselves.
    cross_fitted = report["cross_fitted"]
    assert set(cross_fitted["models"]) == {"two_stage_cross_fitted", "two_tower"}
    assert "two_stage_cross_fitted" in cross_fitted["paired"]["differences"]
    assert [run["name"] for run in tracker.runs][-1] == "reranker/two_stage"
    assert json.loads(paths.metrics_path.read_text()) == report


@pytest.mark.integration
def test_reranker_saves_its_model_and_a_shap_sample_and_cleans_up(paths, run, toy_world):
    run()

    booster = load_ranker(paths.model_dir / "model.txt")
    manifest = json.loads((paths.model_dir / "manifest.json").read_text())
    shap = pl.read_parquet(paths.shap_path)

    assert booster.feature_name() == list(FEATURE_COLUMNS)
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert set(manifest["categorical_codes"]) == {"first_category", "user_active_degree"}
    assert manifest["cutoff"] == toy_world.cutoff.isoformat()
    assert shap.height <= toy_world.reranker_params.shap_sample_rows
    assert {f"shap_{column}" for column in FEATURE_COLUMNS} <= set(shap.columns)
    assert {f"within_user_shap_{column}" for column in FEATURE_COLUMNS} <= set(shap.columns)
    assert not paths.checkpoint_dir.exists()


@pytest.mark.integration
def test_an_interrupted_grid_resumes_without_retraining_finished_configurations(run, monkeypatch):
    real_train = reranker.train_ranker
    calls = {"count": 0, "crashed": False}

    def train_then_crash_once_on_the_second_configuration(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2 and not calls["crashed"]:
            calls["crashed"] = True
            raise SimulatedCrashError("killed while training the second configuration")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(reranker, "train_ranker", train_then_crash_once_on_the_second_configuration)
    with pytest.raises(SimulatedCrashError):
        run()
    calls["count"] = 0
    report = run()

    assert calls["count"] == 1  # only the configuration that crashed is trained again
    assert len(report["ranker"]["trials"]) == 2


def test_the_reranker_stage_never_imports_pytorch():
    # PyTorch's OpenMP runtime would clash with LightGBM's on macOS (see ranker.openmp_threads).
    probe = "import sys, recsys.experiments.reranker; print('torch' in sys.modules)"

    result = subprocess.run(  # noqa: S603 (fixed command, no user input)
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"

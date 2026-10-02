import json
import subprocess
import sys
from datetime import date

import polars as pl
import pytest

from recsys.config import (
    ALSParams,
    CategoryAffinityParams,
    EngagementPopularityParams,
    ModelParams,
    RandomParams,
    RerankerParams,
    TwoTowerEnsembleParams,
)
from recsys.experiments import reranker
from recsys.experiments.reranker import RerankerPaths, run_reranker
from recsys.reranking.features import FEATURE_COLUMNS
from recsys.reranking.ranker import load_ranker
from recsys.tracking import InMemoryTracker

CUTOFF = date(2020, 8, 30)
VALID_USERS = range(6, 46)
EVALUATION_USERS = (0, 3, 100)  # tune users 0 and 3, test user 100: never used to train
VIDEOS = range(10, 16)
MODELS = ModelParams(
    random=RandomParams(seed=0),
    popularity_engagement=EngagementPopularityParams(prior_strength=1.0),
    category_affinity=CategoryAffinityParams(prior_strength=1.0),
    als=ALSParams(factors=2, regularization=0.01, alpha=10.0, iterations=10, seed=0),
    two_tower=TwoTowerEnsembleParams(  # unused here: two-tower scores come precomputed
        loss="bce",
        embedding_dim=4,
        hidden_dim=8,
        epochs=1,
        batch_size=8,
        learning_rate=0.05,
        weight_decay=0.0,
        seeds=(0,),
        device="cpu",
    ),
)
PARAMS = RerankerParams(
    retrieve_top_n=3,
    holdout_user_share=0.25,
    report_user_share=0.25,
    holdout_salt="test-holdout",
    cross_fit_folds=2,
    num_leaves=(3,),
    min_child_samples=(1, 3),
    learning_rate=0.3,
    feature_fraction=1.0,
    max_rounds=15,
    early_stopping_rounds=5,
    eval_at=2,
    prior_strength=2.0,
    shap_sample_rows=5,
    seed=0,
)


class SimulatedCrashError(RuntimeError):
    pass


def valid_likes(user: int, video: int) -> bool:
    return (user % 2 == 0) == (video < 13)


@pytest.fixture
def paths(tmp_path, labeled_splits, feature_tables, make_interactions) -> RerankerPaths:
    """Validation week, platform stats and precomputed two-tower scores for the toy splits."""
    pairs = [(u, v) for u in (*VALID_USERS, *EVALUATION_USERS) for v in VIDEOS]
    users, videos = zip(*pairs, strict=True)
    make_interactions(user_id=list(users), video_id=list(videos)).with_columns(
        duration_bucket=pl.lit(0, dtype=pl.Int64),
        is_positive=pl.Series([valid_likes(u, v) for u, v in pairs]),
    ).write_parquet(labeled_splits / "valid.parquet")
    tune = pl.read_parquet(labeled_splits / "tune.parquet")
    tune.with_columns(user_id=pl.lit(100, pl.Int64)).unique(["user_id", "video_id"]).write_parquet(
        labeled_splits / "test.parquet"
    )

    processed = tmp_path / "processed"
    processed.mkdir()
    pl.DataFrame(
        {
            "video_id": list(VIDEOS),
            "date": [date(2020, 8, 1)] * 6,
            "upload_dt": ["2020-07-01"] * 6,
            **{
                count: [100 * (v - 9) for v in VIDEOS]
                for count in ("play_cnt", "like_cnt", "complete_play_cnt")
            },
            **{count: [1] * 6 for count in ("share_cnt", "comment_cnt", "follow_cnt")},
        }
    ).write_parquet(processed / "video_daily_stats.parquet")

    tune = pl.read_parquet(labeled_splits / "tune.parquet", columns=["user_id", "video_id"])
    needed = pl.concat([pl.DataFrame({"user_id": users, "video_id": videos}), tune]).unique()
    retrieval = tmp_path / "retrieval.parquet"
    needed.with_columns(score=(pl.col("video_id") % 3).cast(pl.Float64)).write_parquet(retrieval)

    return RerankerPaths(
        splits_dir=labeled_splits,
        features_dir=feature_tables,
        processed_dir=processed,
        retrieval_path=retrieval,
        metrics_path=tmp_path / "reranker.json",
        model_dir=tmp_path / "models" / "reranker",
        checkpoint_dir=tmp_path / "checkpoints" / "reranker",
        shap_path=tmp_path / "shap" / "shap_sample.parquet",
    )


def run(paths, evaluation_params, tracker=None):
    return run_reranker(
        paths, MODELS, PARAMS, evaluation_params, CUTOFF, tracker or InMemoryTracker()
    )


@pytest.mark.integration
def test_reranker_trains_on_validation_and_evaluates_the_two_stage_pipeline(
    paths, evaluation_params
):
    tracker = InMemoryTracker()

    report = run(paths, evaluation_params, tracker)

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
    assert users["evaluation_users_excluded"] == len(EVALUATION_USERS)
    assert users["fit"] + users["holdout"] + users["report"] == len(VALID_USERS)
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
def test_reranker_saves_its_model_and_a_shap_sample_and_cleans_up(paths, evaluation_params):
    run(paths, evaluation_params)

    booster = load_ranker(paths.model_dir / "model.txt")
    manifest = json.loads((paths.model_dir / "manifest.json").read_text())
    shap = pl.read_parquet(paths.shap_path)

    assert booster.feature_name() == list(FEATURE_COLUMNS)
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert set(manifest["categorical_codes"]) == {"first_category", "user_active_degree"}
    assert manifest["cutoff"] == CUTOFF.isoformat()
    assert shap.height <= PARAMS.shap_sample_rows
    assert {f"shap_{column}" for column in FEATURE_COLUMNS} <= set(shap.columns)
    assert {f"within_user_shap_{column}" for column in FEATURE_COLUMNS} <= set(shap.columns)
    assert not paths.checkpoint_dir.exists()


@pytest.mark.integration
def test_an_interrupted_grid_resumes_without_retraining_finished_configurations(
    paths, evaluation_params, monkeypatch
):
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
        run(paths, evaluation_params)
    calls["count"] = 0
    report = run(paths, evaluation_params)

    assert calls["count"] == 1  # only the configuration that crashed is trained again
    assert len(report["ranker"]["trials"]) == 2


def test_the_reranker_stage_never_imports_pytorch():
    # PyTorch's OpenMP runtime would clash with LightGBM's on macOS (see ranker.openmp_threads).
    probe = "import sys, recsys.experiments.reranker; print('torch' in sys.modules)"

    result = subprocess.run(  # noqa: S603 (fixed command, no user input)
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"

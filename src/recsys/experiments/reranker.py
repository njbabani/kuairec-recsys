"""Stage two: a LightGBM LambdaRank re-ranker over the two-tower model's shortlist.

Run as a DVC stage: ``python -m recsys.experiments.reranker``. This process never loads
PyTorch: two-tower scores come precomputed from the ``retrieval`` stage.

1. Fit the models whose scores become features (ALS, popularity, category affinity) on train.
2. Learn from the validation week (logged views from the cutoff on, one per pair, shaped like
   the evaluation) of users outside the fully observed matrix, with every feature computed from
   data before the cutoff. Those users are split three ways: rows to fit on, held-out users that
   stop boosting and pick the configuration, and users kept only for reporting.
3. Evaluate the two-stage pipeline once on the tune users, paired against two-tower retrieval
   alone and ALS; then the analyses in :mod:`recsys.experiments.reranker_analysis`.

Checkpointing: every finished grid configuration (holdout score and model file) is recorded, so
an interrupted run resumes without retraining it. The chosen model and a manifest describing
how to feed it go to ``models/reranker``.
"""

import itertools
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Self

import polars as pl

from recsys.checkpointing import ProgressLog, fingerprint, remove_if_empty
from recsys.config import (
    DataParams,
    EvaluationParams,
    ModelParams,
    RerankerParams,
    load_params,
)
from recsys.data.split import split_users
from recsys.evaluation.ranking import EvaluationResult, popularity_percentiles
from recsys.experiments.reranker_analysis import (
    cross_fitted_check,
    explain,
    in_distribution_check,
    shortlist,
)
from recsys.experiments.runner import (
    EVAL_SPLIT,
    MODEL_COLUMNS,
    compare_with_reference,
    evaluate_model,
    load_features,
    load_split,
)
from recsys.io import PARQUET_COMPRESSION, write_json
from recsys.models.als import ALSRecommender
from recsys.models.category_affinity import CategoryAffinityRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.reranking.candidates import PrecomputedScorer, ranker_training_rows
from recsys.reranking.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FeatureSources,
    RankingFeatures,
    StageOneScorers,
)
from recsys.reranking.ranker import (
    RankerConfig,
    RankingData,
    load_ranker,
    ranking_data,
    save_ranker,
    train_ranker,
)
from recsys.reranking.two_stage import (
    LightGBMReranker,
    RetrievalRecall,
    TwoStageRecommender,
    retrieval_recall,
)
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

PROGRESS_FILE = "progress.json"
MODEL_FILE = "model.txt"
MANIFEST_FILE = "manifest.json"
ALL_CANDIDATES = 2**31 - 1  # a shortlist size that keeps every candidate (re-rank everything)
EVALUATION_SPLITS = (EVAL_SPLIT, "test")
DAILY_STATS_COLUMNS = (
    "video_id",
    "date",
    "upload_dt",
    "play_cnt",
    "like_cnt",
    "complete_play_cnt",
    "share_cnt",
    "comment_cnt",
    "follow_cnt",
)


@dataclass(frozen=True)
class InputPaths:
    """Where the inputs of stage two live (shared with the final test evaluation)."""

    splits_dir: Path
    features_dir: Path
    processed_dir: Path
    retrieval_path: Path

    @classmethod
    def from_params(cls, data: DataParams) -> Self:
        return cls(data.splits_dir, data.features_dir, data.processed_dir, data.retrieval_path)


@dataclass(frozen=True)
class RerankerPaths:
    inputs: InputPaths
    metrics_path: Path
    model_dir: Path
    checkpoint_dir: Path
    shap_path: Path


@dataclass(frozen=True)
class Inputs:
    train: pl.DataFrame  # MODEL_COLUMNS + duration_bucket
    valid: pl.DataFrame  # MODEL_COLUMNS + event_time
    tune: pl.DataFrame
    popularity: pl.DataFrame
    user_features: pl.DataFrame
    video_features: pl.DataFrame
    daily_stats: pl.DataFrame
    retrieval: PrecomputedScorer
    evaluation_users: pl.DataFrame  # user_id of every tune and test user


def load_inputs(paths: InputPaths) -> Inputs:
    train = pl.read_parquet(
        paths.splits_dir / "train.parquet", columns=[*MODEL_COLUMNS, "duration_bucket"]
    )
    tune = load_split(paths.splits_dir, EVAL_SPLIT)
    users, videos = load_features(paths.features_dir)
    evaluation_users = pl.concat(
        [
            pl.read_parquet(paths.splits_dir / f"{name}.parquet", columns=["user_id"])
            for name in EVALUATION_SPLITS
        ]
    ).unique()
    return Inputs(
        train=train,
        valid=pl.read_parquet(
            paths.splits_dir / "valid.parquet", columns=[*MODEL_COLUMNS, "event_time"]
        ),
        tune=tune,
        popularity=popularity_percentiles(train, tune["video_id"]),
        user_features=users,
        video_features=videos,
        daily_stats=pl.read_parquet(
            paths.processed_dir / "video_daily_stats.parquet", columns=list(DAILY_STATS_COLUMNS)
        ),
        retrieval=PrecomputedScorer(pl.read_parquet(paths.retrieval_path), name="two_tower"),
        evaluation_users=evaluation_users,
    )


def fit_stage_one(inputs: Inputs, models: ModelParams) -> StageOneScorers:
    """The other recommenders, fitted on train, whose scores become ranking features."""
    train = inputs.train.select(MODEL_COLUMNS)
    prior = models.popularity_engagement.prior_strength
    return StageOneScorers(
        two_tower=inputs.retrieval,
        als=ALSRecommender(**models.als.model_dump()).fit(train),
        engagement=PopularityRecommender(by="engagement", prior_strength=prior).fit(train),
        category_affinity=CategoryAffinityRecommender(
            inputs.video_features,
            prior_strength=models.category_affinity.prior_strength,
            base=PopularityRecommender(by="engagement", prior_strength=prior),
        ).fit(train),
    )


@dataclass(frozen=True)
class RankerRows:
    """Validation views for the ranker, split by user into three disjoint groups."""

    fit: pl.DataFrame  # rows the ranker learns from
    holdout: pl.DataFrame  # users that stop boosting and pick the configuration
    report: pl.DataFrame  # users used only to report the logged-data comparison
    evaluation_users_excluded: int

    def users(self) -> dict[str, int]:
        return {
            "fit": self.fit["user_id"].n_unique(),
            "holdout": self.holdout["user_id"].n_unique(),
            "report": self.report["user_id"].n_unique(),
            "evaluation_users_excluded": self.evaluation_users_excluded,
        }


def split_ranker_rows(inputs: Inputs, params: RerankerParams) -> RankerRows:
    """The ranker never sees a tune or test user, not even their logged views."""
    week = inputs.valid
    excluded = week.join(inputs.evaluation_users, on="user_id", how="semi")["user_id"].n_unique()
    week = week.join(inputs.evaluation_users, on="user_id", how="anti")
    rows = ranker_training_rows(week, inputs.train)
    report, rest = split_users(
        rows, tune_share=params.report_user_share, salt=f"{params.holdout_salt}:report"
    )
    holdout, fit = split_users(
        rest,
        tune_share=params.holdout_user_share / (1 - params.report_user_share),
        salt=params.holdout_salt,
    )
    logger.info("Ranker users: %s", RankerRows(fit, holdout, report, excluded).users())
    return RankerRows(fit, holdout, report, excluded)


def _as_ranking_data(features: RankingFeatures, rows: pl.DataFrame) -> RankingData:
    return ranking_data(features.transform(rows), rows["is_positive"].cast(pl.Int32))


def ranker_grid(params: RerankerParams) -> list[RankerConfig]:
    return [
        RankerConfig(
            num_leaves=leaves,
            min_child_samples=min_child,
            learning_rate=params.learning_rate,
            feature_fraction=params.feature_fraction,
            max_rounds=params.max_rounds,
            early_stopping_rounds=params.early_stopping_rounds,
            eval_at=params.eval_at,
            seed=params.seed,
        )
        for leaves, min_child in itertools.product(params.num_leaves, params.min_child_samples)
    ]


def _config_key(config: RankerConfig) -> str:
    return f"leaves_{config.num_leaves}_min_child_{config.min_child_samples}"


def _train_configuration(
    config: RankerConfig,
    fit: RankingData,
    holdout: RankingData,
    model_path: Path,
    tracker: Tracker,
) -> dict[str, Any]:
    with tracker.run(f"reranker/{_config_key(config)}", {"reranker": asdict(config)}) as run:
        trained = train_ranker(fit, holdout, config)
        metric = f"holdout_ndcg_at_{config.eval_at}"
        for step, score in enumerate(trained.curve, start=1):
            run.log_metrics({metric: score}, step=step)
        run.log_metrics(
            {"holdout_score": trained.holdout_score, "best_iteration": trained.best_iteration}
        )
    save_ranker(trained.booster, model_path)
    logger.info("%s: holdout %.4f", _config_key(config), trained.holdout_score)
    return {
        "config": asdict(config),
        "holdout_score": trained.holdout_score,
        "best_iteration": trained.best_iteration,
        "curve": trained.curve,
    }


def search_ranker(
    fit: RankingData,
    holdout: RankingData,
    params: RerankerParams,
    checkpoint_dir: Path,
    tracker: Tracker,
) -> list[dict[str, Any]]:
    """Train every grid configuration, reusing those an interrupted run already finished.

    Progress is tied to the training data (rows, labels and user groups); each finished
    configuration is reused only while its own settings still match.
    """
    frames = [
        fit.features,
        holdout.features,
        *(
            pl.DataFrame({name: values})
            for name, values in (
                ("fit_labels", fit.labels),
                ("fit_groups", fit.groups),
                ("holdout_labels", holdout.labels),
                ("holdout_groups", holdout.groups),
            )
        ),
    ]
    progress = ProgressLog.open(checkpoint_dir / PROGRESS_FILE, fingerprint({}, frames))
    keys = []
    for config in ranker_grid(params):
        key = _config_key(config)
        keys.append(key)
        model_path = checkpoint_dir / f"{key}.txt"
        entry = progress.entries.get(key)
        if entry is None or entry["config"] != asdict(config) or not model_path.is_file():
            progress.record(key, _train_configuration(config, fit, holdout, model_path, tracker))
    return [{"key": key, **progress.entries[key]} for key in keys]


def evaluate_two_stage(
    inputs: Inputs,
    stage_one: StageOneScorers,
    reranker: LightGBMReranker,
    params: RerankerParams,
    evaluation: EvaluationParams,
) -> dict[str, EvaluationResult]:
    """The pipeline, the same re-ranker over every candidate (no shortlist), and the two
    models it is compared with, on the tune users."""
    scorers = {
        "two_stage": TwoStageRecommender(inputs.retrieval, reranker, params.retrieve_top_n),
        "two_stage_all_candidates": TwoStageRecommender(inputs.retrieval, reranker, ALL_CANDIDATES),
        "two_tower": inputs.retrieval,
        "als": stage_one.als,
    }
    return {
        name: evaluate_model(scorer, inputs.tune, inputs.popularity, evaluation)
        for name, scorer in scorers.items()
    }


def write_manifest(
    path: Path,
    features: RankingFeatures,
    best: dict[str, Any],
    params: RerankerParams,
    cutoff: date,
) -> None:
    """Everything needed to feed the saved model, besides the source tables themselves."""
    codes = {
        column: {str(value): code for value, code in mapping.items()}
        for column, mapping in features.categorical_codes().items()
    }
    write_json(
        {
            "feature_columns": list(FEATURE_COLUMNS),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "categorical_codes": codes,
            "cutoff": cutoff.isoformat(),
            "prior_strength": params.prior_strength,
            "retrieve_top_n": params.retrieve_top_n,
            "config": best["config"],
            "best_iteration": best["best_iteration"],
            "trained_on": "validation-week views of users outside the fully observed matrix",
        },
        path,
    )


def _remove_resume_state(checkpoint_dir: Path, trials: list[dict[str, Any]]) -> None:
    (checkpoint_dir / PROGRESS_FILE).unlink(missing_ok=True)
    for trial in trials:
        (checkpoint_dir / f"{trial['key']}.txt").unlink(missing_ok=True)
    remove_if_empty(checkpoint_dir)


def _report(
    ranker: dict[str, Any],
    results: dict[str, EvaluationResult],
    recall: RetrievalRecall,
    analyses: dict[str, Any],
    params: RerankerParams,
    evaluation: EvaluationParams,
) -> dict[str, Any]:
    return {
        "retrieval": {
            "top_n": params.retrieve_top_n,
            "recall_at_n": recall.recall,
            "recall_ceiling": recall.ceiling,
        },
        "ranker": ranker,
        "models": {name: result.summary for name, result in results.items()},
        "paired": {
            reference: compare_with_reference(results, reference, evaluation)
            for reference in ("two_tower", "als")
        },
        **analyses,
    }


def _analyses(
    inputs: Inputs,
    rows: RankerRows,
    reranker: LightGBMReranker,
    best: dict[str, Any],
    params: RerankerParams,
    evaluation: EvaluationParams,
) -> dict[str, Any]:
    return {
        "in_distribution": in_distribution_check(
            rows.report, inputs.train, reranker, inputs.retrieval, evaluation
        ),
        "cross_fitted": cross_fitted_check(
            inputs.tune,
            inputs.retrieval,
            reranker.features,
            RankerConfig(**best["config"]),
            rounds=best["best_iteration"],
            folds=params.cross_fit_folds,
            top_n=params.retrieve_top_n,
            evaluation=evaluation,
            popularity=inputs.popularity,
        ),
    }


def run_reranker(
    paths: RerankerPaths,
    models: ModelParams,
    params: RerankerParams,
    evaluation: EvaluationParams,
    cutoff: date,
    tracker: Tracker,
) -> dict[str, Any]:
    inputs = load_inputs(paths.inputs)
    stage_one = fit_stage_one(inputs, models)
    sources = FeatureSources(
        inputs.train, inputs.user_features, inputs.video_features, inputs.daily_stats, cutoff
    )
    features = RankingFeatures(stage_one, params.prior_strength).fit(sources)
    rows = split_ranker_rows(inputs, params)
    fit, holdout = _as_ranking_data(features, rows.fit), _as_ranking_data(features, rows.holdout)
    trials = search_ranker(fit, holdout, params, paths.checkpoint_dir, tracker)
    best = max(trials, key=lambda trial: trial["holdout_score"])
    booster = load_ranker(paths.checkpoint_dir / f"{best['key']}.txt")
    save_ranker(booster, paths.model_dir / MODEL_FILE)
    write_manifest(paths.model_dir / MANIFEST_FILE, features, best, params, cutoff)
    reranker = LightGBMReranker(features, booster)

    with tracker.run("reranker/two_stage", {"reranker": params.model_dump()}) as run:
        results = evaluate_two_stage(inputs, stage_one, reranker, params, evaluation)
        recall = retrieval_recall(inputs.tune, inputs.retrieval, params.retrieve_top_n)
        run.log_metrics({**results["two_stage"].summary, "retrieval_recall": recall.recall})
    analyses = _analyses(inputs, rows, reranker, best, params, evaluation)
    importance, sample = explain(
        shortlist(inputs.tune, inputs.retrieval, params.retrieve_top_n),
        reranker,
        params.shap_sample_rows,
        params.seed,
    )
    paths.shap_path.parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(paths.shap_path, compression=PARQUET_COMPRESSION)

    ranker = {
        "selected_on": f"ndcg@{params.eval_at} of held-out validation users",
        "users": rows.users(),
        "best": best,
        "trials": trials,
    }
    report = _report(
        ranker, results, recall, {**analyses, "feature_importance": importance}, params, evaluation
    )
    write_json(report, paths.metrics_path)
    _remove_resume_state(paths.checkpoint_dir, trials)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    data = params.data
    paths = RerankerPaths(
        inputs=InputPaths.from_params(data),
        metrics_path=data.metrics_dir / "reranker.json",
        model_dir=data.models_dir / "reranker",
        checkpoint_dir=data.checkpoints_dir / "reranker",
        shap_path=data.reranker_dir / "shap_sample.parquet",
    )
    run_reranker(
        paths,
        params.models,
        params.reranker,
        params.evaluation,
        params.split.valid_start,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

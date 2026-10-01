"""Fit every model on train, evaluate it on the tune users, and compare them on the same users.

Run as a DVC stage: ``python -m recsys.experiments.leaderboard``. Searched models use the
configuration copied into ``models`` in params.yaml from their search stage's report.

The two-tower entry is an ensemble of one model per seed: a single run varies by about
+-0.006 NDCG@10, so the report also gives the members' spread. The trained members are saved to
``models/two_tower/`` for later stages, and their training is checkpointed so an interrupted run
resumes instead of starting over.
"""

import logging
import statistics
from pathlib import Path
from typing import Any

import polars as pl

from recsys.config import EvaluationParams, ModelParams, load_params
from recsys.evaluation.ranking import (
    EvaluationResult,
    paired_bootstrap_difference,
    paired_win_rate,
)
from recsys.experiments.runner import (
    EVAL_SPLIT,
    ExperimentData,
    evaluate_model,
    fit_and_evaluate,
    load_experiment_data,
    load_features,
)
from recsys.io import write_json
from recsys.models.als import ALSRecommender
from recsys.models.base import Recommender
from recsys.models.category_affinity import CategoryAffinityRecommender
from recsys.models.popularity import PopularityRecommender, RandomRecommender
from recsys.models.two_tower.checkpoint import remove_if_empty
from recsys.models.two_tower.ensemble import TwoTowerEnsemble
from recsys.models.two_tower.recommender import TwoTowerRecommender
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

# Every model is compared on the same users with the strongest non-personalised model (the bar
# any personalisation has to clear) and with ALS (the strongest Phase 2 model).
REFERENCE_MODELS = ("popularity_engagement", "als")


def build_models(
    params: ModelParams,
    features: tuple[pl.DataFrame, pl.DataFrame],
    checkpoint_dir: Path | None = None,
) -> list[tuple[Recommender, dict[str, Any]]]:
    """Each model with the hyperparameters to record alongside it."""
    _, video_features = features

    def engagement_popularity() -> PopularityRecommender:
        return PopularityRecommender(
            by="engagement", prior_strength=params.popularity_engagement.prior_strength
        )

    category_affinity = CategoryAffinityRecommender(
        video_features,
        prior_strength=params.category_affinity.prior_strength,
        base=engagement_popularity(),
    )
    two_tower = TwoTowerEnsemble(
        [
            TwoTowerRecommender(*features, **params.two_tower.member(seed).model_dump())
            for seed in params.two_tower.seeds
        ],
        checkpoint_dir=None if checkpoint_dir is None else checkpoint_dir / "two_tower",
    )
    return [
        (RandomRecommender(params.random.seed), params.random.model_dump()),
        (PopularityRecommender(by="views"), {}),
        (engagement_popularity(), params.popularity_engagement.model_dump()),
        (
            category_affinity,
            {
                **params.category_affinity.model_dump(),
                "base_prior_strength": params.popularity_engagement.prior_strength,
            },
        ),
        (ALSRecommender(**params.als.model_dump()), params.als.model_dump()),
        (two_tower, params.two_tower.model_dump()),
    ]


def seed_spread(
    ensemble: TwoTowerEnsemble, data: ExperimentData, evaluation: EvaluationParams
) -> dict[str, Any]:
    """How much the ensemble's members, each a single training run, differ from one another."""
    per_seed = [
        evaluate_model(member, data.eval_split, data.popularity, evaluation).summary[
            evaluation.select_metric
        ]
        for member in ensemble.members
    ]
    return {
        "seeds": [member.seed for member in ensemble.members],
        evaluation.select_metric: per_seed,
        "mean": statistics.fmean(per_seed),
        "sd": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
    }


def run_leaderboard(
    splits_dir: Path,
    features_dir: Path,
    metrics_path: Path,
    params: ModelParams,
    evaluation: EvaluationParams,
    tracker: Tracker,
    *,
    models_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    data = load_experiment_data(splits_dir)
    features = load_features(features_dir)

    results: dict[str, EvaluationResult] = {}
    spreads: dict[str, dict[str, Any]] = {}
    for model, model_params in build_models(params, features, checkpoint_dir):
        run_params = {
            "model": model.name,
            "eval_split": EVAL_SPLIT,
            model.name: model_params,
            "evaluation": evaluation.model_dump(),
        }
        with tracker.run(f"leaderboard/{model.name}", run_params) as run:
            result, fit_seconds = fit_and_evaluate(
                model, data.train, data.eval_split, data.popularity, evaluation
            )
            run.log_metrics({**result.summary, "fit_seconds": fit_seconds})
            if isinstance(model, TwoTowerEnsemble):
                spreads[model.name] = seed_spread(model, data, evaluation)
                run.log_metrics({"member_mean": spreads[model.name]["mean"]})
                run.log_metrics({"member_sd": spreads[model.name]["sd"]})
                model.save(models_dir / model.name)
        results[model.name] = result
        logger.info(
            "%-22s %s=%.4f",
            model.name,
            evaluation.select_metric,
            result.summary[evaluation.select_metric],
        )

    remove_if_empty(checkpoint_dir)
    report = {
        "models": {name: result.summary for name, result in results.items()},
        "paired": {
            reference: compare_with_reference(results, reference, evaluation)
            for reference in REFERENCE_MODELS
        },
        "seed_spread": spreads,
    }
    write_json(report, metrics_path)
    return report


def compare_with_reference(
    results: dict[str, EvaluationResult], reference: str, evaluation: EvaluationParams
) -> dict[str, Any]:
    """Per-user paired difference (and win rate) of every other model against ``reference``.

    The interval reflects user sampling only: it is conditional on the candidate videos, the
    label cut-offs and each model's fixed seed(s).
    """
    baseline = results[reference].per_user
    differences = {}
    for name, result in results.items():
        if name == reference:
            continue
        mean, low, high = paired_bootstrap_difference(
            result.per_user,
            baseline,
            evaluation.select_metric,
            n_samples=evaluation.bootstrap_samples,
            seed=evaluation.seed,
        )
        differences[name] = {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "win_rate": paired_win_rate(result.per_user, baseline, evaluation.select_metric),
        }
    return {"metric": evaluation.select_metric, "differences": differences}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    run_leaderboard(
        params.data.splits_dir,
        params.data.features_dir,
        params.data.metrics_dir / "leaderboard.json",
        params.models,
        params.evaluation,
        build_tracker(params.tracking, workdir=Path.cwd()),
        models_dir=params.data.models_dir,
        checkpoint_dir=params.data.checkpoints_dir / "leaderboard",
    )


if __name__ == "__main__":
    main()

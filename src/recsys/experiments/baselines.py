"""Fit every baseline on train, evaluate it on the tune users, and track each run.

Run as a DVC stage: ``python -m recsys.experiments.baselines``.
"""

import logging
from pathlib import Path
from typing import Any

from recsys.config import BaselineParams, EvaluationParams, load_params
from recsys.evaluation.ranking import (
    EvaluationResult,
    paired_bootstrap_difference,
    paired_win_rate,
    popularity_percentiles,
)
from recsys.experiments.runner import EVAL_SPLIT, fit_and_evaluate, load_split
from recsys.io import write_json
from recsys.models.als import ALSRecommender
from recsys.models.base import Recommender
from recsys.models.popularity import PopularityRecommender, RandomRecommender
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

# The strongest non-personalized baseline: any personalized model has to beat this.
REFERENCE_MODEL = "popularity_engagement"


def build_baselines(params: BaselineParams) -> list[tuple[Recommender, dict[str, Any]]]:
    """Each baseline with the hyperparameters to record alongside it."""
    return [
        (RandomRecommender(params.random.seed), params.random.model_dump()),
        (PopularityRecommender(by="views"), {}),
        (
            PopularityRecommender(
                by="engagement", prior_strength=params.popularity_engagement.prior_strength
            ),
            params.popularity_engagement.model_dump(),
        ),
        (ALSRecommender(**params.als.model_dump()), params.als.model_dump()),
    ]


def run_baselines(
    splits_dir: Path,
    metrics_path: Path,
    params: BaselineParams,
    evaluation: EvaluationParams,
    tracker: Tracker,
) -> dict[str, Any]:
    """Evaluate every baseline; also compare each one with the reference on the same users."""
    train, eval_split = load_split(splits_dir, "train"), load_split(splits_dir, EVAL_SPLIT)
    popularity = popularity_percentiles(train, eval_split["video_id"])

    results: dict[str, EvaluationResult] = {}
    for model, model_params in build_baselines(params):
        run_params = {
            "model": model.name,
            "eval_split": EVAL_SPLIT,
            model.name: model_params,
            "evaluation": evaluation.model_dump(),
        }
        with tracker.run(f"baseline/{model.name}", run_params) as run:
            result, fit_seconds = fit_and_evaluate(model, train, eval_split, popularity, evaluation)
            run.log_metrics({**result.summary, "fit_seconds": fit_seconds})
        results[model.name] = result
        logger.info(
            "%-22s %s=%.4f",
            model.name,
            evaluation.select_metric,
            result.summary[evaluation.select_metric],
        )

    report = {
        "models": {name: result.summary for name, result in results.items()},
        "paired_vs_reference": compare_with_reference(results, evaluation),
    }
    write_json(report, metrics_path)
    return report


def compare_with_reference(
    results: dict[str, EvaluationResult], evaluation: EvaluationParams
) -> dict[str, Any]:
    """Per-user paired difference (and win rate) of every model against ``REFERENCE_MODEL``.

    The interval reflects user sampling only: it is conditional on the candidate videos, the
    label cut-offs and each model's fixed seed.
    """
    reference = results[REFERENCE_MODEL].per_user
    differences = {}
    for name, result in results.items():
        if name == REFERENCE_MODEL:
            continue
        mean, low, high = paired_bootstrap_difference(
            result.per_user,
            reference,
            evaluation.select_metric,
            n_samples=evaluation.bootstrap_samples,
            seed=evaluation.seed,
        )
        differences[name] = {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "win_rate": paired_win_rate(result.per_user, reference, evaluation.select_metric),
        }
    return {
        "reference": REFERENCE_MODEL,
        "metric": evaluation.select_metric,
        "differences": differences,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    run_baselines(
        params.data.splits_dir,
        params.data.metrics_dir / "baselines.json",
        params.baselines,
        params.evaluation,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

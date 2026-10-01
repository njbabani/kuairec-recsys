"""ALS hyperparameter search, evaluated on the tune users (never on test).

Two ways to run the same trial:

* a local grid, tracked in MLflow and W&B: ``python -m recsys.experiments.als_search``
  (DVC stage ``als_search``);
* a W&B Bayesian sweep (needs ``wandb login``): ``make sweep-als`` creates it from
  ``sweeps/als.yaml``; each ``wandb agent`` trial runs this module with ``--sweep-trial``.

The best grid configuration is written to the report; copying it into ``baselines.als`` in
params.yaml is a deliberate, reviewable step.
"""

import argparse
import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import polars as pl

from recsys.config import ALSParams, ALSSearchParams, EvaluationParams, load_params
from recsys.evaluation.ranking import EvaluationResult, popularity_percentiles
from recsys.evaluation.selection import selection_optimism
from recsys.experiments.runner import EVAL_SPLIT, fit_and_evaluate, load_split
from recsys.io import write_json
from recsys.models.als import ALSRecommender
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)


def als_grid(search: ALSSearchParams) -> list[ALSParams]:
    """Every combination of the searched factors, regularization and alpha values."""
    return [
        ALSParams(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=search.iterations,
            seed=search.seed,
        )
        for factors, regularization, alpha in itertools.product(
            search.factors, search.regularization, search.alpha
        )
    ]


def params_from_sweep_config(config: Mapping[str, Any]) -> ALSParams:
    """Validate the hyperparameters a W&B sweep sampled for one trial."""
    return ALSParams.model_validate(dict(config))


def run_trial(
    als: ALSParams,
    train: pl.DataFrame,
    eval_split: pl.DataFrame,
    popularity: pl.DataFrame,
    evaluation: EvaluationParams,
) -> tuple[EvaluationResult, float]:
    return fit_and_evaluate(
        ALSRecommender(**als.model_dump()), train, eval_split, popularity, evaluation
    )


def _load_inputs(splits_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train, eval_split = load_split(splits_dir, "train"), load_split(splits_dir, EVAL_SPLIT)
    return train, eval_split, popularity_percentiles(train, eval_split["video_id"])


def run_search(
    splits_dir: Path,
    report_path: Path,
    search: ALSSearchParams,
    evaluation: EvaluationParams,
    tracker: Tracker,
) -> dict[str, Any]:
    train, eval_split, popularity = _load_inputs(splits_dir)
    trials = []
    per_user_scores: dict[str, pl.Series] = {}
    for number, als in enumerate(als_grid(search), start=1):
        run_params = {
            "model": "als",
            "eval_split": EVAL_SPLIT,
            "als": als.model_dump(),
            "evaluation": evaluation.model_dump(),
        }
        with tracker.run(f"als_search/{number:02d}", run_params) as run:
            result, fit_seconds = run_trial(als, train, eval_split, popularity, evaluation)
            run.log_metrics({**result.summary, "fit_seconds": fit_seconds})
        metrics = result.summary
        trials.append({"params": als.model_dump(), "metrics": metrics})
        per_user_scores[f"trial_{number:02d}"] = result.per_user[evaluation.select_metric]
        logger.info(
            "trial %02d %s %s=%.4f",
            number,
            als.model_dump(),
            evaluation.select_metric,
            metrics[evaluation.select_metric],
        )

    best = max(trials, key=lambda trial: trial["metrics"][evaluation.select_metric])
    # The winner was picked on these same users, so its score is optimistic (winner's curse).
    optimism = selection_optimism(
        pl.DataFrame(per_user_scores),
        n_samples=evaluation.bootstrap_samples,
        seed=evaluation.seed,
    )
    report = {
        "select_metric": evaluation.select_metric,
        "best": best,
        "selection_optimism": optimism,
        "best_score_optimism_corrected": best["metrics"][evaluation.select_metric] - optimism,
        "trials": trials,
    }
    write_json(report, report_path)
    return report


def sweep_trial(
    splits_dir: Path,
    evaluation: EvaluationParams,
    init: Callable[[], AbstractContextManager[Any]] | None = None,
) -> dict[str, float | int]:
    """One W&B sweep trial: W&B samples the hyperparameters, we fit, evaluate and log.

    ``init`` defaults to ``wandb.init``, which inside ``wandb agent`` returns the sweep's run.
    """
    if init is None:
        import wandb

        init = wandb.init
    train, eval_split, popularity = _load_inputs(splits_dir)
    with init() as run:
        als = params_from_sweep_config(run.config)
        result, fit_seconds = run_trial(als, train, eval_split, popularity, evaluation)
        metrics = {**result.summary, "fit_seconds": fit_seconds}
        run.log(metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ALS hyperparameter search on the tune users")
    parser.add_argument("--sweep-trial", action="store_true", help="run one W&B sweep trial")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    if args.sweep_trial:
        sweep_trial(params.data.splits_dir, params.evaluation)
        return
    run_search(
        params.data.splits_dir,
        params.data.metrics_dir / "als_search.json",
        params.als_search,
        params.evaluation,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

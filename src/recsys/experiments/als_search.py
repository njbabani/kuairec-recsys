"""ALS hyperparameter search, evaluated on the tune users (never on test).

Two ways to run the same trial:

* a local grid, tracked in MLflow and W&B: ``python -m recsys.experiments.als_search``
  (DVC stage ``als_search``);
* a W&B Bayesian sweep (needs ``wandb login``): ``make sweep-als`` creates it from
  ``sweeps/als.yaml``; each ``wandb agent`` trial runs this module with ``--sweep-trial``.

The best grid configuration is written to the report; copying it into ``models.als`` in
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
from recsys.evaluation.ranking import EvaluationResult
from recsys.evaluation.selection import summarise_search
from recsys.experiments.runner import EVAL_SPLIT, fit_and_evaluate, load_experiment_data
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


def run_search(
    splits_dir: Path,
    report_path: Path,
    search: ALSSearchParams,
    evaluation: EvaluationParams,
    tracker: Tracker,
) -> dict[str, Any]:
    data = load_experiment_data(splits_dir)
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
            result, fit_seconds = run_trial(
                als, data.train, data.eval_split, data.popularity, evaluation
            )
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

    # The winner was picked on these same users, so its score is optimistic (winner's curse).
    report = summarise_search(
        trials,
        per_user_scores,
        evaluation.select_metric,
        n_samples=evaluation.bootstrap_samples,
        seed=evaluation.seed,
    )
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
    data = load_experiment_data(splits_dir)
    with init() as run:
        als = params_from_sweep_config(run.config)
        result, fit_seconds = run_trial(
            als, data.train, data.eval_split, data.popularity, evaluation
        )
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

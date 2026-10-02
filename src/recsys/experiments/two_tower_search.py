"""Two-tower hyperparameter search, evaluated on the tune users (never on test).

Run as a DVC stage: ``python -m recsys.experiments.two_tower_search``.

* Every configuration is trained once per seed and scored on tune after every epoch, so the
  epoch count is searched without retraining. A single training run varies by about +-0.006
  NDCG@10 between seeds, so trials are compared on the seed average, not on one lucky run.
* Every (configuration, seed) is one tracked run whose per-epoch metrics form a training curve.
* Checkpointing: each scored epoch is appended to ``progress.json`` and each model's training
  state is checkpointed, so an interrupted search resumes where it stopped. Both are removed
  once the search finishes, and are ignored if the settings or data have changed.
* Picking the best of many trials on the same users flatters the winner, so the report also
  estimates that optimism. Copying the best configuration into ``models.two_tower`` in
  params.yaml is a deliberate, reviewable step.
"""

import itertools
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from recsys.checkpointing import ProgressLog, fingerprint, remove_if_empty
from recsys.config import (
    EvaluationParams,
    TwoTowerEnsembleParams,
    TwoTowerSearchParams,
    load_params,
)
from recsys.evaluation.ranking import EvaluationResult
from recsys.evaluation.selection import summarise_search
from recsys.experiments.runner import (
    EVAL_SPLIT,
    ExperimentData,
    evaluate_model,
    load_experiment_data,
    load_features,
)
from recsys.io import write_json
from recsys.models.two_tower.recommender import TwoTowerRecommender
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

PROGRESS_FILE = "progress.json"


def two_tower_grid(search: TwoTowerSearchParams) -> list[TwoTowerEnsembleParams]:
    """loss x embedding size, with every temperature for the softmax loss (bce has none)."""
    shared = search.model_dump(exclude={"loss", "embedding_dim", "temperature"})
    return [
        TwoTowerEnsembleParams(**shared, loss=loss, embedding_dim=dim, temperature=temperature)
        for loss, dim in itertools.product(search.loss, search.embedding_dim)
        for temperature in (search.temperature if loss == "softmax" else (None,))
    ]


def _scored_key(config: int, seed: int, epoch: int) -> str:
    return f"config_{config:02d}_seed_{seed}_epoch_{epoch:02d}"


def _checkpoint_name(config: int, seed: int) -> str:
    return f"config_{config:02d}_seed_{seed}.pt"


class SearchProgress(ProgressLog):
    """Every (configuration, seed, epoch) scored so far; saved to disk after each one."""

    def record_epoch(
        self,
        key: tuple[int, int, int],
        train_loss: float,
        result: EvaluationResult,
        select_metric: str,
    ) -> None:
        self.record(
            _scored_key(*key),
            {
                "train_loss": train_loss,
                "metrics": result.summary,
                "per_user": result.per_user[select_metric].to_list(),
            },
        )

    def epochs_scored(self, config: int, seed: int, epochs: int) -> int:
        """How many leading epochs of this (configuration, seed) are already scored."""
        done = 0
        while done < epochs and _scored_key(config, seed, done + 1) in self.entries:
            done += 1
        return done

    def get(self, config: int, seed: int, epoch: int) -> dict[str, Any]:
        return self.entries[_scored_key(config, seed, epoch)]


def _search_fingerprint(
    search: TwoTowerSearchParams,
    evaluation: EvaluationParams,
    data: ExperimentData,
    features: tuple[pl.DataFrame, pl.DataFrame],
) -> str:
    settings = [search.model_dump(mode="json"), evaluation.model_dump(mode="json")]
    return fingerprint(settings, [data.train, data.eval_split, *features])


def _train_and_score(
    number: int,
    config: TwoTowerEnsembleParams,
    seed: int,
    data: ExperimentData,
    features: tuple[pl.DataFrame, pl.DataFrame],
    evaluation: EvaluationParams,
    tracker: Tracker,
    progress: SearchProgress,
) -> None:
    """Train one (configuration, seed), scoring and recording it after every epoch."""
    params = config.member(seed)
    model = TwoTowerRecommender(*features, **params.model_dump())
    run_params = {
        "model": model.name,
        "eval_split": EVAL_SPLIT,
        "config": number,
        model.name: params.model_dump(),
        "evaluation": evaluation.model_dump(),
        "epochs_already_scored": progress.epochs_scored(number, seed, config.epochs),
    }
    with tracker.run(f"two_tower_search/{number:02d}/seed_{seed}", run_params) as run:

        def score_epoch(epoch: int, train_loss: float) -> None:
            result = evaluate_model(model, data.eval_split, data.popularity, evaluation)
            run.log_metrics({"train_loss": train_loss, **result.summary}, step=epoch)
            progress.record_epoch(
                (number, seed, epoch), train_loss, result, evaluation.select_metric
            )
            logger.info(
                "config %02d (%s, dim %d, t %s) seed %d epoch %d %s=%.4f",
                number,
                config.loss,
                config.embedding_dim,
                config.temperature,
                seed,
                epoch,
                evaluation.select_metric,
                result.summary[evaluation.select_metric],
            )

        checkpoint = progress.path.parent / _checkpoint_name(number, seed)
        model.fit(data.train, on_epoch_end=score_epoch, checkpoint_path=checkpoint)


def _seed_averaged_trials(
    progress: SearchProgress, grid: list[TwoTowerEnsembleParams], select_metric: str
) -> tuple[list[dict[str, Any]], dict[str, pl.Series]]:
    """One trial per (configuration, epoch): every metric averaged over the seeds, plus the
    per-seed spread of the selection metric. CI bounds are averaged too, so they describe a
    typical single run."""
    trials: list[dict[str, Any]] = []
    per_user: dict[str, pl.Series] = {}
    for number, config in enumerate(grid, start=1):
        for epoch in range(1, config.epochs + 1):
            runs = [progress.get(number, seed, epoch) for seed in config.seeds]
            per_seed = [run["metrics"][select_metric] for run in runs]
            trials.append(
                {
                    "params": {**config.model_dump(mode="json"), "epochs": epoch},
                    "train_loss": statistics.fmean(run["train_loss"] for run in runs),
                    "metrics": {
                        name: statistics.fmean(run["metrics"][name] for run in runs)
                        for name in runs[0]["metrics"]
                    },
                    "seed_spread": {
                        select_metric: per_seed,
                        "sd": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                    },
                }
            )
            per_user[f"trial_{number:02d}_epoch_{epoch:02d}"] = pl.Series(
                np.mean([run["per_user"] for run in runs], axis=0)
            )
    return trials, per_user


def _remove_resume_state(progress: SearchProgress, grid: list[TwoTowerEnsembleParams]) -> None:
    """Delete only the files this search wrote, then the folder if nothing else is in it."""
    progress.path.unlink(missing_ok=True)
    for number, config in enumerate(grid, start=1):
        for seed in config.seeds:
            (progress.path.parent / _checkpoint_name(number, seed)).unlink(missing_ok=True)
    remove_if_empty(progress.path.parent)


def run_two_tower_search(
    splits_dir: Path,
    features_dir: Path,
    report_path: Path,
    checkpoint_dir: Path,
    search: TwoTowerSearchParams,
    evaluation: EvaluationParams,
    tracker: Tracker,
) -> dict[str, Any]:
    data = load_experiment_data(splits_dir)
    features = load_features(features_dir)
    fingerprint = _search_fingerprint(search, evaluation, data, features)
    progress = SearchProgress.open(checkpoint_dir / PROGRESS_FILE, fingerprint)
    grid = two_tower_grid(search)
    for number, config in enumerate(grid, start=1):
        for seed in config.seeds:
            if progress.epochs_scored(number, seed, config.epochs) < config.epochs:
                _train_and_score(
                    number, config, seed, data, features, evaluation, tracker, progress
                )

    trials, per_user_scores = _seed_averaged_trials(progress, grid, evaluation.select_metric)
    report = summarise_search(
        trials,
        per_user_scores,
        evaluation.select_metric,
        n_samples=evaluation.bootstrap_samples,
        seed=evaluation.seed,
    )
    write_json(report, report_path)
    _remove_resume_state(progress, grid)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    run_two_tower_search(
        params.data.splits_dir,
        params.data.features_dir,
        params.data.metrics_dir / "two_tower_search.json",
        params.data.checkpoints_dir / "two_tower_search",
        params.two_tower_search,
        params.evaluation,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

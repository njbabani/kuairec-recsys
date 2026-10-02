"""Shared steps for every experiment: load splits, fit a model, evaluate it on held-out users."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from recsys.config import EvaluationParams
from recsys.evaluation.ranking import (
    EvaluationResult,
    evaluate,
    paired_bootstrap_difference,
    paired_win_rate,
    popularity_percentiles,
)
from recsys.models.base import Recommender, Scorer

# Model selection and every reported metric so far use the tune users; test stays untouched.
EVAL_SPLIT = "tune"
MODEL_COLUMNS = ("user_id", "video_id", "is_positive")
PAIR_COLUMNS = ("user_id", "video_id")


def load_split(splits_dir: Path, name: str) -> pl.DataFrame:
    """The columns every model needs from a labeled split."""
    return pl.read_parquet(splits_dir / f"{name}.parquet", columns=list(MODEL_COLUMNS))


@dataclass(frozen=True)
class ExperimentData:
    train: pl.DataFrame
    eval_split: pl.DataFrame
    popularity: pl.DataFrame  # popularity percentile of every candidate video, from train


def load_experiment_data(splits_dir: Path) -> ExperimentData:
    train, eval_split = load_split(splits_dir, "train"), load_split(splits_dir, EVAL_SPLIT)
    return ExperimentData(train, eval_split, popularity_percentiles(train, eval_split["video_id"]))


def load_features(features_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The user and video feature tables written by the ``features`` stage."""
    return (
        pl.read_parquet(features_dir / "users.parquet"),
        pl.read_parquet(features_dir / "videos.parquet"),
    )


def fit_and_evaluate(
    model: Recommender,
    train: pl.DataFrame,
    eval_split: pl.DataFrame,
    popularity: pl.DataFrame,
    evaluation: EvaluationParams,
) -> tuple[EvaluationResult, float]:
    """Fit ``model`` on ``train`` and rank every user's candidates in ``eval_split``.

    The model only ever sees (user, video) pairs to score, never their labels. Returns the
    evaluation and the fit time in seconds (kept apart because wall-clock time is not
    reproducible and does not belong in versioned metrics).
    """
    started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - started
    return evaluate_model(model, eval_split, popularity, evaluation), round(fit_seconds, 2)


def evaluate_model(
    model: Scorer,
    eval_split: pl.DataFrame,
    popularity: pl.DataFrame,
    evaluation: EvaluationParams,
) -> EvaluationResult:
    """Rank every user's candidates in ``eval_split`` with an already fitted ``model``."""
    scores = model.score(eval_split.select(PAIR_COLUMNS))
    return evaluate(
        eval_split.select(MODEL_COLUMNS).with_columns(score=scores),
        ks=evaluation.ks,
        popularity=popularity,
        n_bootstrap=evaluation.bootstrap_samples,
        seed=evaluation.seed,
    )


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

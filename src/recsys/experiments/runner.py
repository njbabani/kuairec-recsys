"""Shared steps for every experiment: load splits, fit a model, evaluate it on held-out users."""

import time
from pathlib import Path

import polars as pl

from recsys.config import EvaluationParams
from recsys.evaluation.ranking import EvaluationResult, evaluate
from recsys.models.base import Recommender

# Model selection and every reported metric so far use the tune users; test stays untouched.
EVAL_SPLIT = "tune"
MODEL_COLUMNS = ("user_id", "video_id", "is_positive")
PAIR_COLUMNS = ("user_id", "video_id")


def load_split(splits_dir: Path, name: str) -> pl.DataFrame:
    """The columns every model needs from a labeled split."""
    return pl.read_parquet(splits_dir / f"{name}.parquet", columns=list(MODEL_COLUMNS))


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

    scores = model.score(eval_split.select(PAIR_COLUMNS))
    result = evaluate(
        eval_split.select(MODEL_COLUMNS).with_columns(score=scores),
        ks=evaluation.ks,
        popularity=popularity,
        n_bootstrap=evaluation.bootstrap_samples,
        seed=evaluation.seed,
    )
    return result, round(fit_seconds, 2)

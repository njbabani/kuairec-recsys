"""Analyses of the trained re-ranker: does its gain transfer, what could it reach, what drives it.

* :func:`in_distribution_check`: the re-ranker vs two-tower ordering on validation users that
  played no part in training or tuning it (logged views, like its training data).
* :func:`cross_fitted_check`: an upper bound for these features on the fully observed users:
  rankers trained by cross-fitting on the tune users' own shortlists. It uses tune labels for
  training, so it is a diagnostic, never a model choice.
* :func:`explain`: TreeSHAP values, also centred within each user, because only differences
  between a user's candidates can change that user's ranking.
"""

from typing import Any

import numpy as np
import polars as pl

from recsys.config import EvaluationParams
from recsys.evaluation.ranking import popularity_percentiles
from recsys.experiments.runner import compare_with_reference, evaluate_model
from recsys.models.base import Scorer
from recsys.reranking.candidates import PAIR_KEYS, PrecomputedScorer
from recsys.reranking.features import FEATURE_COLUMNS, RankingFeatures
from recsys.reranking.ranker import (
    RankerConfig,
    contributions,
    predict,
    ranking_data,
    train_fixed_rounds,
)
from recsys.reranking.two_stage import LightGBMReranker, TwoStageRecommender, mark_top_n_per_user


def shortlist(pairs: pl.DataFrame, retrieval: Scorer, top_n: int) -> pl.DataFrame:
    """Each user's ``top_n`` candidates by retrieval score (``user_id``, ``video_id``)."""
    candidates = pairs.select(PAIR_KEYS)
    scored = candidates.with_columns(retrieval=retrieval.score(candidates))
    return mark_top_n_per_user(scored, "retrieval", top_n).filter("retrieved").select(PAIR_KEYS)


def in_distribution_check(
    rows: pl.DataFrame,
    train: pl.DataFrame,
    reranker: Scorer,
    retrieval: Scorer,
    evaluation: EvaluationParams,
) -> dict[str, Any]:
    """Re-ranker vs two-tower ordering on ``rows``: logged views of untouched validation users."""
    popularity = popularity_percentiles(train, rows["video_id"])
    results = {
        "reranker": evaluate_model(reranker, rows, popularity, evaluation),
        "two_tower": evaluate_model(retrieval, rows, popularity, evaluation),
    }
    return {
        "users": results["reranker"].summary["users_evaluated"],
        "models": {name: result.summary for name, result in results.items()},
        "paired": compare_with_reference(results, "two_tower", evaluation),
    }


def _round_robin_folds(users: pl.Series, folds: int) -> pl.Series:
    """Fold of each row's user: users in id order are dealt to folds like cards."""
    ordered = users.unique().sort()
    if ordered.len() < folds:
        raise ValueError(f"Cross-fitting needs at least {folds} users, got {ordered.len()}")
    fold_of = {user: position % folds for position, user in enumerate(ordered.to_list())}
    return users.replace_strict(fold_of, return_dtype=pl.Int64)


def _out_of_fold_scores(
    matrix: pl.DataFrame, labels: pl.Series, config: RankerConfig, rounds: int, folds: int
) -> np.ndarray:
    fold = _round_robin_folds(matrix["user_id"], folds).to_numpy()
    scores = np.empty(matrix.height)
    for k in range(folds):
        held_out = fold == k
        training = pl.Series(~held_out)
        booster = train_fixed_rounds(
            ranking_data(matrix.filter(training), labels.filter(training)), config, rounds
        )
        scores[held_out] = predict(
            booster, matrix.filter(pl.Series(held_out)).select(FEATURE_COLUMNS)
        )
    return scores


def cross_fitted_check(
    tune: pl.DataFrame,
    retrieval: Scorer,
    features: RankingFeatures,
    config: RankerConfig,
    rounds: int,
    folds: int,
    top_n: int,
    evaluation: EvaluationParams,
    popularity: pl.DataFrame,
) -> dict[str, Any]:
    """Two-stage pipeline whose re-ranker learned from fully observed shortlists of *other* tune
    users (same features, configuration and boosting rounds), vs two-tower ordering."""
    pairs = shortlist(tune, retrieval, top_n).join(
        tune.select(*PAIR_KEYS, "is_positive"), on=PAIR_KEYS, how="left", maintain_order="left"
    )
    matrix = features.transform(pairs)
    scores = _out_of_fold_scores(matrix, pairs["is_positive"].cast(pl.Int32), config, rounds, folds)
    cross_fitted = PrecomputedScorer(
        pairs.select(PAIR_KEYS).with_columns(score=scores), name="cross_fitted"
    )
    scorers = {
        "two_stage_cross_fitted": TwoStageRecommender(retrieval, cross_fitted, top_n),
        "two_tower": retrieval,
    }
    results = {
        name: evaluate_model(scorer, tune, popularity, evaluation)
        for name, scorer in scorers.items()
    }
    return {
        "folds": folds,
        "models": {name: result.summary for name, result in results.items()},
        "paired": compare_with_reference(results, "two_tower", evaluation),
    }


def explain(
    pairs: pl.DataFrame, reranker: LightGBMReranker, sample_rows: int, seed: int
) -> tuple[dict[str, dict[str, float]], pl.DataFrame]:
    """Mean |SHAP|, mean |SHAP centred within each user| and split gain per feature, plus a
    sample of rows (features, SHAP and within-user SHAP values) for plotting."""
    if pairs.is_empty():
        raise ValueError("No shortlist rows to explain")
    frame = reranker.features.transform(pairs)
    shap = contributions(reranker.booster, frame.select(FEATURE_COLUMNS))
    within = (
        shap.select(FEATURE_COLUMNS)
        .with_columns(user_id=frame["user_id"])
        .select(pl.col(name) - pl.col(name).mean().over("user_id") for name in FEATURE_COLUMNS)
    )
    gain = dict(
        zip(
            reranker.booster.feature_name(),
            reranker.booster.feature_importance(importance_type="gain"),
            strict=True,
        )
    )
    importance = {
        name: {
            "mean_abs_shap": float(shap[name].abs().mean()),
            "mean_abs_within_user_shap": float(within[name].abs().mean()),
            "gain": float(gain[name]),
        }
        for name in FEATURE_COLUMNS
    }
    sample = (
        frame.select(FEATURE_COLUMNS)
        .hstack(shap.rename({column: f"shap_{column}" for column in shap.columns}))
        .hstack(within.rename({column: f"within_user_shap_{column}" for column in within.columns}))
        .sample(n=min(sample_rows, frame.height), seed=seed)
    )
    return importance, sample

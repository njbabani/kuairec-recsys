"""Per-user ranking metrics on the fully observed matrix, with bootstrap confidence intervals.

Every evaluated user has a label for (almost) every candidate video, so a model ranks each
user's whole candidate list and the metrics need no missing-data correction. Metrics are
computed per user, then averaged, so heavy and light watchers count equally.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from recsys.config import RANKING_METRICS
from recsys.evaluation.concentration import gini

REQUIRED_COLUMNS = ("user_id", "video_id", "score", "is_positive")
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class EvaluationResult:
    per_user: pl.DataFrame
    summary: dict[str, float | int]


def _require_columns(scored: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in scored.columns]
    if missing:
        raise ValueError(f"Scored frame is missing column(s): {missing}")


def _require_usable_scores(scored: pl.DataFrame) -> None:
    """Null and NaN sort *first*, so a broken model would silently get its junk ranked on top."""
    _require_columns(scored)
    score = scored["score"]
    if score.null_count() or (score.dtype.is_float() and score.is_nan().any()):
        raise ValueError("Scores must not be null or NaN (use -inf to rank a candidate last)")


def rank_candidates(scored: pl.DataFrame) -> pl.DataFrame:
    """Sort each user's candidates by score and add a 1-based rank.

    Ties are broken by video_id: deterministic, though it slightly favours low ids among exactly
    tied scores (irrelevant for the models here, which rarely tie).
    """
    _require_usable_scores(scored)
    return scored.sort(
        ["user_id", "score", "video_id"], descending=[False, True, False]
    ).with_columns(rank=pl.int_range(1, pl.len() + 1).over("user_id"))


def _ideal_dcg(max_k: int) -> np.ndarray:
    """``_ideal_dcg(k)[m]``: DCG of a perfect ranking with ``m`` relevant items in the top k."""
    gains = [1 / math.log2(rank + 1) for rank in range(1, max_k + 1)]
    return np.concatenate([[0.0], np.cumsum(gains)])


def _metrics_from_ranked(ranked: pl.DataFrame, ks: Sequence[int]) -> pl.DataFrame:
    relevant = pl.col("is_positive").cast(pl.Float64)
    ranked = ranked.with_columns(
        gain=relevant / (pl.col("rank") + 1).cast(pl.Float64).log(2),
        precision_here=relevant
        * pl.col("is_positive").cast(pl.Int64).cum_sum().over("user_id")
        / pl.col("rank"),
    )
    aggregations = [
        pl.len().alias("n_candidates"),
        pl.col("is_positive").sum().alias("n_positives"),
    ]
    for k in ks:
        in_top = pl.col("rank") <= k
        aggregations += [
            relevant.filter(in_top).sum().alias(f"_hits_{k}"),
            pl.col("gain").filter(in_top).sum().alias(f"_dcg_{k}"),
            pl.col("precision_here").filter(in_top).sum().alias(f"_ap_{k}"),
        ]
    # Sorted so bootstrap resampling (positional) does not depend on group_by's row order.
    per_user = (
        ranked.group_by("user_id")
        .agg(aggregations)
        .filter(pl.col("n_positives") > 0)
        .sort("user_id")
    )

    n_positives = per_user["n_positives"].to_numpy()
    metrics = []
    for k in ks:
        relevant_in_top = pl.min_horizontal("n_positives", pl.lit(k))
        ideal = pl.Series(_ideal_dcg(k)[np.minimum(n_positives, k)])
        metrics += [
            (pl.col(f"_hits_{k}") / pl.min_horizontal("n_candidates", pl.lit(k))).alias(
                f"precision_at_{k}"
            ),
            (pl.col(f"_hits_{k}") / pl.col("n_positives")).alias(f"recall_at_{k}"),
            (pl.col(f"_dcg_{k}") / pl.lit(ideal)).alias(f"ndcg_at_{k}"),
            (pl.col(f"_ap_{k}") / relevant_in_top).alias(f"map_at_{k}"),
        ]
    return per_user.with_columns(metrics).select(
        "user_id",
        "n_candidates",
        "n_positives",
        *[f"{m}_at_{k}" for k in ks for m in RANKING_METRICS],
    )


def per_user_metrics(scored: pl.DataFrame, ks: Sequence[int]) -> pl.DataFrame:
    """Precision, recall, NDCG and MAP at each k for every user with at least one positive."""
    return _metrics_from_ranked(rank_candidates(scored), ks)


def bootstrap_mean(
    values: pl.Series, *, n_samples: int, seed: int, confidence: float = CONFIDENCE_LEVEL
) -> tuple[float, float, float]:
    """Mean of ``values`` and a percentile-bootstrap confidence interval (resampling users)."""
    x = values.cast(pl.Float64).to_numpy()
    if x.size == 0:
        raise ValueError("Cannot bootstrap an empty set of values")
    resampled = x[np.random.default_rng(seed).integers(0, x.size, size=(n_samples, x.size))]
    means = resampled.mean(axis=1)
    tail = (1 - confidence) / 2
    return float(x.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_bootstrap_difference(
    per_user_a: pl.DataFrame,
    per_user_b: pl.DataFrame,
    metric: str,
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean per-user difference (a - b) in ``metric`` with a bootstrap interval.

    Pairing on the same users cancels out how much each user watches, the biggest source of
    variance, so a real difference between models shows up far sooner than by comparing two
    separate confidence intervals. Only users present in both frames are compared.
    """
    differences = _paired_differences(per_user_a, per_user_b, metric)
    return bootstrap_mean(differences, n_samples=n_samples, seed=seed)


def paired_win_rate(per_user_a: pl.DataFrame, per_user_b: pl.DataFrame, metric: str) -> float:
    """Share of shared users for whom model a beats model b (ties count half).

    A mean gain can hide that it helps only some users; the win rate makes that visible.
    """
    differences = _paired_differences(per_user_a, per_user_b, metric)
    wins = (differences > 0).sum() + 0.5 * (differences == 0).sum()
    return float(wins / differences.len())


def _paired_differences(
    per_user_a: pl.DataFrame, per_user_b: pl.DataFrame, metric: str
) -> pl.Series:
    paired = (
        per_user_a.select("user_id", a=metric)
        .join(per_user_b.select("user_id", b=metric), on="user_id", maintain_order="left")
        .sort("user_id")
    )
    return paired["a"] - paired["b"]


def popularity_percentiles(train: pl.DataFrame, candidate_videos: pl.Series) -> pl.DataFrame:
    """Each candidate video's percentile (0 = least, 1 = most viewed) of training views."""
    candidates = (
        pl.DataFrame({"video_id": candidate_videos.unique()})
        .join(train.group_by("video_id").len(), on="video_id", how="left")
        .with_columns(pl.col("len").fill_null(0))
    )
    spread = max(candidates.height - 1, 1)
    return candidates.select(
        "video_id", popularity_pct=(pl.col("len").rank("average") - 1) / spread
    ).sort("video_id")


def top_k_concentration(
    ranked: pl.DataFrame, k: int, popularity: pl.DataFrame | None
) -> dict[str, float]:
    """How concentrated the top-k lists are: catalogue coverage, Gini, and mean popularity."""
    top = ranked.filter(pl.col("rank") <= k)
    exposure = (
        ranked.select("video_id")
        .unique()
        .join(top.group_by("video_id").len(), on="video_id", how="left")["len"]
        .fill_null(0)
    )
    result = {
        f"coverage_at_{k}": top["video_id"].n_unique() / exposure.len(),
        f"gini_at_{k}": gini(exposure),
    }
    if popularity is not None:
        recommended = top.join(popularity, on="video_id", how="left")["popularity_pct"]
        result[f"popularity_pct_at_{k}"] = float(recommended.mean())
    return result


def evaluate(
    scored: pl.DataFrame,
    ks: Sequence[int],
    popularity: pl.DataFrame | None,
    n_bootstrap: int,
    seed: int,
) -> EvaluationResult:
    """Per-user metrics plus a flat, JSON-ready summary (means with 95% bootstrap intervals)."""
    ranked = rank_candidates(scored)
    per_user = _metrics_from_ranked(ranked, ks)
    if per_user.is_empty():
        raise ValueError("No evaluated user has a positive interaction")

    summary: dict[str, float | int] = {
        "users_evaluated": per_user.height,
        "users_without_positives": scored["user_id"].n_unique() - per_user.height,
    }
    for k in ks:
        for metric in RANKING_METRICS:
            name = f"{metric}_at_{k}"
            mean, low, high = bootstrap_mean(per_user[name], n_samples=n_bootstrap, seed=seed)
            summary |= {name: mean, f"{name}_ci_low": low, f"{name}_ci_high": high}
        summary |= top_k_concentration(ranked, k, popularity)
    return EvaluationResult(per_user=per_user, summary=summary)

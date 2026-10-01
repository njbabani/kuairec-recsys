"""How concentrated attention is across items: Gini coefficient, Lorenz curve, top-k share.

Used in EDA (how skewed is exposure in the logs?) and in evaluation (do recommendations
collapse onto a few popular videos?).
"""

import math

import polars as pl

FLOAT_TOLERANCE_DIGITS = 9


def _sorted_counts(counts: pl.Series) -> pl.Series:
    if counts.is_empty():
        raise ValueError("Expected at least one count")
    if counts.null_count():
        raise ValueError("Expected counts without nulls")
    if counts.dtype.is_float() and not counts.is_finite().all():
        raise ValueError("Expected finite counts (no NaN or inf)")
    if (counts < 0).any():
        raise ValueError("Expected non-negative counts")
    if counts.sum() == 0:
        raise ValueError("Expected counts that are not all zero")
    return counts.cast(pl.Float64).sort()


def gini(counts: pl.Series) -> float:
    """Gini coefficient: 0 when every item is equally popular, (n-1)/n when one takes all."""
    x = _sorted_counts(counts)
    n = x.len()
    ranks = pl.int_range(1, n + 1, eager=True)
    return float(2 * (ranks * x).sum() / (n * x.sum()) - (n + 1) / n)


def lorenz_curve(counts: pl.Series) -> pl.DataFrame:
    """Cumulative share of interactions held by the least-popular share of items."""
    x = _sorted_counts(counts)
    n = x.len()
    return pl.DataFrame(
        {
            "share_of_items": [0.0, *(i / n for i in range(1, n + 1))],
            "share_of_interactions": [0.0, *(x.cum_sum() / x.sum()).to_list()],
        }
    )


def top_share(counts: pl.Series, fraction: float) -> float:
    """Share of all interactions held by the most popular ``fraction`` of items (at least one)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    x = _sorted_counts(counts)
    # Round before flooring: 0.29 * 100 is 28.999999999999996 in floating point.
    k = max(1, math.floor(round(fraction * x.len(), FLOAT_TOLERANCE_DIGITS)))
    return float(x.sort(descending=True).head(k).sum() / x.sum())

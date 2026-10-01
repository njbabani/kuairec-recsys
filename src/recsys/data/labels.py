"""Duration-debiased relevance labels.

On KuaiRec people watch ~7.5 s of a video whatever its length, so ``watch_ratio`` mostly
measures how *short* a video is: labelling "watch_ratio >= 2" would teach any model to recommend
short videos. Instead, in the spirit of D2Q's duration bucketing (Zhan et al., KDD 2022; D2Q
bins play time, here we bin the watch ratio), a view is positive when its watch ratio is above
the ``quantile`` of training views of similar-length videos.

Length buckets are log-spaced, so each spans at most ``width_ratio``x in length and the
mechanical effect of length on watch ratio inside a bucket is bounded by that factor. Sparse
ranges (very short or very long videos) are merged until each bucket holds ``min_views``
training views, so every cut-off is estimated from enough data. Cut-offs are fitted on training
data only, then applied unchanged to every other split.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import polars as pl

CUTOFF_INTERPOLATION = "linear"
MS_PER_SECOND = 1000


@dataclass(frozen=True)
class DurationThresholds:
    """Watch-ratio cut-offs per video-length bucket, learned from training data.

    Bucket ``i`` holds durations in ``(edges[i-1], edges[i]]``. The first and last buckets are
    open-ended, so lengths never seen in training still get a cut-off.
    """

    bucket_edges_ms: tuple[float, ...]
    watch_ratio_thresholds: tuple[float, ...]
    quantile: float

    def __post_init__(self) -> None:
        if len(self.watch_ratio_thresholds) != len(self.bucket_edges_ms) + 1:
            raise ValueError("Need exactly one more watch-ratio threshold than bucket edges")
        if any(low >= high for low, high in pairwise(self.bucket_edges_ms)):
            raise ValueError("Bucket edges must be strictly increasing")
        if not all(math.isfinite(cutoff) for cutoff in self.watch_ratio_thresholds):
            raise ValueError("Watch-ratio thresholds must be finite")
        if not 0 < self.quantile < 1:
            raise ValueError(f"quantile must be strictly between 0 and 1, got {self.quantile}")

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "quantile": self.quantile,
            "bucket_edges_ms": list(self.bucket_edges_ms),
            "watch_ratio_thresholds": list(self.watch_ratio_thresholds),
        }


def assign_duration_buckets(durations: pl.Series, bucket_edges_ms: tuple[float, ...]) -> pl.Series:
    """Index of each duration's bucket: the number of edges strictly below it."""
    edges = pl.Series(bucket_edges_ms, dtype=pl.Float64)
    return edges.search_sorted(durations.cast(pl.Float64), side="left").cast(pl.Int64)


def _validate_bucketing(width_ratio: float, min_views: int) -> None:
    if width_ratio <= 1:
        raise ValueError(f"width_ratio must be greater than 1, got {width_ratio}")
    if min_views < 1:
        raise ValueError(f"min_views must be at least 1, got {min_views}")


def _log_spaced_edges(shortest: float, longest: float, width_ratio: float) -> list[float]:
    edges: list[float] = []
    edge = shortest * width_ratio
    while edge < longest:
        edges.append(edge)
        edge *= width_ratio
    return edges


def length_buckets(
    durations: pl.Series, *, width_ratio: float, min_views: int
) -> tuple[float, ...]:
    """Bucket edges (ms): log-spaced by ``width_ratio``, then merged from the shortest length up
    until every bucket holds at least ``min_views`` values. Too little data gives one bucket."""
    _validate_bucketing(width_ratio, min_views)
    if durations.is_empty():
        return ()
    candidates = _log_spaced_edges(float(durations.min()), float(durations.max()), width_ratio)
    counts = (
        pl.DataFrame({"bucket": assign_duration_buckets(durations, tuple(candidates))})
        .group_by("bucket")
        .len()
    )
    views_per_candidate = dict(zip(counts["bucket"], counts["len"], strict=True))

    kept: list[float] = []
    running = 0
    for index, edge in enumerate(candidates):
        running += views_per_candidate.get(index, 0)
        if running >= min_views:
            kept.append(edge)
            running = 0
    trailing = running + views_per_candidate.get(len(candidates), 0)
    if kept and trailing < min_views:
        kept.pop()  # fold an under-filled last bucket into its neighbour
    return tuple(kept)


def fit_duration_thresholds(
    train: pl.DataFrame, *, width_ratio: float, min_views_per_bucket: int, quantile: float
) -> DurationThresholds:
    """Learn length buckets on ``train`` and the watch-ratio ``quantile`` within each."""
    _validate_bucketing(width_ratio, min_views_per_bucket)
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}")
    if train.is_empty():
        raise ValueError("Cannot fit label thresholds on empty training data")

    durations = train["video_duration"]
    edges = length_buckets(durations, width_ratio=width_ratio, min_views=min_views_per_bucket)
    cutoffs = (
        pl.DataFrame(
            {
                "duration_bucket": assign_duration_buckets(durations, edges),
                "watch_ratio": train["watch_ratio"],
            }
        )
        .group_by("duration_bucket")
        .agg(pl.col("watch_ratio").quantile(quantile, interpolation=CUTOFF_INTERPOLATION))
        .sort("duration_bucket")
    )
    if cutoffs.height != len(edges) + 1:  # length_buckets guarantees no empty bucket
        raise RuntimeError("Every length bucket must contain training views")
    return DurationThresholds(edges, tuple(cutoffs["watch_ratio"].to_list()), quantile)


def label_interactions(df: pl.DataFrame, thresholds: DurationThresholds) -> pl.DataFrame:
    """Return ``df`` plus ``duration_bucket`` and ``is_positive`` (ratio above its cut-off)."""
    buckets = assign_duration_buckets(df["video_duration"], thresholds.bucket_edges_ms)
    cutoffs = pl.Series(thresholds.watch_ratio_thresholds, dtype=pl.Float64).gather(buckets)
    return df.with_columns(duration_bucket=buckets, is_positive=df["watch_ratio"] > cutoffs)


def positive_rate_by_length(
    labeled: pl.DataFrame, *, width_ratio: float, min_views: int
) -> pl.DataFrame:
    """Positive rate in fine length slices of ``labeled`` itself.

    A check for residual duration bias: for a debiased label this should be roughly flat.
    """
    durations = labeled["video_duration"]
    edges = length_buckets(durations, width_ratio=width_ratio, min_views=min_views)
    return (
        labeled.with_columns(length_slice=assign_duration_buckets(durations, edges))
        .group_by("length_slice")
        .agg(
            length_min_s=pl.col("video_duration").min() / MS_PER_SECOND,
            length_max_s=pl.col("video_duration").max() / MS_PER_SECOND,
            views=pl.len(),
            positive_rate=pl.col("is_positive").mean(),
        )
        .sort("length_slice")
        .drop("length_slice")
    )

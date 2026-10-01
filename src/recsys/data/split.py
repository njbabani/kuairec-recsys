"""Labeled train / valid / tune / test splits.

* **train / valid** come from big_matrix (the logged feed) with one global time cut-off, so no
  model is fitted on events later than those it is validated on. valid serves training-time
  checks (e.g. early stopping), *not* model selection: unlike the test set it contains re-watches
  and cold-start videos, and its exposure was chosen by the platform's own recommender.
* **tune / test** come from small_matrix (fully observed) by a stable hash of the user id. tune
  is for model selection; test is touched only for the final report and the A/B simulation.

small_matrix is *not* later in time than training: it spans the same weeks as big_matrix. It is
a preference-elicitation set (each of its users was shown each of its videos), not a forecast of
the future, and none of its (user, video) pairs appears in big_matrix (checked in ``prepare``).

Labels come from :mod:`recsys.data.labels`, fitted on train only.

Run as a DVC stage: ``python -m recsys.data.split``.
"""

import json
import logging
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import pandera.polars as pa
import polars as pl

from recsys.config import LabelParams, SplitParams, load_params
from recsys.data.labels import (
    DurationThresholds,
    fit_duration_thresholds,
    label_interactions,
    positive_rate_by_length,
)
from recsys.data.schemas import (
    BIG_MATRIX_SCHEMA,
    INTERACTION_KEYS,
    LOCAL_TIMEZONE,
    SMALL_MATRIX_SCHEMA,
    TEST_SPLIT_SCHEMA,
    TRAIN_SPLIT_SCHEMA,
    TUNE_SPLIT_SCHEMA,
    VALID_SPLIT_SCHEMA,
)
from recsys.hashing import stable_fraction
from recsys.io import write_json, write_parquet_tables

logger = logging.getLogger(__name__)

SPLIT_SCHEMAS = MappingProxyType(
    {
        "train": TRAIN_SPLIT_SCHEMA,
        "valid": VALID_SPLIT_SCHEMA,
        "tune": TUNE_SPLIT_SCHEMA,
        "test": TEST_SPLIT_SCHEMA,
    }
)
SUMMARY_DECIMALS = 4
# Residual duration-bias check: positive rate in fine (10%-wide) length slices of each split.
BALANCE_WIDTH_RATIO = 1.1
BALANCE_MIN_VIEWS = 2000


def local_midnight(day: date) -> datetime:
    """Start of ``day`` in the platform's local timezone."""
    return datetime.combine(day, time(), tzinfo=ZoneInfo(LOCAL_TIMEZONE))


def _require_non_empty(context: str, frames: dict[str, pl.DataFrame]) -> None:
    for name, frame in frames.items():
        if frame.is_empty():
            raise ValueError(f"{context} leaves {name} empty")


def temporal_split(
    interactions: pl.DataFrame, valid_start: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Events before ``valid_start`` go to train, events at or after it to validation."""
    missing_time = interactions["event_time"].null_count()
    if missing_time:
        raise ValueError(
            f"{missing_time} interaction(s) have no event_time; a temporal split would drop them"
        )
    in_valid = pl.col("event_time") >= valid_start
    train, valid = interactions.filter(~in_valid), interactions.filter(in_valid)
    _require_non_empty(f"Splitting at {valid_start.isoformat()}", {"train": train, "valid": valid})
    return train, valid


def split_users(
    interactions: pl.DataFrame, *, tune_share: float, salt: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Send each user (all of their rows) to tune with probability ``tune_share``, else test.

    Assignment hashes ``"{salt}:{user_id}"``, so it is stable across runs and machines.
    """
    users = interactions["user_id"].unique().to_list()
    tune_users = [user for user in users if stable_fraction(f"{salt}:{user}") < tune_share]
    in_tune = pl.col("user_id").is_in(tune_users)
    tune, test = interactions.filter(in_tune), interactions.filter(~in_tune)
    _require_non_empty(f"A tune share of {tune_share}", {"tune": tune, "test": test})
    return tune, test


def build_splits(
    big: pl.DataFrame, small: pl.DataFrame, label: LabelParams, split: SplitParams
) -> tuple[dict[str, pl.DataFrame], DurationThresholds]:
    """Split both matrices, fit label cut-offs on train, and label all four splits."""
    train, valid = temporal_split(big, local_midnight(split.valid_start))
    tune, test = split_users(small, tune_share=split.tune_user_share, salt=split.assignment_salt)
    thresholds = fit_duration_thresholds(
        train,
        width_ratio=label.bucket_width_ratio,
        min_views_per_bucket=label.min_views_per_bucket,
        quantile=label.positive_quantile,
    )
    frames = {"train": train, "valid": valid, "tune": tune, "test": test}
    labeled = {name: label_interactions(frame, thresholds) for name, frame in frames.items()}
    return labeled, thresholds


def _share(part: int, whole: int) -> float:
    return round(part / whole, SUMMARY_DECIMALS) if whole else 0.0


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, SUMMARY_DECIMALS)


def _describe(frame: pl.DataFrame) -> dict[str, Any]:
    balance = positive_rate_by_length(
        frame, width_ratio=BALANCE_WIDTH_RATIO, min_views=BALANCE_MIN_VIEWS
    )
    return {
        "rows": frame.height,
        "users": frame["user_id"].n_unique(),
        "videos": frame["video_id"].n_unique(),
        "positive_rate": _rounded(frame["is_positive"].mean()),
        "length_balance": {
            "slices": balance.height,
            "min_positive_rate": _rounded(balance["positive_rate"].min()),
            "max_positive_rate": _rounded(balance["positive_rate"].max()),
        },
    }


def _seen_in_train_share(frame: pl.DataFrame, train: pl.DataFrame, key: str) -> float:
    seen = frame.join(train.select(key).unique(), on=key, how="semi")[key].n_unique()
    return _share(seen, frame[key].n_unique())


def _coverage(frame: pl.DataFrame, train: pl.DataFrame) -> dict[str, Any]:
    """How a fully observed split relates to training: overlap and time range."""
    times = frame["event_time"].drop_nulls()
    return {
        "users_in_train_share": _seen_in_train_share(frame, train, "user_id"),
        "videos_in_train_share": _seen_in_train_share(frame, train, "video_id"),
        "null_event_time_share": _share(frame["event_time"].null_count(), frame.height),
        "first_event": times.min().isoformat() if times.len() else None,
        "last_event": times.max().isoformat() if times.len() else None,
    }


def summarize_splits(
    splits: dict[str, pl.DataFrame], thresholds: DurationThresholds, split: SplitParams
) -> dict[str, Any]:
    """Sizes, label balance, cold-start, repeat and coverage stats for the DVC metrics file."""
    train, valid = splits["train"], splits["valid"]
    cold_start = valid.join(train.select("video_id").unique(), on="video_id", how="anti")
    repeats = valid.join(train.select(INTERACTION_KEYS).unique(), on=INTERACTION_KEYS, how="semi")
    return {
        "valid_start": split.valid_start.isoformat(),
        "tune_user_share": split.tune_user_share,
        "label": thresholds.to_dict(),
        "train": _describe(train),
        "valid": {
            **_describe(valid),
            "cold_start_videos": cold_start["video_id"].n_unique(),
            "cold_start_video_row_share": _share(cold_start.height, valid.height),
            "repeat_pair_row_share": _share(repeats.height, valid.height),
        },
        **{
            name: {**_describe(splits[name]), **_coverage(splits[name], train)}
            for name in ("tune", "test")
        },
    }


def run_split(
    processed_dir: Path,
    out_dir: Path,
    summary_path: Path,
    label: LabelParams,
    split: SplitParams,
) -> dict[str, Any]:
    """Validate inputs, build and validate the splits, then write; a failure writes nothing."""
    big = BIG_MATRIX_SCHEMA.validate(
        pl.read_parquet(processed_dir / "big_matrix.parquet"), lazy=True
    )
    small = SMALL_MATRIX_SCHEMA.validate(
        pl.read_parquet(processed_dir / "small_matrix.parquet"), lazy=True
    )
    splits, thresholds = build_splits(big, small, label, split)
    validated = {name: SPLIT_SCHEMAS[name].validate(df, lazy=True) for name, df in splits.items()}
    summary = summarize_splits(validated, thresholds, split)

    write_parquet_tables(validated, out_dir)
    write_json(summary, summary_path)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    try:
        summary = run_split(
            params.data.processed_dir,
            params.data.splits_dir,
            params.data.split_summary_path,
            params.label,
            params.split,
        )
    except pa.errors.SchemaErrors as exc:
        logger.error("Data contract violated; nothing written.\n%s", exc.failure_cases)
        raise
    logger.info("Split summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

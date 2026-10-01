"""Data contracts for the processed KuaiRec tables, enforced with pandera before writing.

Durations are in milliseconds. ``event_time`` is Asia/Shanghai local time (the platform's
timezone), derived from the source epoch timestamp.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandera.polars as pa
import polars as pl

LOCAL_TIMEZONE = "Asia/Shanghai"
EVENT_TIME_DTYPE = pl.Datetime("ms", LOCAL_TIMEZONE)
INTERACTION_KEYS = ("user_id", "video_id")
COLLECT_COLUMNS = (
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
)

# KuaiRec was logged Jun-Sep 2020; a timestamp outside this window means a unit or timezone bug.
DATASET_WINDOW_START = datetime(2020, 6, 1, tzinfo=ZoneInfo(LOCAL_TIMEZONE))
DATASET_WINDOW_END = datetime(2020, 10, 1, tzinfo=ZoneInfo(LOCAL_TIMEZONE))


def _non_negative_int(*, nullable: bool = False) -> pa.Column:
    return pa.Column(pl.Int64, pa.Check.ge(0), nullable=nullable)


def _interaction_schema(
    name: str, *, event_time_nullable: bool, unique_pairs: bool, labeled: bool = False
) -> pa.DataFrameSchema:
    columns = {
        "user_id": _non_negative_int(),
        "video_id": _non_negative_int(),
        "play_duration": _non_negative_int(),
        "video_duration": pa.Column(pl.Int64, pa.Check.gt(0)),
        # Finite and non-negative: in_range with an open upper bound also rejects inf/NaN.
        "watch_ratio": pa.Column(pl.Float64, pa.Check.in_range(0, math.inf, include_max=False)),
        "event_time": pa.Column(
            EVENT_TIME_DTYPE,
            pa.Check.in_range(DATASET_WINDOW_START, DATASET_WINDOW_END, include_max=False),
            nullable=event_time_nullable,
        ),
    }
    if labeled:
        columns |= {
            "duration_bucket": _non_negative_int(),
            "is_positive": pa.Column(pl.Boolean),
        }
    return pa.DataFrameSchema(
        columns,
        strict=True,
        ordered=True,
        unique=list(INTERACTION_KEYS) if unique_pairs else None,
        name=name,
    )


# Logged interactions shaped by the platform's own recommender; re-watches repeat a pair.
BIG_MATRIX_SCHEMA = _interaction_schema("big_matrix", event_time_nullable=False, unique_pairs=False)

# Fully observed evaluation matrix: exactly one reaction per (user, video); ~4% lack a time.
SMALL_MATRIX_SCHEMA = _interaction_schema(
    "small_matrix", event_time_nullable=True, unique_pairs=True
)

# Labeled splits: train/valid come from big_matrix by time; tune/test split small_matrix by user.
TRAIN_SPLIT_SCHEMA = _interaction_schema(
    "train", event_time_nullable=False, unique_pairs=False, labeled=True
)
VALID_SPLIT_SCHEMA = _interaction_schema(
    "valid", event_time_nullable=False, unique_pairs=False, labeled=True
)
TUNE_SPLIT_SCHEMA = _interaction_schema(
    "tune", event_time_nullable=True, unique_pairs=True, labeled=True
)
TEST_SPLIT_SCHEMA = _interaction_schema(
    "test", event_time_nullable=True, unique_pairs=True, labeled=True
)

USERS_SCHEMA = pa.DataFrameSchema(
    {"user_id": _non_negative_int()}, unique=["user_id"], name="users"
)

VIDEO_CATEGORIES_SCHEMA = pa.DataFrameSchema(
    {"video_id": _non_negative_int(), "category_ids": pa.Column(pl.List(pl.Int64))},
    strict=True,
    unique=["video_id"],
    name="video_categories",
)

VIDEO_CAPTIONS_SCHEMA = pa.DataFrameSchema(
    {"video_id": _non_negative_int()}, unique=["video_id"], name="video_captions"
)

VIDEO_DAILY_STATS_SCHEMA = pa.DataFrameSchema(
    {
        "video_id": _non_negative_int(),
        "date": pa.Column(pl.Date),
        "video_duration": pa.Column(pl.Float64, nullable=True),
        **{column: _non_negative_int(nullable=True) for column in COLLECT_COLUMNS},
    },
    unique=["video_id", "date"],
    name="video_daily_stats",
)

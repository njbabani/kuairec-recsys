"""Model-ready user and video features, built from profile and catalogue data only.

Run as a DVC stage: ``python -m recsys.data.features``.

Nothing here depends on how anyone reacted to a video, so the features can be used for every
split without leaking labels. A video's length is part of the catalogue (it is fixed at upload),
so it is read from every interaction table, including the evaluation ones.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandera.polars as pa
import polars as pl

from recsys.config import load_params
from recsys.io import write_parquet_tables

logger = logging.getLogger(__name__)

# Profile attributes with a small set of values (anonymised one-hot groups, activity levels,
# bucketed counts). Stored as text so the model can treat every column the same way.
USER_CATEGORICAL_COLUMNS = (
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *(f"onehot_feat{i}" for i in range(18)),
)
# Heavy-tailed counts, compressed with log1p.
USER_COUNT_COLUMNS = ("follow_user_num", "fans_user_num", "friend_user_num", "register_days")
MS_PER_SECOND = 1000

# NaN or +-inf in one row would turn every standardised value of the column into NaN.
FINITE = pa.Check(lambda data: data.lazyframe.select(pl.col(data.key).is_finite()), name="finite")

USER_FEATURES_SCHEMA = pa.DataFrameSchema(
    {
        "user_id": pa.Column(pl.Int64, pa.Check.ge(0)),
        **{name: pa.Column(pl.String, nullable=True) for name in USER_CATEGORICAL_COLUMNS},
        **{
            f"log_{name}": pa.Column(pl.Float64, [pa.Check.ge(0), FINITE], nullable=True)
            for name in USER_COUNT_COLUMNS
        },
    },
    strict=True,
    unique=["user_id"],
    name="user_features",
)

VIDEO_FEATURES_SCHEMA = pa.DataFrameSchema(
    {
        "video_id": pa.Column(pl.Int64, pa.Check.ge(0)),
        "category_ids": pa.Column(pl.List(pl.Int64)),
        "log_duration_s": pa.Column(pl.Float64, FINITE, nullable=True),
    },
    strict=True,
    unique=["video_id"],
    name="video_features",
)


def build_user_features(users: pl.DataFrame) -> pl.DataFrame:
    return users.select(
        "user_id",
        *(pl.col(name).cast(pl.String) for name in USER_CATEGORICAL_COLUMNS),
        *(
            pl.col(name).cast(pl.Float64).log1p().alias(f"log_{name}")
            for name in USER_COUNT_COLUMNS
        ),
    ).sort("user_id")


def build_video_features(
    video_categories: pl.DataFrame, interactions: Sequence[pl.DataFrame | pl.LazyFrame]
) -> pl.DataFrame:
    """Categories plus log length in seconds (median over every logged view; null if unseen).

    The median absorbs the few-millisecond jitter in how lengths were logged.
    """
    lengths = (
        pl.concat([frame.lazy().select("video_id", "video_duration") for frame in interactions])
        .group_by("video_id")
        .agg(log_duration_s=(pl.col("video_duration").median() / MS_PER_SECOND).log())
        .collect()
    )
    return (
        video_categories.select("video_id", "category_ids")
        .join(lengths, on="video_id", how="left")
        .sort("video_id")
    )


def run_features(processed_dir: Path, features_dir: Path) -> dict[str, pl.DataFrame]:
    """Build, validate and write both tables; nothing is written if either fails its contract."""
    users = build_user_features(pl.read_parquet(processed_dir / "users.parquet"))
    videos = build_video_features(
        pl.read_parquet(processed_dir / "video_categories.parquet"),
        [
            pl.scan_parquet(processed_dir / f"{table}.parquet")
            for table in ("big_matrix", "small_matrix")
        ],
    )
    tables = {
        "users": USER_FEATURES_SCHEMA.validate(users, lazy=True),
        "videos": VIDEO_FEATURES_SCHEMA.validate(videos, lazy=True),
    }
    write_parquet_tables(tables, features_dir)
    return tables


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    try:
        run_features(params.data.processed_dir, params.data.features_dir)
    except pa.errors.SchemaErrors as exc:
        logger.error("Data contract violated; nothing written.\n%s", exc.failure_cases)
        raise


if __name__ == "__main__":
    main()

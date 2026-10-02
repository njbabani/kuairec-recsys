"""Compact tables for the Streamlit demo, built from the pipeline's outputs.

Run as a DVC stage: ``python -m recsys.demo.bundle``. The demo reads only these tables, the
per-user A/B outcomes and the git-tracked reports, so it starts in seconds and never loads a
model (nor PyTorch or LightGBM):

* ``sessions.parquet``: every policy's session for every test user, one row per video, with the
  video's category and length and what the user did with it;
* ``users.parquet``: each test user's A/B arm and their history before the experiment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from recsys.abtest.outcomes import MS_PER_SECOND, PAIR_KEYS
from recsys.config import load_params
from recsys.data.categories import with_english_category_names
from recsys.io import PARQUET_COMPRESSION

logger = logging.getLogger(__name__)

TOP_CATEGORIES = 3  # favourite categories shown per user
UNKNOWN_CATEGORY = "Unknown"
REACTION_COLUMNS = ("is_positive", "play_duration", "video_duration")


@dataclass(frozen=True)
class DemoDataPaths:
    sessions_path: Path
    outcomes_path: Path
    splits_dir: Path
    captions_path: Path
    out_dir: Path


def video_catalog(captions: pl.DataFrame) -> pl.DataFrame:
    """Each video's first-level category in English (the source name when none is known)."""
    named = with_english_category_names(
        captions.select("video_id", "first_level_category_id", "first_level_category_name")
    )
    return named.select("video_id", category="category_en")


def session_videos(
    sessions: pl.DataFrame, reactions: pl.DataFrame, catalog: pl.DataFrame
) -> pl.DataFrame:
    """One row per (policy, user, position): the video, its category and length, and the
    user's recorded reaction to it."""
    joined = sessions.join(
        reactions.select(*PAIR_KEYS, *REACTION_COLUMNS), on=PAIR_KEYS, how="left"
    ).join(catalog, on="video_id", how="left")
    if joined.height != sessions.height:
        raise ValueError(
            f"{sessions.height} session videos became {joined.height} rows: the reactions or the "
            "catalog have duplicate rows"
        )
    missing = joined.select(pl.any_horizontal(pl.col(REACTION_COLUMNS).is_null()).sum()).item()
    if missing:
        raise ValueError(f"{missing} session video(s) have no complete recorded reaction")
    return joined.select(
        "policy",
        "user_id",
        "position",
        "video_id",
        pl.col("category").fill_null(UNKNOWN_CATEGORY),
        duration_s=pl.col("video_duration") / MS_PER_SECOND,
        watched_s=pl.col("play_duration") / MS_PER_SECOND,
        liked="is_positive",
    ).sort("policy", "user_id", "position")


def _favourite_categories(history: pl.DataFrame) -> pl.DataFrame:
    """Each user's most-liked categories in training (ties: alphabetical)."""
    return (
        history.filter(pl.col("is_positive"))
        .group_by("user_id", "category")
        .len()
        .sort(["user_id", "len", "category"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True)
        .agg(favourite_categories=pl.col("category").head(TOP_CATEGORIES))
    )


def user_profiles(
    train: pl.DataFrame, test: pl.DataFrame, arms: pl.DataFrame, catalog: pl.DataFrame
) -> pl.DataFrame:
    """Per test user: the A/B arm, how much they watched and liked before the experiment, their
    favourite categories, and how many of the candidate videos they like (how picky they are)."""
    users = arms.select("user_id", "in_treatment")
    history = (
        train.join(users.select("user_id"), on="user_id")
        .join(catalog, on="video_id", how="left")
        .with_columns(pl.col("category").fill_null(UNKNOWN_CATEGORY))
    )
    summary = history.group_by("user_id").agg(
        training_views=pl.len(), training_positive_rate=pl.col("is_positive").mean()
    )
    picky = test.group_by("user_id").agg(liked_share=pl.col("is_positive").mean())
    return (
        users.join(summary, on="user_id", how="left")
        .join(_favourite_categories(history), on="user_id", how="left")
        .join(picky, on="user_id", how="left")
        .with_columns(
            pl.col("training_views").fill_null(0).cast(pl.Int64),
            pl.col("favourite_categories").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        )
        .sort("user_id")
    )


def run_demo_data(paths: DemoDataPaths) -> dict[str, Any]:
    catalog = video_catalog(pl.read_parquet(paths.captions_path))
    test = pl.read_parquet(
        paths.splits_dir / "test.parquet",
        columns=[*PAIR_KEYS, "is_positive", "play_duration", "video_duration"],
    )
    arms = pl.read_parquet(paths.outcomes_path, columns=["user_id", "in_treatment"]).unique()
    train = pl.read_parquet(paths.splits_dir / "train.parquet", columns=[*PAIR_KEYS, "is_positive"])
    videos = session_videos(pl.read_parquet(paths.sessions_path), test, catalog)
    users = user_profiles(train, test.select("user_id", "is_positive"), arms, catalog)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    videos.write_parquet(paths.out_dir / "sessions.parquet", compression=PARQUET_COMPRESSION)
    users.write_parquet(paths.out_dir / "users.parquet", compression=PARQUET_COMPRESSION)
    logger.info("Demo tables: %d session videos, %d users", videos.height, users.height)
    return {"session_videos": videos.height, "users": users.height}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = load_params().data
    run_demo_data(
        DemoDataPaths(
            sessions_path=data.ab_dir / "sessions.parquet",
            outcomes_path=data.ab_dir / "user_outcomes.parquet",
            splits_dir=data.splits_dir,
            captions_path=data.processed_dir / "video_captions.parquet",
            out_dir=data.demo_dir,
        )
    )


if __name__ == "__main__":
    main()

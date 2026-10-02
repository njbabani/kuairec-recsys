"""A miniature project root for the demo app: params, reports and data from a synthetic world.

The A/B outputs come from the real ``ab_test`` stage and the demo tables from the real
``demo_data`` stage, so the app is tested against exactly what the pipeline writes.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from recsys.config import ABExperimentParams, ABTestParams
from recsys.data.categories import CATEGORY_NAMES_EN
from recsys.demo.bundle import DemoDataPaths, run_demo_data
from recsys.experiments.ab_test import ABTestPaths, run_ab_test
from recsys.tracking import InMemoryTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_VIDEO_OFFSET = 100  # training views are of other videos than the test matrix
CATEGORY_IDS = (1, 2, 3, 4, 5)  # Dance, Music, Gaming, Beauty & makeup, Fashion


@dataclass(frozen=True)
class DemoWorld:
    """Facts of the synthetic world the demo tests share."""

    n_users: int = 120
    n_videos: int = 20
    session_length: int = 3
    # Each policy always shows the same three videos; later videos are liked more often.
    policy_videos: dict[str, list[int]] = field(
        default_factory=lambda: {
            "random": [0, 7, 14],
            "popularity_engagement": [3, 4, 5],
            "als": [9, 10, 11],
            "two_tower": [15, 16, 17],
            "two_stage": [15, 16, 18],
        }
    )

    @staticmethod
    def category_id(video: int) -> int:
        return CATEGORY_IDS[video % len(CATEGORY_IDS)]

    def category_name(self, video: int) -> str:
        return CATEGORY_NAMES_EN[self.category_id(video)]

    def ab_params(self) -> ABTestParams:
        return ABTestParams(
            session_length=self.session_length,
            treatment_share=0.5,
            salt="demo-test",
            alpha=0.05,
            power=0.8,
            simulations=100,
            looks=4,
            srm_drop_shares=(0.1, 0.5),
            seed=0,
            aa_policy="two_tower",
            experiments=(
                ABExperimentParams(name="two_tower_vs_als", control="als", treatment="two_tower"),
                ABExperimentParams(
                    name="two_stage_vs_two_tower", control="two_tower", treatment="two_stage"
                ),
            ),
            primary_experiment="two_tower_vs_als",
        )


WORLD = DemoWorld()


@pytest.fixture(scope="session")
def demo_world() -> DemoWorld:
    return WORLD


def _write_splits(splits: Path, rng: np.random.Generator) -> None:
    activity = rng.uniform(0.1, 0.6, size=WORLD.n_users)  # each user's tendency to like
    users = np.repeat(np.arange(WORLD.n_users), WORLD.n_videos)
    videos = np.tile(np.arange(WORLD.n_videos), WORLD.n_users)
    pl.DataFrame(
        {
            "user_id": users,
            "video_id": videos,
            "is_positive": rng.random(users.size) < np.clip(activity[users] + videos / 40, 0, 1),
            "play_duration": rng.integers(1000, 20000, size=users.size),
            "video_duration": 1000 * (5 + videos),
        }
    ).write_parquet(splits / "test.parquet")
    pl.DataFrame(
        {
            "user_id": users,
            "video_id": videos + TRAIN_VIDEO_OFFSET,
            "is_positive": rng.random(users.size) < activity[users],
        }
    ).write_parquet(splits / "train.parquet")


def _write_captions(path: Path) -> None:
    videos = [
        *range(WORLD.n_videos),
        *range(TRAIN_VIDEO_OFFSET, TRAIN_VIDEO_OFFSET + WORLD.n_videos),
    ]
    pl.DataFrame(
        {
            "video_id": videos,
            "first_level_category_id": [WORLD.category_id(video) for video in videos],
            "first_level_category_name": ["source name"] * len(videos),
        }
    ).write_parquet(path)


def _write_sessions(path: Path) -> None:
    pl.DataFrame(
        [
            (policy, user, video, position)
            for policy, chosen in WORLD.policy_videos.items()
            for user in range(WORLD.n_users)
            for position, video in enumerate(chosen, start=1)
        ],
        schema=["policy", "user_id", "video_id", "position"],
        orient="row",
    ).write_parquet(path)


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo_root")
    splits, processed, ab = root / "data/splits", root / "data/processed", root / "data/ab"
    for directory in (splits, processed, ab, root / "reports/metrics"):
        directory.mkdir(parents=True)
    _write_splits(splits, np.random.default_rng(0))
    _write_captions(processed / "video_captions.parquet")
    _write_sessions(ab / "sessions.parquet")
    shutil.copy(REPO_ROOT / "params.yaml", root / "params.yaml")
    shutil.copy(  # git-tracked, so available wherever the tests run
        REPO_ROOT / "reports/metrics/test_evaluation.json", root / "reports/metrics"
    )
    run_ab_test(
        ABTestPaths(
            sessions_path=ab / "sessions.parquet",
            splits_dir=splits,
            metrics_path=root / "reports/metrics/ab_test.json",
            outcomes_path=ab / "user_outcomes.parquet",
        ),
        WORLD.ab_params(),
        InMemoryTracker(),
    )
    run_demo_data(
        DemoDataPaths(
            sessions_path=ab / "sessions.parquet",
            outcomes_path=ab / "user_outcomes.parquet",
            splits_dir=splits,
            captions_path=processed / "video_captions.parquet",
            out_dir=root / "data/demo",
        )
    )
    return root


@pytest.fixture
def reports_only_root(demo_root, tmp_path) -> Path:
    """A fresh clone before any pipeline run: params and the git-tracked reports, no data."""
    root = tmp_path / "fresh_clone"
    (root / "reports" / "metrics").mkdir(parents=True)
    shutil.copy(demo_root / "params.yaml", root / "params.yaml")
    for report in ("test_evaluation.json", "ab_test.json"):
        shutil.copy(demo_root / "reports" / "metrics" / report, root / "reports" / "metrics")
    return root

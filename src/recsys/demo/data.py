"""What the demo app reads, where it lives, and what is still missing.

Paths come from the project's ``params.yaml``, so the app shows exactly what the pipeline wrote.
Each loader checks that a file has what the app needs and fails with a clear message otherwise;
pages use :func:`artifacts` to tell a visitor which command builds anything missing.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import polars as pl

from recsys.config import Policy, load_params
from recsys.experiments.ab_test import UserTable

ROOT_ENV = "RECSYS_ROOT"  # points the app at a project folder other than the working directory
PARAMS_FILE = "params.yaml"
POLICY_ORDER = get_args(Policy)  # weakest to strongest, as in the reports

TEST_REPORT_KEYS = ("users", "models", "paired")
AB_REPORT_KEYS = (
    "users",
    "session_length",
    "design",
    "policies",
    "power",
    "experiments",
    "aa_test",
    "assignment",
    "peeking",
    "srm_demo",
)
OUTCOME_COLUMNS = ("user_id", "policy", "liked", "watch_time_s", "covariate", "in_treatment")
SESSION_VIDEO_COLUMNS = (
    "policy",
    "user_id",
    "position",
    "video_id",
    "category",
    "duration_s",
    "watched_s",
    "liked",
)
USER_COLUMNS = (
    "user_id",
    "in_treatment",
    "training_views",
    "training_positive_rate",
    "favourite_categories",
    "liked_share",
)
SESSION_COLUMNS = ("position", "video_id", "category", "duration_s", "watched_s", "liked")


def find_root(start: Path | None = None) -> Path:
    """The project folder: ``$RECSYS_ROOT`` if set, otherwise the nearest folder at or above
    ``start`` that holds ``params.yaml``. Without ``start``, the working directory is searched
    first, then the folders above this file (the app's own repository)."""
    configured = os.environ.get(ROOT_ENV)
    if configured:
        root = Path(configured)
        if not (root / PARAMS_FILE).is_file():
            raise FileNotFoundError(f"{ROOT_ENV}={root} has no {PARAMS_FILE}")
        return root
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).parent]
    for candidate in starts:
        resolved = candidate.resolve()
        for folder in (resolved, *resolved.parents):
            if (folder / PARAMS_FILE).is_file():
                return folder
    raise FileNotFoundError(
        f"No {PARAMS_FILE} at or above {starts[0]}: run the app from the repository or set "
        f"{ROOT_ENV}"
    )


@dataclass(frozen=True)
class DemoPaths:
    root: Path
    test_report: Path
    ab_report: Path
    user_outcomes: Path
    session_videos: Path
    users: Path

    @classmethod
    def from_root(cls, root: Path) -> "DemoPaths":
        data = load_params(root / PARAMS_FILE).data
        return cls(
            root=root,
            test_report=root / data.metrics_dir / "test_evaluation.json",
            ab_report=root / data.metrics_dir / "ab_test.json",
            user_outcomes=root / data.ab_dir / "user_outcomes.parquet",
            session_videos=root / data.demo_dir / "sessions.parquet",
            users=root / data.demo_dir / "users.parquet",
        )


@dataclass(frozen=True)
class Artifact:
    """Files a page needs, and the command that builds them."""

    label: str
    paths: tuple[Path, ...]
    command: str

    @property
    def available(self) -> bool:
        return all(path.is_file() for path in self.paths)


def artifacts(paths: DemoPaths) -> dict[str, Artifact]:
    return {
        "test_report": Artifact(
            "The final test evaluation report", (paths.test_report,), "make experiments"
        ),
        "ab_report": Artifact("The A/B testing report", (paths.ab_report,), "make experiments"),
        "ab_outcomes": Artifact(
            "Every test user's outcome under every policy",
            (paths.user_outcomes,),
            "make experiments",
        ),
        "demo_tables": Artifact(
            "Sessions and profiles of the test users",
            (paths.session_videos, paths.users),
            "make demo-data",
        ),
    }


def _read_report(path: Path, keys: tuple[str, ...], what: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in keys if key not in report]
    if missing:
        raise ValueError(f"{what} ({path}) is missing {missing}; rebuild it with the pipeline")
    return report


def _read_table(path: Path, columns: tuple[str, ...], what: str) -> pl.DataFrame:
    table = pl.read_parquet(path)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{what} ({path}) is missing column(s) {missing}")
    return table.select(columns)


def load_test_report(path: Path) -> dict[str, Any]:
    return _read_report(path, TEST_REPORT_KEYS, "The test evaluation report")


def load_ab_report(path: Path) -> dict[str, Any]:
    return _read_report(path, AB_REPORT_KEYS, "The A/B report")


def load_user_table(path: Path) -> UserTable:
    return UserTable(_read_table(path, OUTCOME_COLUMNS, "The A/B outcomes table"))


def load_session_videos(path: Path) -> pl.DataFrame:
    return _read_table(path, SESSION_VIDEO_COLUMNS, "The session videos table")


def load_users(path: Path) -> pl.DataFrame:
    return _read_table(path, USER_COLUMNS, "The user profiles table")


def leaderboard(report: dict[str, Any], metric: str) -> pl.DataFrame:
    """Every model's ``metric`` with its 95% interval (when it has one), best first."""
    rows = [
        {
            "model": name,
            "value": summary[metric],
            "ci_low": summary.get(f"{metric}_ci_low", summary[metric]),
            "ci_high": summary.get(f"{metric}_ci_high", summary[metric]),
        }
        for name, summary in report["models"].items()
    ]
    return pl.DataFrame(rows).sort("value", descending=True)


def paired_comparisons(report: dict[str, Any], reference: str) -> pl.DataFrame:
    """Every other model minus ``reference`` on the same users, with the share it wins."""
    differences = report["paired"][reference]["differences"]
    return pl.DataFrame(
        [
            {
                "model": name,
                "difference": difference["mean"],
                "ci_low": difference["ci_low"],
                "ci_high": difference["ci_high"],
                "win_rate": difference["win_rate"],
            }
            for name, difference in differences.items()
        ]
    ).sort("difference", descending=True)


def user_session(videos: pl.DataFrame, user_id: int, policy: str) -> pl.DataFrame:
    """The videos ``policy`` showed ``user_id``, in order, and what the user did with each."""
    return (
        videos.filter((pl.col("user_id") == user_id) & (pl.col("policy") == policy))
        .sort("position")
        .select(SESSION_COLUMNS)
    )


def session_totals(videos: pl.DataFrame, user_id: int) -> pl.DataFrame:
    """Per policy: how many of its videos ``user_id`` liked and how long they watched."""
    rank = pl.col("policy").replace_strict(
        {policy: index for index, policy in enumerate(POLICY_ORDER)},
        default=len(POLICY_ORDER),  # a policy the config does not know goes last
        return_dtype=pl.Int64,
    )
    return (
        videos.filter(pl.col("user_id") == user_id)
        .group_by("policy")
        .agg(liked=pl.col("liked").sum(), watched_s=pl.col("watched_s").sum())
        .sort(rank, "policy")
    )

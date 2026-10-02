"""Streamlit glue shared by the demo pages: cached loading, notices and guarded simulations.

Cached functions take a file's path *and* a version (its modification time and size), so a page
picks up a re-run of the pipeline without restarting the app. A/B lab results are cached per
setting, so moving a slider back to an earlier value is instant.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from recsys.demo import data, lab
from recsys.demo.data import Artifact, DemoPaths
from recsys.experiments.ab_test import UserTable

CACHED_FILES = 4  # versions of each file kept in memory
CACHED_RESULTS = 64  # lab results kept per function


def stamp(path: Path) -> tuple[str, str]:
    """A cache key for a file: its path and a version that changes whenever the file does."""
    stat = path.stat()
    return str(path), f"{stat.st_mtime_ns}-{stat.st_size}"


def demo_paths() -> DemoPaths:
    return DemoPaths.from_root(data.find_root())


def status(paths: DemoPaths) -> dict[str, Artifact]:
    return data.artifacts(paths)


def require(artifact: Artifact) -> bool:
    """True when ``artifact`` exists; otherwise tell the visitor how to build it."""
    if artifact.available:
        return True
    st.info(
        f"**{artifact.label}** is not built yet. Run `{artifact.command}` in the repository, "
        "then reload this page. On a fresh clone this first downloads KuaiRec and runs the "
        "pipeline (about half an hour on a laptop; see the README's quickstart)."
    )
    return False


def guarded(compute: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any] | None:
    """``compute(*args)``, or a warning instead of a traceback when the settings are ones the
    simulation cannot handle (e.g. so few users in an arm that a t-test is undefined)."""
    try:
        return compute(*args)
    except ValueError as error:
        st.warning(f"These settings cannot be simulated: {error}")
        return None


@st.cache_data(show_spinner=False, max_entries=CACHED_FILES)
def _test_report(path: str, version: str) -> dict[str, Any]:
    return data.load_test_report(Path(path))


@st.cache_data(show_spinner=False, max_entries=CACHED_FILES)
def _ab_report(path: str, version: str) -> dict[str, Any]:
    return data.load_ab_report(Path(path))


@st.cache_data(show_spinner=False, max_entries=CACHED_FILES)
def _session_videos(path: str, version: str) -> pl.DataFrame:
    return data.load_session_videos(Path(path))


@st.cache_data(show_spinner=False, max_entries=CACHED_FILES)
def _users(path: str, version: str) -> pl.DataFrame:
    return data.load_users(Path(path))


@st.cache_resource(show_spinner=False, max_entries=1)  # one table in memory at a time
def _user_table(path: str, version: str) -> UserTable:
    return data.load_user_table(Path(path))


def test_report(path: Path) -> dict[str, Any]:
    return _test_report(*stamp(path))


def ab_report(path: Path) -> dict[str, Any]:
    return _ab_report(*stamp(path))


def session_videos(path: Path) -> pl.DataFrame:
    return _session_videos(*stamp(path))


def users(path: Path) -> pl.DataFrame:
    return _users(*stamp(path))


# The A/B lab: ``outcomes`` and ``version`` come from ``stamp(paths.user_outcomes)``.


@st.cache_data(show_spinner="Running the experiment...", max_entries=CACHED_RESULTS)
def experiment(
    outcomes: str,
    version: str,
    control: str,
    treatment: str,
    salt: str,
    treatment_share: float,
    alpha: float,
) -> dict[str, Any]:
    table = _user_table(outcomes, version)
    return lab.run_experiment(table, control, treatment, salt, treatment_share, alpha)


@st.cache_data(show_spinner=False, max_entries=CACHED_RESULTS)
def plan(
    outcomes: str,
    version: str,
    control: str,
    treatment: str,
    n_users: int,
    treatment_share: float,
    alpha: float,
    power: float,
) -> dict[str, Any]:
    table = _user_table(outcomes, version)
    return lab.plan(table, control, treatment, n_users, treatment_share, alpha, power)


@st.cache_data(
    show_spinner="Re-running the experiment with fresh random splits...",
    max_entries=CACHED_RESULTS,
)
def rerandomise(
    outcomes: str,
    version: str,
    control: str,
    treatment: str,
    n_sims: int,
    treatment_share: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    table = _user_table(outcomes, version)
    return lab.rerandomise(table, control, treatment, n_sims, treatment_share, alpha, seed)


@st.cache_data(
    show_spinner="Simulating experiments that are checked as users arrive...",
    max_entries=CACHED_RESULTS,
)
def peek(
    outcomes: str,
    version: str,
    control: str,
    treatment: str,
    looks: int,
    n_sims: int,
    treatment_share: float,
    alpha: float,
    tau: float,
    seed: int,
) -> dict[str, Any]:
    table = _user_table(outcomes, version)
    return lab.peek(table, control, treatment, looks, n_sims, treatment_share, alpha, tau, seed)


@st.cache_data(show_spinner=False, max_entries=CACHED_RESULTS)
def logging_bug(
    outcomes: str,
    version: str,
    policy: str,
    salt: str,
    treatment_share: float,
    drop_share: float,
    alpha: float,
) -> dict[str, Any]:
    table = _user_table(outcomes, version)
    return lab.logging_bug(table, policy, salt, treatment_share, drop_share, alpha)

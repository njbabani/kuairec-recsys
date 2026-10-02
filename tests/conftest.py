"""Shared fixtures for the whole suite."""

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
import polars as pl
import pytest

# PyTorch and LightGBM each ship an OpenMP runtime; on macOS they only coexist in one process if
# PyTorch's is loaded first (LightGBM then runs single-threaded, see
# recsys.reranking.ranker.openmp_threads). The pipeline keeps them in separate processes; the
# test suite cannot, so it loads PyTorch before any test module can import LightGBM.
import torch  # noqa: F401  (imported for its side effect: load order)

from recsys.data.schemas import EVENT_TIME_DTYPE

# Headless rendering for every test that draws a figure (must run before pyplot is imported).
matplotlib.use("Agg")

SHANGHAI = ZoneInfo("Asia/Shanghai")


INTERACTION_DEFAULTS = {
    "user_id": 0,
    "video_id": 10,
    "play_duration": 5000,
    "video_duration": 10000,
    "watch_ratio": 0.5,
    "event_time": datetime(2020, 7, 5, tzinfo=SHANGHAI),
}


@pytest.fixture
def make_interactions() -> Callable[..., pl.DataFrame]:
    """Build a cleaned-interactions frame: pass per-row lists, omitted columns use defaults."""

    def _make(**columns: list) -> pl.DataFrame:
        n_rows = max((len(values) for values in columns.values()), default=1)
        data = {
            name: columns.get(name, [default] * n_rows)
            for name, default in INTERACTION_DEFAULTS.items()
        }
        return pl.DataFrame(data).with_columns(pl.col("event_time").cast(EVENT_TIME_DTYPE))

    return _make

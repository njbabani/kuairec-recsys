from pathlib import Path

import polars as pl
import pytest

from recsys.config import EvaluationParams


def _likes(user: int, video: int) -> bool:
    """Two taste groups: users 0-2 like videos 10-12, users 3-5 like videos 13-15."""
    return (user < 3) == (video < 13)


@pytest.fixture
def evaluation_params() -> EvaluationParams:
    return EvaluationParams(ks=(2,), bootstrap_samples=50, seed=0, select_metric="ndcg_at_2")


@pytest.fixture
def labeled_splits(tmp_path: Path, make_interactions) -> Path:
    """Write tiny labeled train/tune splits with learnable tastes; return the splits dir."""

    def labeled(pairs: list[tuple[int, int]]) -> pl.DataFrame:
        users, videos = zip(*pairs, strict=True)
        return make_interactions(user_id=list(users), video_id=list(videos)).with_columns(
            duration_bucket=pl.lit(0, dtype=pl.Int64),
            is_positive=pl.Series([_likes(u, v) for u, v in pairs]),
        )

    held_out = {(0, 10), (3, 13)}
    train = labeled([(u, v) for u in range(6) for v in range(10, 16) if (u, v) not in held_out])
    tune = labeled([(u, v) for u in (0, 3) for v in range(10, 16)])

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train.write_parquet(splits_dir / "train.parquet")
    tune.write_parquet(splits_dir / "tune.parquet")
    return splits_dir

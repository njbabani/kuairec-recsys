"""Score every pair the re-ranker needs with the saved two-tower ensemble.

Run as a DVC stage: ``python -m recsys.experiments.retrieval``.

This is the only stage after the leaderboard that loads PyTorch. It scores the unique
(user, video) pairs of the validation week (the re-ranker's training data) and of the tune and
test matrices (its evaluation candidates) and writes them to a table. Later stages read that
table instead of the model: offline candidate scoring, as a production two-stage system does,
and it keeps PyTorch's OpenMP runtime out of the LightGBM process.
"""

import logging
from pathlib import Path

import polars as pl

from recsys.config import load_params
from recsys.io import PARQUET_COMPRESSION
from recsys.models.two_tower.ensemble import TwoTowerEnsemble

logger = logging.getLogger(__name__)

SCORED_SPLITS = ("valid", "tune", "test")


def retrieval_pairs(splits_dir: Path) -> pl.DataFrame:
    """Unique (user, video) pairs across the scored splits, in a stable order."""
    return (
        pl.concat(
            [
                pl.read_parquet(splits_dir / f"{name}.parquet", columns=["user_id", "video_id"])
                for name in SCORED_SPLITS
            ]
        )
        .unique()
        .sort("user_id", "video_id")
    )


def run_retrieval(
    splits_dir: Path, model_dir: Path, output_path: Path, device: str
) -> pl.DataFrame:
    pairs = retrieval_pairs(splits_dir)
    ensemble = TwoTowerEnsemble.load(model_dir, device=device)
    scores = pairs.with_columns(score=ensemble.score(pairs))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(output_path, compression=PARQUET_COMPRESSION)
    logger.info(
        "Scored %s pairs with %d two-tower members", f"{scores.height:,}", len(ensemble.members)
    )
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    run_retrieval(
        params.data.splits_dir,
        params.data.models_dir / "two_tower",
        params.data.retrieval_path,
        device=params.models.two_tower.device,
    )


if __name__ == "__main__":
    main()

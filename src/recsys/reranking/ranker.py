"""The LightGBM LambdaRank re-ranker: training data, training, saving and explaining.

LambdaRank optimises NDCG directly: within each user's group of candidates it pushes liked
videos above the rest, weighting each swap by how much it would change NDCG. Models are saved
in LightGBM's own text format (no pickle), and explained with TreeSHAP: LightGBM's
``pred_contrib`` gives each feature's exact additive contribution to every prediction.
"""

import functools
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from recsys.checkpointing import atomic_write
from recsys.reranking.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

BIAS_COLUMN = "bias"
ALL_CORES = 0  # LightGBM's default: one OpenMP thread per core


def openmp_threads(platform: str = sys.platform, modules: Mapping[str, Any] = sys.modules) -> int:
    """Threads LightGBM may use here.

    PyTorch and LightGBM each ship an OpenMP runtime. On macOS two runtimes in one process crash
    once both start thread pools, so next to PyTorch LightGBM uses a single thread. (The
    pipeline avoids this altogether: the reranker stage reads precomputed two-tower scores
    and never imports PyTorch.)
    """
    if platform == "darwin" and "torch" in modules:
        _warn_single_threaded()
        return 1
    return ALL_CORES


@functools.cache
def _warn_single_threaded() -> None:
    logger.warning("PyTorch is loaded: running LightGBM on one thread to avoid an OpenMP clash")


@dataclass(frozen=True)
class RankingData:
    """Feature rows sorted so each user's candidates are contiguous (one LightGBM group)."""

    features: pl.DataFrame  # FEATURE_COLUMNS only
    labels: np.ndarray
    groups: np.ndarray  # rows per user, in row order


def ranking_data(features: pl.DataFrame, labels: pl.Series) -> RankingData:
    frame = features.with_columns(label=labels.alias("label")).sort("user_id", maintain_order=True)
    groups = frame.group_by("user_id", maintain_order=True).len()["len"].to_numpy()
    return RankingData(frame.select(FEATURE_COLUMNS), frame["label"].to_numpy(), groups)


@dataclass(frozen=True)
class RankerConfig:
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    feature_fraction: float
    max_rounds: int
    early_stopping_rounds: int
    eval_at: int
    seed: int

    def lightgbm_params(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [self.eval_at],
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_child_samples,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "seed": self.seed,
            "num_threads": openmp_threads(),
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


@dataclass(frozen=True)
class TrainedRanker:
    booster: lgb.Booster
    best_iteration: int
    holdout_score: float  # NDCG@eval_at on the held-out users at the best iteration
    curve: list[float]  # holdout NDCG@eval_at after every boosting round


def _matrix(features: pl.DataFrame) -> np.ndarray:
    """Float matrix in FEATURE_COLUMNS order; missing values become NaN (LightGBM's missing)."""
    return features.select(pl.col(FEATURE_COLUMNS).cast(pl.Float64)).to_numpy()


def _dataset(data: RankingData, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(
        _matrix(data.features),
        label=data.labels,
        group=data.groups,
        feature_name=list(FEATURE_COLUMNS),
        categorical_feature=list(CATEGORICAL_COLUMNS),
        reference=reference,
        free_raw_data=False,
    )


def train_ranker(train: RankingData, holdout: RankingData, config: RankerConfig) -> TrainedRanker:
    """Boost until holdout NDCG stops improving for ``early_stopping_rounds`` rounds."""
    train_set = _dataset(train)
    history: dict = {}
    booster = lgb.train(
        config.lightgbm_params(),
        train_set,
        num_boost_round=config.max_rounds,
        valid_sets=[_dataset(holdout, reference=train_set)],
        valid_names=["holdout"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.record_evaluation(history),
        ],
    )
    curve = [float(score) for score in history["holdout"][f"ndcg@{config.eval_at}"]]
    best = booster.best_iteration or len(curve)
    return TrainedRanker(booster, best, curve[best - 1], curve)


def predict(booster: lgb.Booster, features: pl.DataFrame) -> np.ndarray:
    return booster.predict(
        _matrix(features),
        num_iteration=booster.best_iteration or None,
        num_threads=openmp_threads(),
    )


def contributions(booster: lgb.Booster, features: pl.DataFrame) -> pl.DataFrame:
    """TreeSHAP values: one column per feature plus the bias; each row sums to the prediction."""
    values = booster.predict(
        _matrix(features),
        num_iteration=booster.best_iteration or None,
        pred_contrib=True,
        num_threads=openmp_threads(),
    )
    return pl.DataFrame(values, schema=[*FEATURE_COLUMNS, BIAS_COLUMN], orient="row")


def train_fixed_rounds(data: RankingData, config: RankerConfig, rounds: int) -> lgb.Booster:
    """Boost exactly ``rounds`` times, with no holdout (e.g. to reuse a chosen round count)."""
    return lgb.train(config.lightgbm_params(), _dataset(data), num_boost_round=rounds)


def save_ranker(booster: lgb.Booster, path: Path) -> None:
    """Save only the trees up to the best iteration, atomically, in LightGBM's text format."""
    best = booster.best_iteration or None
    atomic_write(path, lambda partial: booster.save_model(str(partial), num_iteration=best))


def load_ranker(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))

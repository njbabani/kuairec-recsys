from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from recsys.config import (
    ALSParams,
    CategoryAffinityParams,
    EngagementPopularityParams,
    EvaluationParams,
    ModelParams,
    RandomParams,
    RerankerParams,
    TwoTowerEnsembleParams,
)
from recsys.experiments.reranker import InputPaths, RerankerPaths

CUTOFF = date(2020, 8, 30)
VALID_USERS = range(6, 46)
EVALUATION_USERS = (0, 3, 100)  # tune users 0 and 3, test user 100: never used to train
VIDEOS = range(10, 16)


def valid_likes(user: int, video: int) -> bool:
    return (user % 2 == 0) == (video < 13)


TOY_MODELS = ModelParams(
    random=RandomParams(seed=0),
    popularity_engagement=EngagementPopularityParams(prior_strength=1.0),
    category_affinity=CategoryAffinityParams(prior_strength=1.0),
    als=ALSParams(factors=2, regularization=0.01, alpha=10.0, iterations=10, seed=0),
    two_tower=TwoTowerEnsembleParams(  # unused here: two-tower scores come precomputed
        loss="bce",
        embedding_dim=4,
        hidden_dim=8,
        epochs=1,
        batch_size=8,
        learning_rate=0.05,
        weight_decay=0.0,
        seeds=(0,),
        device="cpu",
    ),
)
TOY_RERANKER_PARAMS = RerankerParams(
    retrieve_top_n=3,
    holdout_user_share=0.25,
    report_user_share=0.25,
    holdout_salt="test-holdout",
    cross_fit_folds=2,
    num_leaves=(3,),
    min_child_samples=(1, 3),
    learning_rate=0.3,
    feature_fraction=1.0,
    max_rounds=15,
    early_stopping_rounds=5,
    eval_at=2,
    prior_strength=2.0,
    shap_sample_rows=5,
    seed=0,
)


@dataclass(frozen=True)
class ToyWorld:
    """Facts and settings of the synthetic world that tests share."""

    cutoff: date = CUTOFF
    valid_users: range = VALID_USERS
    evaluation_users: tuple[int, ...] = EVALUATION_USERS
    models: ModelParams = TOY_MODELS
    reranker_params: RerankerParams = TOY_RERANKER_PARAMS


@pytest.fixture
def toy_world() -> ToyWorld:
    return ToyWorld()


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


@pytest.fixture
def feature_tables(tmp_path: Path) -> Path:
    """User and video features for the users (0-5) and videos (10-15) of ``labeled_splits``."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    pl.DataFrame(
        {
            "user_id": list(range(6)),
            "user_active_degree": ["high_active"] * 3 + ["low_active"] * 3,
            "log_register_days": [float(user) for user in range(6)],
        }
    ).write_parquet(features_dir / "users.parquet")
    pl.DataFrame(
        {
            "video_id": list(range(10, 16)),
            "category_ids": [[1]] * 3 + [[2]] * 3,
            "log_duration_s": [1.0, 2.0, 3.0] * 2,
        }
    ).write_parquet(features_dir / "videos.parquet")
    return features_dir


@pytest.fixture
def reranker_world(tmp_path, labeled_splits, feature_tables, make_interactions) -> RerankerPaths:
    """Validation week, test users, platform stats and precomputed two-tower scores for the toy
    splits, with the reranker stage's paths (shared by the reranker and test-policy tests)."""
    pairs = [(u, v) for u in (*VALID_USERS, *EVALUATION_USERS) for v in VIDEOS]
    users, videos = zip(*pairs, strict=True)
    make_interactions(user_id=list(users), video_id=list(videos)).with_columns(
        duration_bucket=pl.lit(0, dtype=pl.Int64),
        is_positive=pl.Series([valid_likes(u, v) for u, v in pairs]),
    ).write_parquet(labeled_splits / "valid.parquet")
    tune = pl.read_parquet(labeled_splits / "tune.parquet")
    tune.with_columns(user_id=pl.lit(100, pl.Int64)).unique(["user_id", "video_id"]).write_parquet(
        labeled_splits / "test.parquet"
    )

    processed = tmp_path / "processed"
    processed.mkdir()
    pl.DataFrame(
        {
            "video_id": list(VIDEOS),
            "date": [date(2020, 8, 1)] * 6,
            "upload_dt": ["2020-07-01"] * 6,
            **{
                count: [100 * (v - 9) for v in VIDEOS]
                for count in ("play_cnt", "like_cnt", "complete_play_cnt")
            },
            **{count: [1] * 6 for count in ("share_cnt", "comment_cnt", "follow_cnt")},
        }
    ).write_parquet(processed / "video_daily_stats.parquet")

    scored = [
        pl.read_parquet(labeled_splits / f"{name}.parquet", columns=["user_id", "video_id"])
        for name in ("tune", "test")
    ]
    needed = pl.concat([pl.DataFrame({"user_id": users, "video_id": videos}), *scored]).unique()
    retrieval = tmp_path / "retrieval.parquet"
    needed.with_columns(score=(pl.col("video_id") % 3).cast(pl.Float64)).write_parquet(retrieval)

    return RerankerPaths(
        inputs=InputPaths(
            splits_dir=labeled_splits,
            features_dir=feature_tables,
            processed_dir=processed,
            retrieval_path=retrieval,
        ),
        metrics_path=tmp_path / "reranker.json",
        model_dir=tmp_path / "models" / "reranker",
        checkpoint_dir=tmp_path / "checkpoints" / "reranker",
        shap_path=tmp_path / "shap" / "shap_sample.parquet",
    )

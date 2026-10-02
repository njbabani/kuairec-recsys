"""Typed, validated access to params.yaml — the single source of truth for pipeline settings."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

DEFAULT_PARAMS_PATH = Path("params.yaml")
RANKING_METRICS = ("precision", "recall", "ndcg", "map")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataParams(_FrozenModel):
    url: HttpUrl
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    archive_path: Path
    processed_dir: Path
    summary_path: Path
    splits_dir: Path
    split_summary_path: Path
    metrics_dir: Path
    features_dir: Path
    checkpoints_dir: Path  # resumable training state (not versioned; deleted once a stage succeeds)
    models_dir: Path  # trained models written by the leaderboard (versioned by DVC)
    retrieval_path: Path  # two-tower scores of every pair the re-ranker needs
    reranker_dir: Path  # re-ranker outputs too large for git (SHAP sample)


class LabelParams(_FrozenModel):
    bucket_width_ratio: float = Field(gt=1)
    min_views_per_bucket: int = Field(ge=1)
    positive_quantile: float = Field(gt=0, lt=1)


class SplitParams(_FrozenModel):
    valid_start: date
    tune_user_share: float = Field(gt=0, lt=1)
    assignment_salt: str = Field(min_length=1)


class EvaluationParams(_FrozenModel):
    ks: tuple[PositiveInt, ...] = Field(min_length=1)
    bootstrap_samples: int = Field(ge=10)
    seed: int
    select_metric: str

    @model_validator(mode="after")
    def _select_metric_is_evaluated(self) -> "EvaluationParams":
        if len(set(self.ks)) != len(self.ks):
            raise ValueError(f"ks must be distinct, got {self.ks}")
        evaluated = {f"{metric}_at_{k}" for metric in RANKING_METRICS for k in self.ks}
        if self.select_metric not in evaluated:
            raise ValueError(f"select_metric must be one of {sorted(evaluated)}")
        return self


class RandomParams(_FrozenModel):
    seed: int


class EngagementPopularityParams(_FrozenModel):
    prior_strength: NonNegativeFloat


class ALSParams(_FrozenModel):
    factors: PositiveInt
    regularization: NonNegativeFloat
    alpha: NonNegativeFloat  # confidence 1 + alpha * count; 0 = plain binary matrix
    iterations: PositiveInt
    seed: int


class CategoryAffinityParams(_FrozenModel):
    # Pseudo-views pulling a user's per-category positive rate towards the category's rate
    prior_strength: PositiveFloat


TwoTowerLoss = Literal["bce", "softmax"]
Device = Literal["auto", "cpu", "mps"]


def _require_distinct(values: tuple, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be distinct, got {values}")


class _TwoTowerTraining(_FrozenModel):
    """Settings shared by single two-tower models, seed ensembles and the search."""

    hidden_dim: PositiveInt
    epochs: PositiveInt
    batch_size: int = Field(ge=2)  # the softmax loss needs other rows as negatives
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    # cpu is deterministic (and as fast as MPS for a model this small); auto = MPS when present
    device: Device


class _TwoTowerArchitecture(_TwoTowerTraining):
    loss: TwoTowerLoss
    embedding_dim: PositiveInt
    temperature: PositiveFloat | None = None  # softmax loss only

    @model_validator(mode="after")
    def _temperature_only_for_softmax(self) -> Self:
        if (self.loss == "softmax") != (self.temperature is not None):
            raise ValueError("temperature must be set for the softmax loss, and only for it")
        return self


class TwoTowerParams(_TwoTowerArchitecture):
    """One two-tower model."""

    seed: int


class TwoTowerEnsembleParams(_TwoTowerArchitecture):
    """One two-tower model per seed; the leaderboard scores their average."""

    seeds: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _seeds_are_distinct(self) -> Self:
        _require_distinct(self.seeds, "seeds")
        return self

    def member(self, seed: int) -> TwoTowerParams:
        return TwoTowerParams(**self.model_dump(exclude={"seeds"}), seed=seed)


class ModelParams(_FrozenModel):
    random: RandomParams
    popularity_engagement: EngagementPopularityParams
    category_affinity: CategoryAffinityParams
    als: ALSParams
    two_tower: TwoTowerEnsembleParams


class ALSSearchParams(_FrozenModel):
    factors: tuple[PositiveInt, ...] = Field(min_length=1)
    regularization: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    alpha: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    iterations: PositiveInt
    seed: int


class TwoTowerSearchParams(_TwoTowerTraining):
    """Grid over loss x embedding size (x temperature, for the softmax loss only).

    Every configuration is trained once per seed and scored after each epoch, so the epoch
    count is searched too; selection uses the average over seeds, not one lucky run.
    """

    loss: tuple[TwoTowerLoss, ...] = Field(min_length=1)
    embedding_dim: tuple[PositiveInt, ...] = Field(min_length=1)
    temperature: tuple[PositiveFloat, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _grid_values_are_distinct(self) -> Self:
        for name in ("loss", "embedding_dim", "temperature", "seeds"):
            _require_distinct(getattr(self, name), name)
        return self


class RerankerParams(_FrozenModel):
    """The LightGBM re-ranker: trained on the validation week, tuned on held-out users of it."""

    retrieve_top_n: PositiveInt  # candidates per user the two-tower model passes on
    holdout_user_share: float = Field(gt=0, lt=1)  # validation users for early stopping/grid
    report_user_share: float = Field(gt=0, lt=1)  # validation users only for the logged check
    holdout_salt: str = Field(min_length=1)
    cross_fit_folds: int = Field(ge=2)  # folds of the fully observed upper-bound check
    num_leaves: tuple[PositiveInt, ...] = Field(min_length=1)
    min_child_samples: tuple[PositiveInt, ...] = Field(min_length=1)
    learning_rate: PositiveFloat
    feature_fraction: float = Field(gt=0, le=1)
    max_rounds: PositiveInt
    early_stopping_rounds: PositiveInt
    eval_at: PositiveInt
    prior_strength: PositiveFloat  # shrinkage of per-user length rates (a feature)
    shap_sample_rows: PositiveInt
    seed: int

    @model_validator(mode="after")
    def _grid_values_are_distinct(self) -> Self:
        for name in ("num_leaves", "min_child_samples"):
            _require_distinct(getattr(self, name), name)
        if self.holdout_user_share + self.report_user_share >= 1:
            raise ValueError("holdout and report user shares must leave users to fit on")
        return self


class TrackingParams(_FrozenModel):
    backends: tuple[Literal["mlflow", "wandb"], ...]
    experiment: str = Field(min_length=1)
    mlflow_tracking_uri: str = Field(min_length=1)
    wandb_project: str = Field(min_length=1)
    wandb_mode: Literal["online", "offline", "disabled"]

    @model_validator(mode="after")
    def _backends_are_distinct(self) -> "TrackingParams":
        if len(set(self.backends)) != len(self.backends):
            raise ValueError(f"backends must be distinct, got {self.backends}")
        return self


class Params(_FrozenModel):
    data: DataParams
    label: LabelParams
    split: SplitParams
    evaluation: EvaluationParams
    models: ModelParams
    als_search: ALSSearchParams
    two_tower_search: TwoTowerSearchParams
    reranker: RerankerParams
    tracking: TrackingParams


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> Params:
    """Load and validate pipeline parameters, failing fast on missing, empty or malformed files."""
    if not path.is_file():
        raise FileNotFoundError(f"Params file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raise ValueError(f"Params file is empty: {path}")

    return Params.model_validate(raw)

"""Typed, validated access to params.yaml — the single source of truth for pipeline settings."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
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


class BaselineParams(_FrozenModel):
    random: RandomParams
    popularity_engagement: EngagementPopularityParams
    als: ALSParams


class ALSSearchParams(_FrozenModel):
    factors: tuple[PositiveInt, ...] = Field(min_length=1)
    regularization: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    alpha: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    iterations: PositiveInt
    seed: int


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
    baselines: BaselineParams
    als_search: ALSSearchParams
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

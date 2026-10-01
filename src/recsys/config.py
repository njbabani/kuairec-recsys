"""Typed, validated access to params.yaml — the single source of truth for pipeline settings."""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

DEFAULT_PARAMS_PATH = Path("params.yaml")


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


class LabelParams(_FrozenModel):
    bucket_width_ratio: float = Field(gt=1)
    min_views_per_bucket: int = Field(ge=1)
    positive_quantile: float = Field(gt=0, lt=1)


class SplitParams(_FrozenModel):
    valid_start: date
    tune_user_share: float = Field(gt=0, lt=1)
    assignment_salt: str = Field(min_length=1)


class Params(_FrozenModel):
    data: DataParams
    label: LabelParams
    split: SplitParams


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

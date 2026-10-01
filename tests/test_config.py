from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from recsys.config import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_PARAMS = """
data:
  url: https://example.org/KuaiRec.zip
  md5: 0123456789abcdef0123456789abcdef
  archive_path: data/raw/KuaiRec.zip
  processed_dir: data/processed
  summary_path: reports/data_summary.json
  splits_dir: data/splits
  split_summary_path: reports/split_summary.json
  metrics_dir: reports/metrics
label:
  bucket_width_ratio: 1.15
  min_views_per_bucket: 10000
  positive_quantile: 0.8
split:
  valid_start: "2020-08-30"
  tune_user_share: 0.2
  assignment_salt: kuairec-tune-v1
evaluation:
  ks: [10, 50]
  bootstrap_samples: 1000
  seed: 42
  select_metric: ndcg_at_10
baselines:
  random:
    seed: 42
  popularity_engagement:
    prior_strength: 50
  als:
    factors: 64
    regularization: 0.05
    alpha: 20
    iterations: 15
    seed: 42
als_search:
  factors: [32, 64]
  regularization: [0.01, 0.1]
  alpha: [5, 20]
  iterations: 15
  seed: 42
tracking:
  backends: [mlflow, wandb]
  experiment: kuairec-recsys
  mlflow_tracking_uri: sqlite:///mlflow.db
  wandb_project: kuairec-recsys
  wandb_mode: offline
"""


def write_params(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text)
    return path


def test_load_params_parses_data_section(tmp_path):
    params = load_params(write_params(tmp_path, VALID_PARAMS))

    assert str(params.data.url) == "https://example.org/KuaiRec.zip"
    assert params.data.md5 == "0123456789abcdef0123456789abcdef"
    assert params.data.archive_path == Path("data/raw/KuaiRec.zip")
    assert params.data.processed_dir == Path("data/processed")


def test_load_params_parses_label_and_split_sections(tmp_path):
    params = load_params(write_params(tmp_path, VALID_PARAMS))

    assert params.label.bucket_width_ratio == 1.15
    assert params.label.min_views_per_bucket == 10000
    assert params.label.positive_quantile == 0.8
    assert params.split.valid_start == date(2020, 8, 30)
    assert params.split.tune_user_share == 0.2
    assert params.split.assignment_salt == "kuairec-tune-v1"
    assert params.data.splits_dir == Path("data/splits")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("positive_quantile: 0.8", "positive_quantile: 1.0"),
        ("bucket_width_ratio: 1.15", "bucket_width_ratio: 1.0"),
        ("min_views_per_bucket: 10000", "min_views_per_bucket: 0"),
        ('valid_start: "2020-08-30"', 'valid_start: "not-a-date"'),
        ("tune_user_share: 0.2", "tune_user_share: 1.0"),
        ("assignment_salt: kuairec-tune-v1", 'assignment_salt: ""'),
    ],
    ids=["quantile", "width-ratio", "min-views", "bad-date", "tune-share", "empty-salt"],
)
def test_load_params_rejects_invalid_label_and_split_settings(tmp_path, old, new):
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, VALID_PARAMS.replace(old, new)))


def test_load_params_parses_modelling_and_tracking_sections(tmp_path):
    params = load_params(write_params(tmp_path, VALID_PARAMS))

    assert params.evaluation.ks == (10, 50)
    assert params.baselines.als.factors == 64
    assert params.als_search.alpha == (5.0, 20.0)
    assert params.tracking.backends == ("mlflow", "wandb")
    assert params.tracking.wandb_mode == "offline"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("select_metric: ndcg_at_10", "select_metric: ndcg_at_7"),
        ("select_metric: ndcg_at_10", "select_metric: auc"),
        ("backends: [mlflow, wandb]", "backends: [mlflow, tensorboard]"),
        ("wandb_mode: offline", "wandb_mode: sometimes"),
        ("factors: [32, 64]", "factors: []"),
        ("ks: [10, 50]", "ks: [10, 10]"),
        ("backends: [mlflow, wandb]", "backends: [mlflow, mlflow]"),
    ],
    ids=[
        "metric-k-not-evaluated",
        "unknown-metric",
        "unknown-backend",
        "bad-mode",
        "empty-grid",
        "duplicate-k",
        "duplicate-backend",
    ],
)
def test_load_params_rejects_invalid_modelling_settings(tmp_path, old, new):
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, VALID_PARAMS.replace(old, new)))


def test_load_params_rejects_malformed_md5(tmp_path):
    text = VALID_PARAMS.replace("0123456789abcdef0123456789abcdef", "not-a-hash")

    with pytest.raises(ValidationError, match="md5"):
        load_params(write_params(tmp_path, text))


def test_load_params_rejects_unknown_keys_to_catch_typos(tmp_path):
    text = VALID_PARAMS.replace("data:\n", "data:\n  procesed_dir: typo\n")

    with pytest.raises(ValidationError, match="procesed_dir"):
        load_params(write_params(tmp_path, text))


def test_load_params_raises_clear_error_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"missing\.yaml"):
        load_params(tmp_path / "missing.yaml")


def test_load_params_raises_clear_error_when_file_empty(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_params(write_params(tmp_path, ""))


def test_load_params_reports_path_when_yaml_is_malformed(tmp_path):
    with pytest.raises(ValueError, match=r"Invalid YAML in .*params\.yaml"):
        load_params(write_params(tmp_path, "data: [unclosed"))


def test_params_are_immutable(tmp_path):
    params = load_params(write_params(tmp_path, VALID_PARAMS))

    with pytest.raises(ValidationError):
        params.data.md5 = "ffffffffffffffffffffffffffffffff"


def test_repo_params_use_only_types_dvc_can_track():
    # DVC rejects YAML-native dates/timestamps in params files (dates must be quoted strings).
    def scalars(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from scalars(value)
        elif isinstance(node, list):
            for value in node:
                yield from scalars(value)
        else:
            yield node

    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))

    assert all(isinstance(value, str | int | float | bool) for value in scalars(raw))


def test_repo_params_file_is_valid():
    params = load_params(REPO_ROOT / "params.yaml")

    assert params.data.archive_path.name == "KuaiRec.zip"
    # The repo quotes the date for DVC; pydantic must still parse it into a real date.
    assert isinstance(params.split.valid_start, date)

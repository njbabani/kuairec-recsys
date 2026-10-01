"""Experiment tracking: every run is logged to MLflow and/or Weights & Biases.

* **MLflow** keeps a local run history (``sqlite:///mlflow.db``); browse with ``make mlflow-ui``.
* **W&B** adds shareable dashboards and hyperparameter sweeps. It defaults to *offline* mode so
  the project runs without an account. To publish runs: ``wandb login`` (you, once), then set
  ``WANDB_MODE=online`` (or ``tracking.wandb_mode``), or upload past runs with
  ``wandb sync wandb/offline-run-*``.

Experiments depend only on the small :class:`Tracker` protocol, so tests use the in-memory one.
"""

import os
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any, Protocol

from recsys.config import TrackingParams


class TrackingRun(Protocol):
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...


class Tracker(Protocol):
    def run(self, name: str, params: Mapping[str, Any]) -> AbstractContextManager[TrackingRun]:
        """Open a run, log ``params`` once, and close it as FINISHED or FAILED."""
        ...


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"als": {"factors": 64}}`` -> ``{"als.factors": 64}`` (both backends want flat keys)."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat |= flatten_params(value, prefix=f"{name}.")
        else:
            flat[name] = value
    return flat


class _NoopRun:
    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        return None


class NoopTracker:
    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[TrackingRun]:
        yield _NoopRun()


class _InMemoryRun:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._record["metrics"] |= dict(metrics)


class InMemoryTracker:
    """Keeps runs in a list: used by tests, and handy for quick comparisons in notebooks."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[TrackingRun]:
        record: dict[str, Any] = {
            "name": name,
            "params": flatten_params(params),
            "metrics": {},
            "status": "RUNNING",
        }
        self.runs.append(record)
        try:
            yield _InMemoryRun(record)
        except BaseException:
            record["status"] = "FAILED"
            raise
        record["status"] = "FINISHED"


class _MultiRun:
    def __init__(self, runs: Sequence[TrackingRun]) -> None:
        self._runs = runs

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        for run in self._runs:
            run.log_metrics(metrics)


class MultiTracker:
    """Fans every call out to several trackers; each run is closed even if the body fails."""

    def __init__(self, trackers: Sequence[Tracker]) -> None:
        self.trackers = tuple(trackers)

    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[TrackingRun]:
        with ExitStack() as stack:
            yield _MultiRun([stack.enter_context(t.run(name, params)) for t in self.trackers])


class _MLflowRun:
    def __init__(self, client: Any, run_id: str) -> None:
        self._client = client
        self._run_id = run_id

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        from mlflow.entities import Metric

        timestamp_ms = int(time.time() * 1000)
        self._client.log_batch(
            self._run_id,
            metrics=[Metric(key, float(value), timestamp_ms, 0) for key, value in metrics.items()],
        )


class MLflowTracker:
    """Logs through an explicit ``MlflowClient`` (no global MLflow state is touched)."""

    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        artifact_root: Path,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.artifact_root = artifact_root
        self.tags = dict(tags or {})

    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[TrackingRun]:
        from mlflow.entities import Param
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                self.experiment, artifact_location=self.artifact_root.resolve().as_uri()
            )
        )
        run_id = client.create_run(experiment_id, run_name=name, tags=self.tags).info.run_id
        try:
            client.log_batch(
                run_id,
                params=[Param(key, str(value)) for key, value in flatten_params(params).items()],
            )
            yield _MLflowRun(client, run_id)
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id, status="FINISHED")


class _WandbRun:
    def __init__(self, run: Any) -> None:
        self._run = run

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._run.log(dict(metrics))


class WandbTracker:
    """Weights & Biases runs; the ``WANDB_MODE`` environment variable overrides ``mode``."""

    def __init__(self, project: str, mode: str, directory: Path) -> None:
        self.project = project
        self.mode = mode
        self.directory = directory

    @property
    def effective_mode(self) -> str:
        return os.environ.get("WANDB_MODE", self.mode)

    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[TrackingRun]:
        import wandb

        self.directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=self.project,
            name=name,
            config=flatten_params(params),
            mode=self.effective_mode,
            dir=str(self.directory),
        )
        try:
            yield _WandbRun(run)
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish()


def provenance_tags(workdir: Path) -> dict[str, str]:
    """The git commit (and whether the tree had uncommitted changes) behind a run."""
    git = ["git", "-C", str(workdir)]
    try:
        # Fixed git commands with no user input.
        sha = subprocess.run(  # noqa: S603
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "git_dirty": "unknown"}
    return {"git_sha": sha, "git_dirty": str(bool(status.strip())).lower()}


def build_tracker(params: TrackingParams, workdir: Path) -> Tracker:
    """The configured backends combined; no backends gives a tracker that records nothing."""
    trackers: list[Tracker] = []
    for backend in params.backends:
        if backend == "mlflow":
            trackers.append(
                MLflowTracker(
                    params.mlflow_tracking_uri,
                    params.experiment,
                    artifact_root=workdir / "mlartifacts",
                    tags=provenance_tags(workdir),
                )
            )
        elif backend == "wandb":
            trackers.append(WandbTracker(params.wandb_project, params.wandb_mode, workdir))
    return MultiTracker(trackers) if trackers else NoopTracker()

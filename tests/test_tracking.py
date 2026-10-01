from pathlib import Path

import pytest

from recsys.config import TrackingParams
from recsys.tracking import (
    InMemoryTracker,
    MLflowTracker,
    MultiTracker,
    NoopTracker,
    WandbTracker,
    build_tracker,
    flatten_params,
    provenance_tags,
)


@pytest.fixture(autouse=True)
def no_ambient_wandb_mode(monkeypatch):
    """A developer's exported WANDB_MODE must never make these tests touch the network."""
    monkeypatch.delenv("WANDB_MODE", raising=False)


class _FailsToStart:
    def run(self, name, params):
        raise RuntimeError("backend unavailable")


def test_multi_tracker_closes_already_started_runs_when_a_later_backend_fails():
    first = InMemoryTracker()

    with pytest.raises(RuntimeError, match="unavailable"):
        MultiTracker((first, _FailsToStart())).run("trial", {}).__enter__()

    assert [run["status"] for run in first.runs] == ["FAILED"]


def test_provenance_tags_record_the_git_commit(tmp_path):
    repo_tags = provenance_tags(Path(__file__).resolve().parents[1])

    assert len(repo_tags["git_sha"]) == 40
    assert repo_tags["git_dirty"] in {"true", "false"}
    assert provenance_tags(tmp_path) == {"git_sha": "unknown", "git_dirty": "unknown"}


def test_flatten_params_joins_nested_keys_with_dots():
    assert flatten_params({"als": {"factors": 64, "alpha": 20.0}, "model": "als"}) == {
        "als.factors": 64,
        "als.alpha": 20.0,
        "model": "als",
    }


def test_noop_tracker_accepts_runs_and_metrics():
    with NoopTracker().run("trial", {"a": 1}) as run:
        run.log_metrics({"ndcg_at_10": 0.5})
        run.log_metrics({"train_loss": 0.9}, step=1)


def test_in_memory_tracker_records_params_metrics_and_status():
    tracker = InMemoryTracker()

    with tracker.run("trial", {"als": {"factors": 8}}) as run:
        run.log_metrics({"ndcg_at_10": 0.5})

    assert tracker.runs == [
        {
            "name": "trial",
            "params": {"als.factors": 8},
            "metrics": {"ndcg_at_10": 0.5},
            "history": [],
            "status": "FINISHED",
        }
    ]


def test_in_memory_tracker_keeps_metrics_logged_per_step_and_their_latest_values():
    tracker = InMemoryTracker()

    with tracker.run("trial", {}) as run:
        run.log_metrics({"train_loss": 0.9}, step=1)
        run.log_metrics({"train_loss": 0.5}, step=2)

    assert tracker.runs[0]["history"] == [(1, {"train_loss": 0.9}), (2, {"train_loss": 0.5})]
    assert tracker.runs[0]["metrics"] == {"train_loss": 0.5}


def test_multi_tracker_forwards_the_step_to_every_backend():
    first, second = InMemoryTracker(), InMemoryTracker()

    with MultiTracker((first, second)).run("trial", {}) as run:
        run.log_metrics({"train_loss": 0.9}, step=3)

    assert first.runs[0]["history"] == second.runs[0]["history"] == [(3, {"train_loss": 0.9})]


def test_multi_tracker_fans_out_and_closes_every_run_even_on_error():
    first, second = InMemoryTracker(), InMemoryTracker()

    def crashing_trial() -> None:
        with MultiTracker((first, second)).run("trial", {}) as run:
            run.log_metrics({"ndcg_at_10": 0.5})
            raise RuntimeError("training crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        crashing_trial()

    for tracker in (first, second):
        assert [r["status"] for r in tracker.runs] == ["FAILED"]
        assert tracker.runs[0]["metrics"] == {"ndcg_at_10": 0.5}


@pytest.mark.integration
def test_mlflow_tracker_records_params_and_metrics(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(
        uri, experiment="test", artifact_root=tmp_path / "artifacts", tags={"git_sha": "abc"}
    )

    with tracker.run("trial", {"als": {"factors": 8}}) as run:
        run.log_metrics({"ndcg_at_10": 0.25})

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("test")
    [record] = client.search_runs([experiment.experiment_id])
    assert record.info.run_name == "trial"
    assert record.info.status == "FINISHED"
    assert record.data.params == {"als.factors": "8"}
    assert record.data.metrics == {"ndcg_at_10": 0.25}
    assert record.data.tags["git_sha"] == "abc"


@pytest.mark.integration
def test_mlflow_tracker_records_a_metric_history_per_step(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(uri, experiment="test", artifact_root=tmp_path / "artifacts")

    with tracker.run("trial", {}) as run:
        run.log_metrics({"train_loss": 0.9}, step=1)
        run.log_metrics({"train_loss": 0.5}, step=2)

    client = MlflowClient(tracking_uri=uri)
    [record] = client.search_runs([client.get_experiment_by_name("test").experiment_id])
    history = client.get_metric_history(record.info.run_id, "train_loss")
    assert [(m.step, m.value) for m in history] == [(1, 0.9), (2, 0.5)]


@pytest.mark.integration
def test_mlflow_run_is_marked_failed_even_when_logging_params_fails(tmp_path):
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(uri, experiment="test", artifact_root=tmp_path / "artifacts")

    def open_run_with_invalid_param() -> None:
        with tracker.run("trial", {"bad:key!": 1}):
            pass

    with pytest.raises(MlflowException):
        open_run_with_invalid_param()

    client = MlflowClient(tracking_uri=uri)
    [record] = client.search_runs([client.get_experiment_by_name("test").experiment_id])
    assert record.info.status == "FAILED"


def test_wandb_tracker_runs_in_disabled_mode_and_propagates_failures(tmp_path):
    tracker = WandbTracker(project="test", mode="disabled", directory=tmp_path)

    with tracker.run("trial", {"a": 1}) as run:
        run.log_metrics({"train_loss": 0.9}, step=1)
        run.log_metrics({"ndcg_at_10": 0.1})

    def crashing_trial() -> None:
        with tracker.run("trial", {}):
            raise RuntimeError("training crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        crashing_trial()


@pytest.mark.integration
def test_wandb_offline_mode_writes_a_local_run_that_can_be_synced_later(tmp_path):
    tracker = WandbTracker(project="test", mode="offline", directory=tmp_path)

    with tracker.run("trial", {"a": 1}) as run:
        run.log_metrics({"ndcg_at_10": 0.1})

    assert list((tmp_path / "wandb").glob("offline-run-*"))


def test_wandb_mode_environment_variable_overrides_the_configured_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    tracker = WandbTracker(project="test", mode="online", directory=tmp_path)

    assert tracker.effective_mode == "disabled"


def test_build_tracker_combines_the_configured_backends(tmp_path):
    params = TrackingParams(
        backends=("mlflow", "wandb"),
        experiment="e",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'm.db'}",
        wandb_project="p",
        wandb_mode="disabled",
    )

    tracker = build_tracker(params, workdir=tmp_path)

    assert isinstance(tracker, MultiTracker)
    assert [type(t) for t in tracker.trackers] == [MLflowTracker, WandbTracker]


def test_build_tracker_without_backends_is_a_noop(tmp_path):
    params = TrackingParams(
        backends=(),
        experiment="e",
        mlflow_tracking_uri="sqlite:///unused.db",
        wandb_project="p",
        wandb_mode="disabled",
    )

    assert isinstance(build_tracker(params, workdir=tmp_path), NoopTracker)

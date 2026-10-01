import json

import pytest

from recsys.config import TwoTowerSearchParams
from recsys.experiments import two_tower_search
from recsys.experiments.two_tower_search import run_two_tower_search, two_tower_grid
from recsys.tracking import InMemoryTracker

SEARCH = TwoTowerSearchParams(
    loss=("bce", "softmax"),
    embedding_dim=(4,),
    temperature=(0.1, 0.5),
    seeds=(0, 1),
    hidden_dim=8,
    epochs=3,
    batch_size=8,
    learning_rate=0.05,
    weight_decay=0.0,
    device="cpu",
)
N_CONFIGS, N_SEEDS, N_EPOCHS = 3, 2, 3


class SimulatedCrashError(RuntimeError):
    pass


@pytest.fixture
def search(labeled_splits, feature_tables, evaluation_params, tmp_path):
    """Run the search with fresh tracking; returns (report, tracker)."""

    def run(checkpoint_dir=tmp_path / "checkpoints"):
        tracker = InMemoryTracker()
        report = run_two_tower_search(
            labeled_splits,
            feature_tables,
            tmp_path / "two_tower_search.json",
            checkpoint_dir,
            SEARCH,
            evaluation_params,
            tracker,
        )
        return report, tracker

    return run


def test_grid_tries_every_temperature_for_the_softmax_loss_only():
    grid = two_tower_grid(SEARCH)

    assert [(p.loss, p.temperature) for p in grid] == [
        ("bce", None),
        ("softmax", 0.1),
        ("softmax", 0.5),
    ]
    assert all(p.seeds == (0, 1) and p.epochs == 3 and p.device == "cpu" for p in grid)


@pytest.mark.integration
def test_search_trains_every_seed_and_selects_on_the_seed_average(search, tmp_path):
    report, tracker = search()

    trials = report["trials"]
    assert len(tracker.runs) == N_CONFIGS * N_SEEDS  # one training curve per (config, seed)
    assert all(len(run["history"]) == N_EPOCHS for run in tracker.runs)
    assert len(trials) == N_CONFIGS * N_EPOCHS  # one trial per (config, epoch), seeds averaged
    assert [trial["params"]["epochs"] for trial in trials[:N_EPOCHS]] == [1, 2, 3]
    first = trials[0]
    per_seed = first["seed_spread"]["ndcg_at_2"]
    assert len(per_seed) == N_SEEDS
    assert first["metrics"]["ndcg_at_2"] == pytest.approx(sum(per_seed) / N_SEEDS)

    best = max(trials, key=lambda trial: trial["metrics"]["ndcg_at_2"])
    assert report["best"] == best
    corrected = report["best_score_optimism_corrected"]
    assert corrected == pytest.approx(best["metrics"]["ndcg_at_2"] - report["selection_optimism"])
    assert json.loads((tmp_path / "two_tower_search.json").read_text()) == report
    assert not (tmp_path / "checkpoints").exists()  # resume state is removed after success


@pytest.mark.integration
def test_an_interrupted_search_resumes_without_rescoring_finished_epochs(
    search, monkeypatch, tmp_path
):
    uninterrupted, _ = search(checkpoint_dir=tmp_path / "other_checkpoints")
    real_evaluate = two_tower_search.evaluate_model
    calls = {"count": 0, "crashed": False}

    def evaluate_then_crash_once_on_call_8(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 8 and not calls["crashed"]:  # config 2, seed 0, epoch 2
            calls["crashed"] = True
            raise SimulatedCrashError("killed mid-search")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(two_tower_search, "evaluate_model", evaluate_then_crash_once_on_call_8)
    with pytest.raises(SimulatedCrashError):
        search()
    calls["count"] = 0
    resumed, tracker = search()

    # Left to do: config 2 seed 0 epochs 2-3, config 2 seed 1, config 3 (both seeds).
    assert calls["count"] == 2 + N_EPOCHS + N_SEEDS * N_EPOCHS
    assert len(tracker.runs) == 4
    assert resumed["best"]["params"] == uninterrupted["best"]["params"]
    assert [t["metrics"]["ndcg_at_2"] for t in resumed["trials"]] == pytest.approx(
        [t["metrics"]["ndcg_at_2"] for t in uninterrupted["trials"]]
    )


@pytest.mark.integration
def test_progress_from_other_settings_is_discarded(search, tmp_path):
    stale = tmp_path / "checkpoints" / "progress.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"fingerprint": "made-with-other-settings", "scored": {}}))

    report, tracker = search()

    assert len(tracker.runs) == N_CONFIGS * N_SEEDS
    assert len(report["trials"]) == N_CONFIGS * N_EPOCHS

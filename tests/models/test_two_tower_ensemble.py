import logging

import polars as pl
import pytest

from recsys.models.two_tower.ensemble import TwoTowerEnsemble
from recsys.models.two_tower.recommender import TwoTowerRecommender

SETTINGS = {
    "loss": "bce",
    "embedding_dim": 4,
    "hidden_dim": 8,
    "epochs": 4,
    "batch_size": 8,
    "learning_rate": 0.05,
    "weight_decay": 0.0,
    "device": "cpu",
}
PAIRS = pl.DataFrame({"user_id": [0, 0, 4, 999], "video_id": [10, 13, 13, 10]})


class SimulatedCrashError(RuntimeError):
    pass


def views() -> pl.DataFrame:
    """Users 0-2 like videos 10-12, users 3-5 like videos 13-15."""
    rows = [(u, v, (u < 3) == (v < 13)) for u in range(6) for v in range(10, 16)]
    return pl.DataFrame(rows, schema=["user_id", "video_id", "is_positive"], orient="row")


def member(seed: int, **overrides) -> TwoTowerRecommender:
    users = pl.DataFrame({"user_id": list(range(6)), "group": ["a"] * 3 + ["b"] * 3})
    videos = pl.DataFrame({"video_id": list(range(10, 16)), "category_ids": [[1]] * 3 + [[2]] * 3})
    return TwoTowerRecommender(users, videos, **(SETTINGS | overrides), seed=seed)


def test_ensemble_scores_are_the_average_of_its_members():
    members = [member(1).fit(views()), member(2).fit(views())]

    ensemble = TwoTowerEnsemble(members)

    expected = (members[0].score(PAIRS) + members[1].score(PAIRS)) / 2
    assert ensemble.score(PAIRS).to_list() == pytest.approx(expected.to_list())


def test_ensemble_saves_every_member_and_loads_back_identically(tmp_path):
    ensemble = TwoTowerEnsemble([member(2), member(1)]).fit(views())

    ensemble.save(tmp_path / "two_tower")
    loaded = TwoTowerEnsemble.load(tmp_path / "two_tower", device="cpu")

    assert sorted(path.name for path in (tmp_path / "two_tower").iterdir()) == [
        "seed_1.pt",
        "seed_2.pt",
    ]
    assert loaded.score(PAIRS).to_list() == pytest.approx(ensemble.score(PAIRS).to_list())


def test_an_interrupted_ensemble_resumes_finished_members_instead_of_retraining(tmp_path, caplog):
    checkpoints = tmp_path / "checkpoints"
    crashing = member(2)

    def crash(*args, **kwargs):
        raise SimulatedCrashError("killed while training the second member")

    crashing.fit = crash
    with pytest.raises(SimulatedCrashError):
        TwoTowerEnsemble([member(1), crashing], checkpoint_dir=checkpoints).fit(views())
    with caplog.at_level(logging.INFO, logger="recsys.models.two_tower.recommender"):
        resumed = TwoTowerEnsemble([member(1), member(2)], checkpoint_dir=checkpoints).fit(views())
    uninterrupted = TwoTowerEnsemble([member(1), member(2)]).fit(views())

    assert "after epoch 4" in caplog.text  # member 1 was restored, not retrained
    assert resumed.score(PAIRS).to_list() == pytest.approx(uninterrupted.score(PAIRS).to_list())
    assert not checkpoints.exists()  # resume state is removed once every member is done


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([], "at least one"),
        ([member(1), member(1)], "distinct seeds"),
        ([member(1), member(2, embedding_dim=8)], "only in their seed"),
    ],
    ids=["empty", "same-seed", "different-settings"],
)
def test_ensemble_members_must_be_the_same_model_with_different_seeds(members, message):
    with pytest.raises(ValueError, match=message):
        TwoTowerEnsemble(members)


def test_loading_an_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="two-tower"):
        TwoTowerEnsemble.load(tmp_path)


def test_a_finished_fit_leaves_no_empty_checkpoint_folder_behind(tmp_path):
    checkpoints = tmp_path / "checkpoints" / "two_tower"

    TwoTowerEnsemble([member(1)], checkpoint_dir=checkpoints).fit(views())

    assert not checkpoints.exists()

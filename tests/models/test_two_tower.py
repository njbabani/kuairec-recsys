import math

import polars as pl
import pytest
import torch

from recsys.models.two_tower.recommender import TwoTowerRecommender, resolve_device

GROUP_A, GROUP_B = range(5), range(5, 10)
COLD_A, COLD_B = 20, 21  # in the catalogue, never in training
SETTINGS = {
    "embedding_dim": 8,
    "hidden_dim": 16,
    "epochs": 40,
    "batch_size": 16,
    "learning_rate": 0.02,
    "weight_decay": 0.0,
    "seed": 0,
    "device": "cpu",
}


def user_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": list(range(10)),
            "group": ["a" if user in GROUP_A else "b" for user in range(10)],
            "log_activity": [float(user % 3) for user in range(10)],
        }
    )


def video_features() -> pl.DataFrame:
    videos = [*range(10), COLD_A, COLD_B]
    return pl.DataFrame(
        {
            "video_id": videos,
            "category_ids": [[1] if v in GROUP_A or v == COLD_A else [2] for v in videos],
            "log_duration_s": [1.0 + (v % 4) for v in videos],
        }
    )


def taste_groups() -> pl.DataFrame:
    """Everyone saw every catalogue video and liked their own group's; (0, 0) is held out."""
    rows = [
        (user, video, (user in GROUP_A) == (video in GROUP_A))
        for user in range(10)
        for video in range(10)
        if (user, video) != (0, 0)
    ]
    return pl.DataFrame(rows, schema=["user_id", "video_id", "is_positive"], orient="row")


def model(loss: str = "bce", **overrides) -> TwoTowerRecommender:
    temperature = {"temperature": 0.2} if loss == "softmax" else {}
    return TwoTowerRecommender(
        user_features(), video_features(), loss=loss, **(SETTINGS | temperature | overrides)
    )


def scores(recommender, user: int, videos: list[int]) -> list[float]:
    pairs = pl.DataFrame({"user_id": [user] * len(videos), "video_id": videos})
    return recommender.score(pairs).to_list()


@pytest.mark.parametrize("loss", ["bce", "softmax"])
def test_two_tower_ranks_a_held_out_in_taste_video_above_the_other_groups_videos(loss):
    fitted = model(loss).fit(taste_groups())

    held_out, *other_group = scores(fitted, user=0, videos=[0, *GROUP_B])

    assert held_out > max(other_group)


@pytest.mark.parametrize("loss", ["bce", "softmax"])
def test_two_tower_scores_cold_start_videos_from_their_categories(loss):
    fitted = model(loss).fit(taste_groups())

    cold_a, cold_b = scores(fitted, user=0, videos=[COLD_A, COLD_B])

    assert cold_a > cold_b


def test_two_tower_scores_unknown_users_and_videos_without_failing():
    fitted = model().fit(taste_groups())

    result = scores(fitted, user=999, videos=[0, 5, 12345])

    assert all(math.isfinite(score) for score in result)


def test_scores_keep_the_row_order_of_the_pairs():
    fitted = model(epochs=5).fit(taste_groups())
    pairs = pl.DataFrame({"user_id": [7, 0, 7, 3], "video_id": [1, 6, 6, 2]})

    together = fitted.score(pairs).to_list()

    assert together == pytest.approx([scores(fitted, u, [v])[0] for u, v in pairs.iter_rows()])


def test_training_reports_a_falling_loss_after_every_epoch():
    reported: list[tuple[int, float]] = []

    fitted = model(epochs=6).fit(
        taste_groups(), on_epoch_end=lambda epoch, loss: reported.append((epoch, loss))
    )

    assert [epoch for epoch, _ in reported] == [1, 2, 3, 4, 5, 6]
    assert fitted.train_losses == [loss for _, loss in reported]
    assert reported[-1][1] < reported[0][1]


def test_scoring_inside_the_epoch_callback_does_not_disturb_training():
    def score_midway(epoch: int, loss: float) -> None:
        scores(watched, user=0, videos=[0, 5])

    watched = model(epochs=6)
    watched.fit(taste_groups(), on_epoch_end=score_midway)
    untouched = model(epochs=6).fit(taste_groups())

    assert watched.train_losses == pytest.approx(untouched.train_losses)


def test_two_tower_is_reproducible_for_a_seed():
    first = scores(model(epochs=5).fit(taste_groups()), user=3, videos=[0, 5, 9])
    again = scores(model(epochs=5).fit(taste_groups()), user=3, videos=[0, 5, 9])

    assert first == pytest.approx(again)


def test_two_tower_refuses_training_data_without_positives():
    train = taste_groups().with_columns(is_positive=pl.lit(False))

    with pytest.raises(ValueError, match="positive"):
        model().fit(train)


def test_scoring_before_fitting_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        model().score(pl.DataFrame({"user_id": [0], "video_id": [0]}))


def test_resolve_device_picks_the_apple_gpu_only_when_it_exists(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="MPS"):
        resolve_device("mps")


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="loss"):
        model(loss="hinge")


class SimulatedCrashError(RuntimeError):
    """Stands in for a crash or Ctrl-C partway through training."""


def interrupt_at(stop_epoch: int):
    def callback(epoch: int, loss: float) -> None:
        if epoch == stop_epoch:
            raise SimulatedCrashError(f"stopped at epoch {epoch}")

    return callback


def test_saved_model_loads_back_and_scores_identically(tmp_path):
    fitted = model(epochs=5).fit(taste_groups())
    pairs = pl.DataFrame({"user_id": [0, 3, 999], "video_id": [0, COLD_B, 5]})

    fitted.save(tmp_path / "two_tower.pt")
    loaded = TwoTowerRecommender.load(tmp_path / "two_tower.pt", device="cpu")

    assert loaded.score(pairs).to_list() == pytest.approx(fitted.score(pairs).to_list())
    assert loaded.train_losses == fitted.train_losses
    assert loaded.embedding_dim == fitted.embedding_dim


def test_a_loaded_model_can_be_trained_again_from_scratch(tmp_path):
    model(epochs=2).fit(taste_groups()).save(tmp_path / "two_tower.pt")

    retrained = TwoTowerRecommender.load(tmp_path / "two_tower.pt", device="cpu").fit(
        taste_groups()
    )

    assert len(retrained.train_losses) == 2


def test_loading_rejects_a_file_that_is_not_a_saved_model(tmp_path):
    path = tmp_path / "not_a_model.pt"
    torch.save({"format_version": 999}, path)

    with pytest.raises(ValueError, match="two-tower model"):
        TwoTowerRecommender.load(path, device="cpu")


def test_interrupted_training_resumes_from_its_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "last.pt"
    uninterrupted = model(epochs=6).fit(taste_groups())

    with pytest.raises(SimulatedCrashError):
        model(epochs=6).fit(
            taste_groups(), on_epoch_end=interrupt_at(4), checkpoint_path=checkpoint
        )
    reported: list[int] = []
    resumed = model(epochs=6).fit(
        taste_groups(),
        on_epoch_end=lambda epoch, loss: reported.append(epoch),
        checkpoint_path=checkpoint,
    )

    # Epoch 4 crashed before its checkpoint was written, so training restarts at epoch 4.
    assert reported == [4, 5, 6]
    assert resumed.train_losses == pytest.approx(uninterrupted.train_losses)
    assert scores(resumed, 0, [0, 5, COLD_A]) == pytest.approx(
        scores(uninterrupted, 0, [0, 5, COLD_A])
    )


def test_a_checkpoint_from_different_settings_is_not_resumed(tmp_path):
    checkpoint = tmp_path / "last.pt"
    with pytest.raises(SimulatedCrashError):
        model(epochs=6).fit(
            taste_groups(), on_epoch_end=interrupt_at(3), checkpoint_path=checkpoint
        )
    reported: list[int] = []

    model(epochs=6, learning_rate=0.01).fit(
        taste_groups(),
        on_epoch_end=lambda epoch, loss: reported.append(epoch),
        checkpoint_path=checkpoint,
    )

    assert reported == [1, 2, 3, 4, 5, 6]


def test_a_checkpoint_from_different_training_data_is_not_resumed(tmp_path):
    checkpoint = tmp_path / "last.pt"
    with pytest.raises(SimulatedCrashError):
        model(epochs=6).fit(
            taste_groups(), on_epoch_end=interrupt_at(3), checkpoint_path=checkpoint
        )
    reported: list[int] = []

    model(epochs=6).fit(
        taste_groups().filter(pl.col("user_id") != 9),
        on_epoch_end=lambda epoch, loss: reported.append(epoch),
        checkpoint_path=checkpoint,
    )

    assert reported[0] == 1


def test_checkpoints_are_written_atomically(tmp_path):
    model(epochs=2).fit(taste_groups(), checkpoint_path=tmp_path / "ckpt" / "last.pt")

    assert [path.name for path in (tmp_path / "ckpt").iterdir()] == ["last.pt"]


def test_softmax_describes_videos_never_liked_by_their_features_alone():
    # Video 20 was only ever skipped, so the softmax loss (positives only) never trains its id
    # embedding: it must score exactly like video 23, an unseen twin with the same features.
    twin = pl.DataFrame({"video_id": [23], "category_ids": [[1]], "log_duration_s": [1.0]})
    skipped = pl.DataFrame({"user_id": [0], "video_id": [COLD_A], "is_positive": [False]})
    recommender = TwoTowerRecommender(
        user_features(),
        pl.concat([video_features(), twin]),
        loss="softmax",
        temperature=0.2,
        **(SETTINGS | {"epochs": 3}),
    )

    fitted = recommender.fit(pl.concat([taste_groups(), skipped]))
    skipped_score, twin_score = scores(fitted, user=0, videos=[COLD_A, 23])

    assert skipped_score == pytest.approx(twin_score)


@pytest.mark.parametrize(
    ("loss", "overrides", "message"),
    [
        ("softmax", {}, "temperature"),
        ("bce", {"batch_size": 1}, "batch_size"),
    ],
    ids=["softmax-without-temperature", "batch-of-one"],
)
def test_invalid_settings_are_rejected(loss, overrides, message):
    settings = SETTINGS | overrides

    with pytest.raises(ValueError, match=message):
        TwoTowerRecommender(user_features(), video_features(), loss=loss, **settings)

import math

import numpy as np
import polars as pl
import pytest
import torch

from recsys.models.two_tower.encoding import encode_entities
from recsys.models.two_tower.network import Tower, in_batch_softmax_loss, pointwise_loss


@pytest.fixture
def video_features():
    table = pl.DataFrame(
        {"video_id": [10, 11, 12], "category_ids": [[1], [2], [1, 2]], "length": [1.0, 2.0, 3.0]}
    )
    return encode_entities(
        table,
        "video_id",
        extra_ids=pl.Series([13]),
        numeric=["length"],
        multi_valued="category_ids",
    )


def test_tower_maps_each_entity_to_one_vector(video_features):
    torch.manual_seed(0)
    tower = Tower(video_features, trained=np.ones(5, dtype=bool), embedding_dim=4, hidden_dim=8)

    vectors = tower(torch.tensor([0, 1, 2, 3, 4]))

    assert vectors.shape == (5, 4)
    assert torch.isfinite(vectors).all()


def test_entities_without_training_data_start_with_a_zero_id_embedding(video_features):
    trained = np.array([False, True, True, False, True])  # unknown row and video 12 unseen

    tower = Tower(video_features, trained=trained, embedding_dim=4, hidden_dim=8)

    weights = tower.id_embedding.weight.detach()
    assert torch.all(weights[[0, 3]] == 0)
    assert torch.all(weights[[1, 2, 4]].abs().sum(dim=1) > 0)


def test_an_untrained_entity_is_still_described_by_its_features(video_features):
    torch.manual_seed(0)
    trained = np.array([False, True, True, False, True])
    tower = Tower(video_features, trained=trained, embedding_dim=4, hidden_dim=8)

    cold_video, unknown = tower(torch.tensor([3, 0]))

    assert not torch.allclose(cold_video, unknown)


def test_pointwise_loss_is_binary_cross_entropy_on_the_dot_product():
    users = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    videos = torch.tensor([[3.0, 0.0], [0.0, -1.0]])
    labels = torch.tensor([1.0, 0.0])

    loss = pointwise_loss(users, videos, labels)

    expected = (math.log1p(math.exp(-3.0)) + math.log1p(math.exp(-2.0))) / 2
    assert loss.item() == pytest.approx(expected)


def test_in_batch_softmax_loss_corrects_for_how_often_each_video_is_sampled():
    # Equal similarities everywhere, so the loss is driven by the logQ correction alone.
    users = torch.ones(2, 3)
    videos = torch.ones(2, 3)
    sampling_probability = torch.tensor([0.9, 0.1])

    loss = in_batch_softmax_loss(
        users,
        videos,
        video_index=torch.tensor([0, 1]),
        log_q=sampling_probability.log(),
        temperature=1.0,
    )

    # Row i's loss is log(q_i * sum_j 1 / q_j): popular positives are discounted the most.
    q = sampling_probability.tolist()
    expected = sum(math.log(q_i * sum(1 / q_j for q_j in q)) for q_i in q) / 2
    assert loss.item() == pytest.approx(expected)


def test_in_batch_softmax_loss_never_treats_a_video_as_its_own_negative():
    # Both rows liked the same video; without masking each row would see the other's copy of
    # its positive as a negative and the loss would be log(2) instead of 0.
    users = torch.ones(2, 3)
    videos = torch.ones(2, 3)

    loss = in_batch_softmax_loss(
        users,
        videos,
        video_index=torch.tensor([7, 7]),
        log_q=torch.zeros(2),
        temperature=1.0,
    )

    assert loss.item() == pytest.approx(0.0)


def test_feature_values_that_no_trained_entity_has_start_at_zero():
    table = pl.DataFrame({"video_id": [10, 11], "kind": ["seen", "only_on_a_cold_video"]})
    features = encode_entities(
        table, "video_id", extra_ids=pl.Series([], dtype=pl.Int64), categorical=["kind"]
    )

    # rows: unknown entity, video 10 (trained), video 11 (never trained)
    tower = Tower(features, trained=np.array([False, True, False]), embedding_dim=4, hidden_dim=8)

    # codes: 0 unknown, 1 "only_on_a_cold_video", 2 "seen"
    weights = tower.categorical_embedding.weight.detach()
    assert torch.all(weights[[0, 1]] == 0)
    assert weights[2].abs().sum() > 0


def test_tower_rejects_a_trained_mask_of_the_wrong_length(video_features):
    with pytest.raises(ValueError, match="one flag per entity"):
        Tower(video_features, trained=np.ones(2, dtype=bool), embedding_dim=4, hidden_dim=8)

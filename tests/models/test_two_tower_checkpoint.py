import pickle

import numpy as np
import polars as pl
import pytest
import torch

from recsys.checkpointing import frame_fingerprint
from recsys.models.two_tower.checkpoint import (
    entity_from_payload,
    entity_to_payload,
    frame_from_tensor,
    frame_to_tensor,
    load_payload,
    save_atomic,
)
from recsys.models.two_tower.encoding import encode_entities


class NotData:
    """Any class instance: a pickle that would run code on load."""


def test_save_atomic_writes_the_file_and_leaves_no_temporary_behind(tmp_path):
    path = tmp_path / "nested" / "state.pt"

    save_atomic({"epoch": 3, "weights": torch.ones(2)}, path)

    assert [p.name for p in path.parent.iterdir()] == ["state.pt"]
    assert load_payload(path)["epoch"] == 3


def test_load_payload_refuses_files_that_would_execute_code(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"payload": NotData()}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_payload(path)


def test_frames_survive_a_round_trip_with_their_exact_types():
    frame = pl.DataFrame(
        {
            "video_id": [1, 2],
            "category_ids": [[8], [27, 9]],
            "label": ["a", None],
            "length": [1.5, None],
        }
    )

    restored = frame_from_tensor(frame_to_tensor(frame))

    assert restored.equals(frame)
    assert restored.schema == frame.schema


def test_frame_fingerprint_changes_with_content_and_not_otherwise():
    frame = pl.DataFrame({"user_id": [1, 2], "category_ids": [[1], [2, 3]]})

    assert frame_fingerprint(frame) == frame_fingerprint(frame.clone())
    assert frame_fingerprint(frame) != frame_fingerprint(frame.with_columns(user_id=pl.lit(1)))


def test_entity_features_survive_a_round_trip():
    table = pl.DataFrame(
        {"video_id": [10, 11], "category_ids": [[1], [2, 3]], "kind": ["a", "b"], "x": [1.0, 2.0]}
    )
    features = encode_entities(
        table,
        "video_id",
        extra_ids=pl.Series([12]),
        categorical=["kind"],
        numeric=["x"],
        multi_valued="category_ids",
    )

    restored = entity_from_payload(entity_to_payload(features))

    assert restored.ids == features.ids
    assert restored.cardinalities == features.cardinalities
    assert restored.multi_valued_cardinality == features.multi_valued_cardinality
    for name in ("categorical", "numeric", "multi_valued"):
        assert np.array_equal(getattr(restored, name), getattr(features, name)), name

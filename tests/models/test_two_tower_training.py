import polars as pl
import pytest

from recsys.models.two_tower.training import FeatureRoles, feature_roles


def test_feature_roles_follow_the_column_types():
    table = pl.DataFrame(
        {"video_id": [1], "kind": ["a"], "length": [2.0], "count": [3], "tags": [[1, 2]]}
    )

    roles = feature_roles(table, "video_id")

    assert roles == FeatureRoles(
        categorical=("kind",), numeric=("length", "count"), multi_valued="tags"
    )


def test_feature_roles_reject_unsupported_types_and_extra_list_columns():
    with pytest.raises(TypeError, match="when"):
        feature_roles(pl.DataFrame({"id": [1], "when": [pl.date(2020, 1, 1)]}), "id")
    with pytest.raises(ValueError, match="one list column"):
        feature_roles(pl.DataFrame({"id": [1], "a": [[1]], "b": [[2]]}), "id")

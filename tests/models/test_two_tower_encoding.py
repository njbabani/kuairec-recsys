import numpy as np
import polars as pl
import pytest

from recsys.models.two_tower.encoding import UNKNOWN, Vocabulary, encode_entities

NO_EXTRA_IDS = pl.Series([], dtype=pl.Int64)


def test_vocabulary_reserves_index_zero_for_unknown_and_missing_values():
    vocabulary = Vocabulary.fit(pl.Series(["b", "a", None, "b"]))

    encoded = vocabulary.encode(pl.Series(["a", "b", "never-seen", None]))

    assert len(vocabulary) == 3  # unknown + "a" + "b"
    assert encoded.tolist() == [1, 2, UNKNOWN, UNKNOWN]


def test_entities_are_indexed_from_one_and_include_ids_missing_from_the_table():
    table = pl.DataFrame({"user_id": [7, 3], "level": ["high", "low"]})

    features = encode_entities(table, "user_id", extra_ids=pl.Series([3, 9]))

    assert features.ids.encode(pl.Series([3, 7, 9, 100])).tolist() == [1, 2, 3, UNKNOWN]
    assert features.size == 4


def test_categorical_columns_get_their_own_vocabulary_and_unknown_rows():
    table = pl.DataFrame({"user_id": [7, 3], "level": ["high", "low"], "flag": ["1", None]})

    features = encode_entities(
        table, "user_id", extra_ids=pl.Series([9]), categorical=["level", "flag"]
    )

    # rows: unknown entity, user 3, user 7, user 9 (not in the table)
    assert features.categorical.tolist() == [[0, 0], [2, 0], [1, 1], [0, 0]]
    assert features.cardinalities == (3, 2)


def test_numeric_columns_are_standardised_and_missing_values_sit_at_the_mean():
    table = pl.DataFrame({"user_id": [1, 2, 3], "x": [1.0, 3.0, None]})

    features = encode_entities(table, "user_id", extra_ids=NO_EXTRA_IDS, numeric=["x"])

    assert features.numeric[:, 0] == pytest.approx([0.0, -1.0, 1.0, 0.0])
    assert features.numeric.dtype == np.float32


def test_constant_numeric_columns_do_not_divide_by_zero():
    table = pl.DataFrame({"user_id": [1, 2], "x": [5.0, 5.0]})

    features = encode_entities(table, "user_id", extra_ids=NO_EXTRA_IDS, numeric=["x"])

    assert features.numeric[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_multi_valued_column_is_padded_with_zeros():
    table = pl.DataFrame({"video_id": [10, 11], "category_ids": [[8], [27, 9]]})

    features = encode_entities(
        table, "video_id", extra_ids=pl.Series([12]), multi_valued="category_ids"
    )

    # vocabulary: 8 -> 1, 9 -> 2, 27 -> 3; video 12 has no categories
    assert features.multi_valued.tolist() == [[0, 0], [1, 0], [3, 2], [0, 0]]
    assert features.multi_valued_cardinality == 4


def test_tables_without_feature_columns_still_encode_ids():
    features = encode_entities(
        pl.DataFrame({"video_id": [5]}), "video_id", extra_ids=pl.Series([6])
    )

    assert features.categorical.shape == (3, 0)
    assert features.numeric.shape == (3, 0)
    assert features.multi_valued.shape == (3, 0)


def test_duplicate_keys_are_rejected_instead_of_misaligning_rows():
    table = pl.DataFrame({"user_id": [1, 1], "x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="unique"):
        encode_entities(table, "user_id", extra_ids=NO_EXTRA_IDS, numeric=["x"])


def test_non_finite_numeric_values_are_treated_as_missing():
    table = pl.DataFrame({"user_id": [1, 2, 3], "x": [1.0, 3.0, float("nan")]})

    features = encode_entities(table, "user_id", extra_ids=NO_EXTRA_IDS, numeric=["x"])

    assert features.numeric[:, 0] == pytest.approx([0.0, -1.0, 1.0, 0.0])

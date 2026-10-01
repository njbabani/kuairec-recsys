from pathlib import Path

import polars as pl
import pytest

from recsys.data.categories import CATEGORY_NAMES_EN, with_english_category_names

CAPTIONS = Path(__file__).resolve().parents[2] / "data/processed/video_captions.parquet"


def test_english_names_cover_kuairecs_38_categories_plus_unknown():
    assert len(CATEGORY_NAMES_EN) == 39
    assert CATEGORY_NAMES_EN[28] == "Local news & society"
    assert CATEGORY_NAMES_EN[-124] == "Unknown"


@pytest.mark.data
@pytest.mark.skipif(not CAPTIONS.is_file(), reason="needs the processed dataset (`make data`)")
def test_every_category_in_the_dataset_has_an_english_name():
    category_ids = set(pl.read_parquet(CAPTIONS)["first_level_category_id"].unique())

    assert category_ids <= set(CATEGORY_NAMES_EN)


def test_with_english_category_names_falls_back_to_the_source_name_for_unknown_ids():
    captions = pl.DataFrame(
        {"first_level_category_id": [8, 999], "first_level_category_name": ["颜值", "新类目"]}
    )

    result = with_english_category_names(captions)

    assert result["category_en"].to_list() == ["Good looks", "新类目"]
    assert captions.columns == ["first_level_category_id", "first_level_category_name"]

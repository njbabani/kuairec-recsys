"""English display names for KuaiRec's first-level video categories (source names are Chinese)."""

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

CATEGORY_NAMES_EN: Mapping[int, str] = MappingProxyType(
    {
        1: "Dance",
        2: "Music",
        3: "Gaming",
        4: "Beauty & makeup",
        5: "Fashion",
        6: "Celebrities & entertainment",
        7: "Sports",
        8: "Good looks",
        9: "Comedy",
        10: "Travel",
        11: "Lifestyle",
        12: "Food",
        13: "Rural life",
        14: "Education",
        15: "Arts",
        16: "Health",
        17: "Pets",
        18: "Cars",
        19: "Relationships",
        20: "Anime & ACG",
        21: "Humanities",
        22: "Finance",
        23: "Politics & current affairs",
        24: "Astrology",
        25: "Parenting",
        26: "Photography",
        27: "Tech & gadgets",
        28: "Local news & society",
        29: "Science & law",
        31: "Fitness",
        32: "Short dramas",
        33: "Selfies",
        34: "Other",
        35: "Military",
        36: "Home & real estate",
        37: "Oddities",
        38: "Books",
        39: "Film, TV & variety",
        -124: "Unknown",
    }
)


def with_english_category_names(captions: pl.DataFrame) -> pl.DataFrame:
    """Return ``captions`` plus ``category_en``; unmapped ids keep their source name."""
    return captions.with_columns(
        category_en=pl.col("first_level_category_id").replace_strict(
            dict(CATEGORY_NAMES_EN),
            default=pl.col("first_level_category_name"),
            return_dtype=pl.String,
        )
    )

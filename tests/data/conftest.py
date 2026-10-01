"""Synthetic KuaiRec-shaped archive: same member layout and column formats as the real zip."""

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

ARCHIVE_DATA_DIR = "KuaiRec 2.0/data"

# Row 2 is an exact duplicate of row 1 (logging duplicate, dropped); row 3 is a re-watch (kept).
BIG_MATRIX_CSV = """\
user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio
0,10,5000,10000,2020-07-05 00:08:23.438,20200705,1593878903.438,0.5
0,10,5000,10000,2020-07-05 00:08:23.438,20200705,1593878903.438,0.5
0,10,12000,10000,2020-07-06 00:08:23.438,20200706,1593965303.438,1.2
1,11,3000,6000,2020-07-05 01:00:00.000,20200705,1593882000.0,0.5
"""

# Disjoint (user, video) pairs from BIG_MATRIX_CSV; second row has no time information.
SMALL_MATRIX_CSV = """\
user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio
0,11,6000,6000,2020-07-05 02:00:00.000,20200705.0,1593885600.0,1.0
1,10,0,10000,,,,0.0
"""

USER_FEATURES_CSV = """\
user_id,user_active_degree,register_days
0,high_active,107
1,full_active,30
"""

ITEM_CATEGORIES_CSV = """\
video_id,feat
10,[8]
11,"[27, 9]"
"""

CAPTION_CATEGORY_CSV = """\
video_id,caption,first_level_category_name
10,hello,beauty
11,,UNKNOWN
"""

ITEM_DAILY_FEATURES_CSV = """\
video_id,date,video_duration,collect_cnt,collect_user_num,cancel_collect_cnt,cancel_collect_user_num
10,20200705,10000.0,1.0,1.0,0.0,0.0
11,20200705,,,,,
"""

DEFAULT_MEMBERS = {
    "big_matrix": BIG_MATRIX_CSV,
    "small_matrix": SMALL_MATRIX_CSV,
    "user_features": USER_FEATURES_CSV,
    "item_categories": ITEM_CATEGORIES_CSV,
    "kuairec_caption_category": CAPTION_CATEGORY_CSV,
    "item_daily_features": ITEM_DAILY_FEATURES_CSV,
}


@pytest.fixture
def make_archive(tmp_path: Path) -> Callable[..., Path]:
    """Build a KuaiRec-shaped zip; pass ``overrides`` to replace (or ``None`` to omit) members."""

    def _make(overrides: dict[str, str | None] | None = None) -> Path:
        members = {**DEFAULT_MEMBERS, **(overrides or {})}
        archive_path = tmp_path / "KuaiRec.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, content in members.items():
                if content is not None:
                    archive.writestr(f"{ARCHIVE_DATA_DIR}/{name}.csv", content)
        return archive_path

    return _make

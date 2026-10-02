import polars as pl
import pytest

from recsys.demo.bundle import session_videos, user_profiles, video_catalog

CAPTIONS = pl.DataFrame(
    {
        "video_id": [10, 11, 12],
        "first_level_category_id": [1, 9, 999],
        "first_level_category_name": ["舞蹈", "搞笑", "未知类目"],
    }
)
SESSIONS = pl.DataFrame(
    {"policy": ["als"] * 2, "user_id": [1, 1], "video_id": [11, 13], "position": [1, 2]}
)
REACTIONS = pl.DataFrame(
    {
        "user_id": [1, 1, 1, 1],
        "video_id": [10, 11, 12, 13],
        "is_positive": [True, False, True, True],
        "play_duration": [4000, 2500, 9000, 7000],
        "video_duration": [10000, 5000, 30000, 8000],
    }
)


def test_the_catalog_gives_each_video_its_english_category():
    catalog = video_catalog(CAPTIONS)

    assert catalog.sort("video_id").rows() == [
        (10, "Dance"),
        (11, "Comedy"),
        (12, "未知类目"),  # no English name known: keep the source name
    ]


def test_session_videos_carry_each_videos_category_length_and_reaction():
    videos = session_videos(SESSIONS, REACTIONS, video_catalog(CAPTIONS))

    assert videos.columns == [
        "policy",
        "user_id",
        "position",
        "video_id",
        "category",
        "duration_s",
        "watched_s",
        "liked",
    ]
    assert videos.rows() == [
        ("als", 1, 1, 11, "Comedy", 5.0, 2.5, False),
        ("als", 1, 2, 13, "Unknown", 8.0, 7.0, True),  # not in the catalog
    ]


def test_a_session_video_without_a_recorded_reaction_is_an_error():
    reactions = REACTIONS.filter(pl.col("video_id") != 13)

    with pytest.raises(ValueError, match="reaction"):
        session_videos(SESSIONS, reactions, video_catalog(CAPTIONS))


@pytest.mark.parametrize("column", ["is_positive", "play_duration", "video_duration"])
def test_a_session_video_with_an_incomplete_reaction_is_an_error(column):
    reactions = REACTIONS.with_columns(
        pl.when(pl.col("video_id") == 13).then(None).otherwise(pl.col(column)).alias(column)
    )

    with pytest.raises(ValueError, match="reaction"):
        session_videos(SESSIONS, reactions, video_catalog(CAPTIONS))


@pytest.mark.parametrize("duplicated", ["reactions", "captions"])
def test_duplicate_reactions_or_captions_are_an_error_not_extra_rows(duplicated):
    reactions = pl.concat([REACTIONS, REACTIONS]) if duplicated == "reactions" else REACTIONS
    captions = pl.concat([CAPTIONS, CAPTIONS]) if duplicated == "captions" else CAPTIONS

    with pytest.raises(ValueError, match="duplicate"):
        session_videos(SESSIONS, reactions, video_catalog(captions))


def test_user_profiles_summarise_each_test_users_history_before_the_experiment():
    train = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "video_id": [10, 10, 11, 11, 12],
            "is_positive": [True, True, True, False, True],
        }
    )
    arms = pl.DataFrame({"user_id": [1, 2], "in_treatment": [True, False]})
    test = pl.concat(
        [
            REACTIONS.select("user_id", "is_positive"),
            pl.DataFrame({"user_id": [2], "is_positive": [False]}),
        ]
    )

    profiles = user_profiles(train, test, arms, video_catalog(CAPTIONS)).sort("user_id")

    first, second = profiles.to_dicts()
    assert first["in_treatment"] is True
    assert first["training_views"] == 5
    assert first["training_positive_rate"] == pytest.approx(0.8)
    # Positives: Dance twice, Comedy and the unnamed category once each (ties: alphabetical).
    assert first["favourite_categories"] == ["Dance", "Comedy", "未知类目"]
    assert first["liked_share"] == pytest.approx(0.75)
    assert second["training_views"] == 0  # no history: counts are zero, rates unknown
    assert second["training_positive_rate"] is None
    assert second["favourite_categories"] == []


@pytest.mark.integration
def test_the_demo_stage_writes_a_row_per_session_video_and_per_test_user(demo_root, demo_world):
    videos = pl.read_parquet(demo_root / "data/demo/sessions.parquet")
    users = pl.read_parquet(demo_root / "data/demo/users.parquet")

    sessions = len(demo_world.policy_videos) * demo_world.n_users
    assert videos.height == sessions * demo_world.session_length
    assert users.height == demo_world.n_users
    first = videos.filter(
        pl.col("policy") == "als", pl.col("user_id") == 0, pl.col("position") == 1
    ).row(0, named=True)
    shown = demo_world.policy_videos["als"][0]
    assert (first["video_id"], first["category"]) == (shown, demo_world.category_name(shown))

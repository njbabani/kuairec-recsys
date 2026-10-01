import json
import math
import zipfile
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandera.polars as pa
import polars as pl
import pytest

from recsys.data.prepare import (
    COLLECT_COLUMNS,
    INTERACTION_SOURCE_DTYPES,
    DataIntegrityError,
    DataLeakageError,
    check_disjoint_pairs,
    check_referential_integrity,
    clean_interactions,
    parse_int_list,
    prepare,
    read_csv_from_zip,
    summarize_interactions,
    to_event_time,
)
from recsys.data.schemas import (
    BIG_MATRIX_SCHEMA,
    EVENT_TIME_DTYPE,
    SMALL_MATRIX_SCHEMA,
    USERS_SCHEMA,
    VIDEO_CAPTIONS_SCHEMA,
    VIDEO_CATEGORIES_SCHEMA,
    VIDEO_DAILY_STATS_SCHEMA,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def event_times(*values: datetime | None) -> pl.Series:
    return pl.Series(list(values), dtype=EVENT_TIME_DTYPE)


def interactions(**overrides) -> pl.DataFrame:
    base = {
        "user_id": [0],
        "video_id": [10],
        "play_duration": [5000],
        "video_duration": [10000],
        "watch_ratio": [0.5],
        "event_time": event_times(datetime(2020, 7, 5, tzinfo=SHANGHAI)),
    }
    return pl.DataFrame({**base, **overrides})


def daily_stats(**overrides) -> pl.DataFrame:
    base = {
        "video_id": [10],
        "date": [date(2020, 7, 5)],
        "video_duration": [10000.0],
        **{column: [1] for column in COLLECT_COLUMNS},
    }
    return pl.DataFrame({**base, **overrides})


def raw_interactions(**columns) -> pl.DataFrame:
    """Source-shaped rows (with the redundant time/date columns) for clean_interactions."""
    return pl.DataFrame(
        {
            "user_id": [0] * len(columns["timestamp"]),
            "video_id": [10] * len(columns["timestamp"]),
            "play_duration": [5000] * len(columns["timestamp"]),
            "video_duration": [10000] * len(columns["timestamp"]),
            "watch_ratio": [0.5] * len(columns["timestamp"]),
            **columns,
        }
    )


# --- unit: transformations ---------------------------------------------------------------


def test_to_event_time_converts_epoch_seconds_to_shanghai_local_time():
    df = pl.DataFrame({"timestamp": [1593878903.438, None]})

    result = df.select(event_time=to_event_time(pl.col("timestamp")))["event_time"]

    assert result.dtype == EVENT_TIME_DTYPE
    assert result[0] == datetime(2020, 7, 5, 0, 8, 23, 438000, tzinfo=SHANGHAI)
    # Wall-clock must match the source `time` column, not just the same instant.
    assert result.dt.strftime("%Y-%m-%d %H:%M:%S%.3f")[0] == "2020-07-05 00:08:23.438"
    assert result[1] is None


@pytest.mark.parametrize("table", ["big_matrix", "small_matrix"])
def test_event_time_reproduces_the_source_time_column(make_archive, table):
    with zipfile.ZipFile(make_archive()) as archive:
        raw = read_csv_from_zip(archive, table).filter(pl.col("timestamp").is_not_null())

    rendered = raw.select(
        to_event_time(pl.col("timestamp")).dt.strftime("%Y-%m-%d %H:%M:%S%.3f")
    ).to_series()

    assert rendered.to_list() == raw["time"].to_list()


def test_clean_interactions_drops_exact_duplicates_but_keeps_rewatches(make_archive):
    with zipfile.ZipFile(make_archive()) as archive:
        raw = read_csv_from_zip(archive, "big_matrix")

    cleaned, dropped = clean_interactions(raw)

    assert dropped == 1
    assert cleaned.height == 3
    assert cleaned.filter((pl.col("user_id") == 0) & (pl.col("video_id") == 10)).height == 2
    assert cleaned.columns == [
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "watch_ratio",
        "event_time",
    ]


def test_clean_interactions_ignores_partition_date_when_deduplicating():
    # Same event logged under two daily partitions, then a genuine re-watch a day later.
    raw = raw_interactions(
        time=["2020-07-05 00:08:23.438", "2020-07-05 00:08:23.438", "2020-07-06 00:08:23.438"],
        date=[20200705, 20200707, 20200706],
        timestamp=[1593878903.438, 1593878903.438, 1593965303.438],
    )

    cleaned, dropped = clean_interactions(raw)

    assert dropped == 1
    assert cleaned["event_time"].dt.day().to_list() == [5, 6]


def test_parse_int_list_handles_single_multi_and_empty_lists():
    df = pl.DataFrame({"feat": ["[8]", "[27, 9]", "[]"]})

    result = df.select(parse_int_list("feat"))["feat"].to_list()

    assert result == [[8], [27, 9], []]


def test_read_csv_from_zip_applies_explicit_dtypes_instead_of_inferring(make_archive):
    integer_looking = (
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,10,5000,5000,2020-07-05 00:08:23.438,20200705,1593878903.438,1\n"
    )
    with zipfile.ZipFile(make_archive({"big_matrix": integer_looking})) as archive:
        df = read_csv_from_zip(archive, "big_matrix", INTERACTION_SOURCE_DTYPES)

    assert df.schema == pl.Schema(INTERACTION_SOURCE_DTYPES)


def test_read_csv_from_zip_raises_clear_error_for_missing_member(make_archive):
    with (
        zipfile.ZipFile(make_archive({"user_features": None})) as archive,
        pytest.raises(FileNotFoundError, match="user_features"),
    ):
        read_csv_from_zip(archive, "user_features")


def test_summarize_interactions_handles_an_empty_table():
    empty = interactions().clear()

    summary = summarize_interactions(empty, duplicates_dropped=0)

    assert summary["rows"] == 0
    assert summary["density"] == 0.0


# --- unit: integrity guards --------------------------------------------------------------


def test_check_disjoint_pairs_returns_zero_when_no_pair_is_shared():
    assert check_disjoint_pairs(interactions(video_id=[10]), interactions(video_id=[11])) == 0


def test_check_disjoint_pairs_raises_with_examples_when_a_pair_is_in_both_tables():
    with pytest.raises(DataLeakageError, match=r"1 \(user_id, video_id\) pair.*\(0, 10\)"):
        check_disjoint_pairs(interactions(), interactions())


def test_check_referential_integrity_rejects_unknown_video_ids():
    tables = {
        "big_matrix": interactions(),
        "small_matrix": interactions(video_id=[99]),
        "users": pl.DataFrame({"user_id": [0]}),
        "video_categories": pl.DataFrame({"video_id": [10]}),
    }

    with pytest.raises(DataIntegrityError, match=r"1 video_id value.*small_matrix"):
        check_referential_integrity(tables)


# --- unit: data contracts ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "bad_frame", "failing_column"),
    [
        pytest.param(
            BIG_MATRIX_SCHEMA, interactions(watch_ratio=[-0.1]), "watch_ratio", id="negative-ratio"
        ),
        pytest.param(
            BIG_MATRIX_SCHEMA, interactions(watch_ratio=[math.inf]), "watch_ratio", id="inf-ratio"
        ),
        pytest.param(
            BIG_MATRIX_SCHEMA, interactions(video_duration=[0]), "video_duration", id="zero-length"
        ),
        pytest.param(
            BIG_MATRIX_SCHEMA,
            interactions(event_time=event_times(None)),
            "event_time",
            id="train-missing-time",
        ),
        pytest.param(
            SMALL_MATRIX_SCHEMA,
            interactions(event_time=event_times(datetime(2021, 1, 1, tzinfo=SHANGHAI))),
            "event_time",
            id="time-outside-dataset-window",
        ),
        pytest.param(
            BIG_MATRIX_SCHEMA, interactions().with_columns(extra=pl.lit(1)), "extra", id="strict"
        ),
        pytest.param(
            VIDEO_DAILY_STATS_SCHEMA,
            daily_stats(collect_cnt=[1.7]),
            "collect_cnt",
            id="float-counter",
        ),
    ],
)
def test_schemas_reject_invalid_values(schema, bad_frame, failing_column):
    with pytest.raises(pa.errors.SchemaErrors, match=failing_column):
        schema.validate(bad_frame, lazy=True)


def test_small_matrix_schema_allows_missing_event_time():
    SMALL_MATRIX_SCHEMA.validate(interactions(event_time=event_times(None)), lazy=True)


@pytest.mark.parametrize(
    ("schema", "duplicated"),
    [
        pytest.param(SMALL_MATRIX_SCHEMA, pl.concat([interactions(), interactions()]), id="pairs"),
        pytest.param(USERS_SCHEMA, pl.DataFrame({"user_id": [0, 0]}), id="users"),
        pytest.param(
            VIDEO_CATEGORIES_SCHEMA,
            pl.DataFrame({"video_id": [1, 1], "category_ids": [[8], [9]]}),
            id="categories",
        ),
        pytest.param(VIDEO_CAPTIONS_SCHEMA, pl.DataFrame({"video_id": [1, 1]}), id="captions"),
        pytest.param(
            VIDEO_DAILY_STATS_SCHEMA, pl.concat([daily_stats(), daily_stats()]), id="daily-stats"
        ),
    ],
)
def test_schemas_reject_duplicate_keys(schema, duplicated):
    with pytest.raises(pa.errors.SchemaErrors, match="uniqueness"):
        schema.validate(duplicated, lazy=True)


# --- integration: full stage -------------------------------------------------------------


@pytest.mark.integration
def test_prepare_writes_validated_parquet_tables_and_summary(make_archive, tmp_path):
    out_dir, summary_path = tmp_path / "processed", tmp_path / "reports" / "summary.json"

    prepare(make_archive(), out_dir, summary_path)

    written = sorted(p.stem for p in out_dir.glob("*.parquet"))
    assert written == [
        "big_matrix",
        "small_matrix",
        "users",
        "video_captions",
        "video_categories",
        "video_daily_stats",
    ]
    categories = pl.read_parquet(out_dir / "video_categories.parquet")
    assert categories.schema["category_ids"] == pl.List(pl.Int64)
    daily = pl.read_parquet(out_dir / "video_daily_stats.parquet")
    assert daily.schema["date"] == pl.Date
    assert daily.schema["collect_cnt"] == pl.Int64

    summary = json.loads(summary_path.read_text())
    assert summary["big_matrix"]["rows"] == 3
    assert summary["big_matrix"]["duplicate_rows_dropped"] == 1
    assert summary["big_matrix"]["watch_ratio_p50"] == 0.5
    assert summary["big_matrix"]["watch_ratio_max"] == 1.2
    assert summary["small_matrix"]["null_event_time"] == 1
    assert summary["small_matrix"]["density"] == pytest.approx(2 / (2 * 2))
    assert summary["leakage_pairs"] == 0


@pytest.mark.integration
def test_prepare_refuses_to_write_when_eval_pairs_leak_into_training(make_archive, tmp_path):
    leaking_small = (
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,10,5000,10000,2020-07-05 00:08:23.438,20200705.0,1593878903.438,0.5\n"
    )
    out_dir = tmp_path / "processed"

    with pytest.raises(DataLeakageError):
        prepare(make_archive({"small_matrix": leaking_small}), out_dir, tmp_path / "s.json")

    assert not out_dir.exists()


@pytest.mark.integration
def test_prepare_refuses_to_write_when_interactions_reference_unknown_videos(
    make_archive, tmp_path
):
    unknown_video = (
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "0,99,5000,10000,2020-07-05 00:08:23.438,20200705.0,1593878903.438,0.5\n"
    )
    out_dir = tmp_path / "processed"

    with pytest.raises(DataIntegrityError, match="video_id"):
        prepare(make_archive({"small_matrix": unknown_video}), out_dir, tmp_path / "s.json")

    assert not out_dir.exists()

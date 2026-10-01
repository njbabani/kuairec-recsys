"""Convert the raw KuaiRec archive into validated, typed parquet tables plus a data summary.

Run as a DVC stage: ``python -m recsys.data.prepare``. Nothing is written unless every
table passes its data contract, the train/eval leakage check and the referential checks.
"""

import json
import logging
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandera.polars as pa
import polars as pl

from recsys.config import load_params
from recsys.data.schemas import (
    BIG_MATRIX_SCHEMA,
    COLLECT_COLUMNS,
    INTERACTION_KEYS,
    LOCAL_TIMEZONE,
    SMALL_MATRIX_SCHEMA,
    USERS_SCHEMA,
    VIDEO_CAPTIONS_SCHEMA,
    VIDEO_CATEGORIES_SCHEMA,
    VIDEO_DAILY_STATS_SCHEMA,
)
from recsys.io import write_json, write_parquet_tables

logger = logging.getLogger(__name__)

ARCHIVE_DATA_DIR = "KuaiRec 2.0/data"
INTERACTION_SOURCE_DTYPES: dict[str, pl.DataType] = {
    "user_id": pl.Int64(),
    "video_id": pl.Int64(),
    "play_duration": pl.Int64(),
    "video_duration": pl.Int64(),
    "timestamp": pl.Float64(),
    "watch_ratio": pl.Float64(),
}
INTERACTION_TABLES = ("big_matrix", "small_matrix")
# (foreign key, parent table) every interaction table must satisfy.
INTERACTION_REFERENCES = (("user_id", "users"), ("video_id", "video_categories"))
MS_PER_SECOND = 1000
SUMMARY_DECIMALS = 4
MAX_EXAMPLES_IN_ERROR = 5

TABLE_SCHEMAS: Mapping[str, pa.DataFrameSchema] = MappingProxyType(
    {
        "big_matrix": BIG_MATRIX_SCHEMA,
        "small_matrix": SMALL_MATRIX_SCHEMA,
        "users": USERS_SCHEMA,
        "video_categories": VIDEO_CATEGORIES_SCHEMA,
        "video_captions": VIDEO_CAPTIONS_SCHEMA,
        "video_daily_stats": VIDEO_DAILY_STATS_SCHEMA,
    }
)


class DataIntegrityError(RuntimeError):
    """Raised when tables are individually valid but inconsistent with each other."""


class DataLeakageError(DataIntegrityError):
    """Raised when evaluation (user, video) pairs also appear in the training interactions."""


def read_csv_from_zip(
    archive: zipfile.ZipFile, name: str, dtypes: dict[str, pl.DataType] | None = None
) -> pl.DataFrame:
    """Read ``<name>.csv`` from the archive's data folder without extracting to disk.

    With ``dtypes`` only those columns are read, with exactly those types; otherwise types are
    inferred from the whole file.
    """
    member = f"{ARCHIVE_DATA_DIR}/{name}.csv"
    try:
        raw = archive.read(member)
    except KeyError as exc:
        raise FileNotFoundError(f"{member} not found in {archive.filename}") from exc
    if dtypes is None:
        return pl.read_csv(raw, infer_schema_length=None)
    return pl.read_csv(raw, columns=list(dtypes), schema_overrides=dtypes)


def to_event_time(timestamp_seconds: pl.Expr) -> pl.Expr:
    """Epoch seconds (float) -> millisecond datetime in the platform's local timezone."""
    epoch_ms = (timestamp_seconds * MS_PER_SECOND).round().cast(pl.Int64)
    return (
        pl.from_epoch(epoch_ms, time_unit="ms")
        # from_epoch returns microsecond precision regardless of the input unit (polars 1.x).
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(LOCAL_TIMEZONE)
    )


def clean_interactions(raw: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Replace the source time/date/timestamp columns with one typed ``event_time`` and drop
    duplicate log rows. Re-watches (same pair at a different time) are kept, in source order.

    ``timestamp`` is the source of truth: ``time`` matches it in every row, while ``date`` is a
    log-partition label (it disagrees with the event's local date in ~0.1% of rows, and ~2.2k
    big_matrix events were filed under two partitions). Duplicates are therefore matched on the
    event fields only.

    Returns the cleaned frame and the number of duplicate rows removed.
    """
    deduped = raw.select(list(INTERACTION_SOURCE_DTYPES)).unique(maintain_order=True)
    cleaned = deduped.select(
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "watch_ratio",
        event_time=to_event_time(pl.col("timestamp")),
    )
    return cleaned, raw.height - deduped.height


def parse_int_list(column: str) -> pl.Expr:
    """Parse list strings such as ``"[27, 9]"`` into ``List[Int64]``."""
    return pl.col(column).str.json_decode(pl.List(pl.Int64))


def clean_daily_stats(raw: pl.DataFrame) -> pl.DataFrame:
    """Type the YYYYMMDD ``date`` as a Date and the null-padded collect counters as integers."""
    return raw.with_columns(
        pl.col("date").cast(pl.String).str.to_date("%Y%m%d"),
        pl.col(COLLECT_COLUMNS).cast(pl.Int64),
    )


def check_disjoint_pairs(train: pl.DataFrame, evaluation: pl.DataFrame) -> int:
    """Fail if any evaluation (user, video) pair was also seen in training.

    Returns the number of shared pairs, which is always 0 when this returns.
    """
    shared = (
        train.select(INTERACTION_KEYS)
        .unique()
        .join(evaluation.select(INTERACTION_KEYS).unique(), on=INTERACTION_KEYS, how="inner")
    )
    if shared.height:
        examples = shared.head(MAX_EXAMPLES_IN_ERROR).rows()
        raise DataLeakageError(
            f"{shared.height} (user_id, video_id) pair(s) appear in both training and "
            f"evaluation data, e.g. {examples}"
        )
    return shared.height


def check_referential_integrity(tables: dict[str, pl.DataFrame]) -> None:
    """Fail if an interaction references a user or video missing from its reference table."""
    for table in INTERACTION_TABLES:
        for key, parent in INTERACTION_REFERENCES:
            orphans = (
                tables[table].select(key).unique().join(tables[parent], on=key, how="anti")[key]
            )
            if orphans.len():
                raise DataIntegrityError(
                    f"{orphans.len()} {key} value(s) in {table} missing from {parent}, "
                    f"e.g. {orphans.head(MAX_EXAMPLES_IN_ERROR).to_list()}"
                )


def load_tables(archive_path: Path) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Read and clean every table; also return duplicate rows dropped per interaction table."""
    with zipfile.ZipFile(archive_path) as archive:
        big, big_dropped = clean_interactions(
            read_csv_from_zip(archive, "big_matrix", INTERACTION_SOURCE_DTYPES)
        )
        small, small_dropped = clean_interactions(
            read_csv_from_zip(archive, "small_matrix", INTERACTION_SOURCE_DTYPES)
        )
        categories = read_csv_from_zip(archive, "item_categories").select(
            "video_id", category_ids=parse_int_list("feat")
        )
        tables = {
            "big_matrix": big,
            "small_matrix": small,
            "users": read_csv_from_zip(archive, "user_features"),
            "video_categories": categories,
            "video_captions": read_csv_from_zip(archive, "kuairec_caption_category"),
            "video_daily_stats": clean_daily_stats(
                read_csv_from_zip(archive, "item_daily_features")
            ),
        }
    return tables, {"big_matrix": big_dropped, "small_matrix": small_dropped}


def summarize_interactions(df: pl.DataFrame, duplicates_dropped: int) -> dict[str, int | float]:
    users, videos = df["user_id"].n_unique(), df["video_id"].n_unique()
    cells = users * videos
    stats = df.select(
        watch_ratio_p50=pl.col("watch_ratio").median(),
        watch_ratio_p99=pl.col("watch_ratio").quantile(0.99),
        watch_ratio_max=pl.col("watch_ratio").max(),
        null_event_time=pl.col("event_time").null_count(),
    ).row(0, named=True)
    return {
        "rows": df.height,
        "users": users,
        "videos": videos,
        "density": round(df.select(INTERACTION_KEYS).n_unique() / cells, 4) if cells else 0.0,
        "duplicate_rows_dropped": duplicates_dropped,
        **{k: round(v, SUMMARY_DECIMALS) if isinstance(v, float) else v for k, v in stats.items()},
    }


def build_summary(
    tables: dict[str, pl.DataFrame], dropped: dict[str, int], leakage_pairs: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        name: summarize_interactions(tables[name], n) for name, n in dropped.items()
    }
    summary |= {name: {"rows": df.height} for name, df in tables.items() if name not in summary}
    summary["leakage_pairs"] = leakage_pairs
    return summary


def prepare(archive_path: Path, out_dir: Path, summary_path: Path) -> dict[str, Any]:
    """Load, clean, integrity-check and validate every table, then write parquet + summary.

    All checks and the summary run before the first write, so a failure leaves nothing behind.
    """
    tables, dropped = load_tables(archive_path)
    leakage_pairs = check_disjoint_pairs(tables["big_matrix"], tables["small_matrix"])
    check_referential_integrity(tables)
    validated = {name: TABLE_SCHEMAS[name].validate(df, lazy=True) for name, df in tables.items()}
    summary = build_summary(validated, dropped, leakage_pairs)

    write_parquet_tables(validated, out_dir)
    write_json(summary, summary_path)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    try:
        summary = prepare(
            params.data.archive_path, params.data.processed_dir, params.data.summary_path
        )
    except pa.errors.SchemaErrors as exc:
        logger.error("Data contract violated; nothing written.\n%s", exc.failure_cases)
        raise
    logger.info("Data summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

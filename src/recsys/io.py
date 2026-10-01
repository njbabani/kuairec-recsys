"""Shared writers for pipeline outputs, so every stage logs and serializes the same way."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

PARQUET_COMPRESSION = "zstd"


def write_parquet_tables(tables: Mapping[str, pl.DataFrame], out_dir: Path) -> None:
    """Write each frame to ``<out_dir>/<name>.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(out_dir / f"{name}.parquet", compression=PARQUET_COMPRESSION)
        logger.info("Wrote %s.parquet (%s rows)", name, f"{df.height:,}")


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a small JSON report (pretty-printed, trailing newline, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would otherwise be written as invalid JSON and break DVC metrics.
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")

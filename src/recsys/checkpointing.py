"""Shared pieces of resumable stages: content fingerprints, atomic writes and cleanup.

A stage that saves its progress must never resume from progress made for different inputs, and
a crash while saving must never leave a half-written file. Fingerprints cover both the settings
and the data; writes go to a temporary name that is renamed into place.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import polars as pl

logger = logging.getLogger(__name__)


def frame_fingerprint(frame: pl.DataFrame) -> str:
    """Content hash of a frame (stable within one Polars version, which is all resuming needs)."""
    digest = hashlib.sha256(str(frame.schema).encode())
    digest.update(frame.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


def fingerprint(settings: Any, frames: Sequence[pl.DataFrame]) -> str:
    """One hash of JSON-serialisable ``settings`` and the content of ``frames``."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for frame in frames:
        digest.update(frame_fingerprint(frame).encode())
    return digest.hexdigest()


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Run ``write`` on a temporary file next to ``path``, then rename it into place.

    A crash mid-write leaves the previous file untouched; a failed write removes its partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        write(partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    text = json.dumps(payload, allow_nan=False)
    atomic_write(path, lambda partial: partial.write_text(text, encoding="utf-8"))


def remove_if_empty(directory: Path | None) -> None:
    """Drop a checkpoint folder once nothing is left in it (never touches other files)."""
    if directory is not None and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


@dataclass
class ProgressLog:
    """Results of finished units of work, saved after each one so an interrupted stage resumes.

    Entries are only reused when the fingerprint (settings and data) still matches.
    """

    path: Path
    fingerprint: str
    entries: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path, fingerprint: str) -> Self:
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            entries = saved.get("entries")
            if saved.get("fingerprint") == fingerprint and isinstance(entries, dict):
                logger.info("Resuming from %s: %d entries already done", path, len(entries))
                return cls(path, fingerprint, entries)
            logger.warning("Discarding progress in %s: other settings, data or format", path)
        return cls(path, fingerprint)

    def record(self, key: str, value: Any) -> None:
        self.entries[key] = value
        write_json_atomic({"fingerprint": self.fingerprint, "entries": self.entries}, self.path)

"""Saving and loading two-tower state: training checkpoints and finished models.

* Files are written to a temporary name and then renamed, so a crash mid-write never leaves a
  half-written checkpoint where a good one used to be.
* Files are read with ``weights_only=True``: only tensors and plain containers are accepted,
  so opening a checkpoint can never run code hidden in a pickle.
* A fingerprint of the training data and feature tables is stored with each training
  checkpoint, so training never resumes from a checkpoint made for different inputs.
"""

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from recsys.models.two_tower.encoding import EntityFeatures, Vocabulary

FORMAT_VERSION = 1
TRAINING_STATE = "two_tower_training_state"
MODEL = "two_tower_model"


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    torch.save(payload, partial)
    partial.replace(path)


def remove_if_empty(directory: Path | None) -> None:
    """Drop a checkpoint folder once nothing is left in it (never touches other files)."""
    if directory is not None and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def frame_fingerprint(frame: pl.DataFrame) -> str:
    """Content hash of a frame (stable within one Polars version, which is all resuming needs)."""
    digest = hashlib.sha256(str(frame.schema).encode())
    digest.update(frame.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


def frame_to_tensor(frame: pl.DataFrame) -> torch.Tensor:
    """A frame as Arrow IPC bytes in a uint8 tensor: exact types, and safe to load."""
    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def frame_from_tensor(tensor: torch.Tensor) -> pl.DataFrame:
    return pl.read_ipc(io.BytesIO(tensor.numpy().tobytes()))


def entity_to_payload(features: EntityFeatures) -> dict[str, Any]:
    return {
        "ids": list(features.ids.values),
        "categorical": torch.tensor(features.categorical),
        "cardinalities": list(features.cardinalities),
        "numeric": torch.tensor(features.numeric),
        "multi_valued": torch.tensor(features.multi_valued),
        "multi_valued_cardinality": features.multi_valued_cardinality,
    }


def entity_from_payload(payload: dict[str, Any]) -> EntityFeatures:
    return EntityFeatures(
        ids=Vocabulary(tuple(payload["ids"])),
        categorical=payload["categorical"].numpy().astype(np.int64),
        cardinalities=tuple(payload["cardinalities"]),
        numeric=payload["numeric"].numpy().astype(np.float32),
        multi_valued=payload["multi_valued"].numpy().astype(np.int64),
        multi_valued_cardinality=int(payload["multi_valued_cardinality"]),
    )

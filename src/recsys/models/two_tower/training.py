"""Training pieces for the two-tower model: device choice, feature roles, examples and epochs."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from recsys.models.two_tower.network import (
    TwoTowerNetwork,
    in_batch_softmax_loss,
    pointwise_loss,
)

LOSSES = ("bce", "softmax")
EMPTY = torch.empty(0)


def resolve_device(requested: str) -> torch.device:
    """``auto`` uses the Apple-silicon GPU when PyTorch can see one, otherwise the CPU."""
    available = torch.backends.mps.is_available()
    if requested == "auto":
        return torch.device("mps" if available else "cpu")
    if requested == "mps" and not available:
        raise ValueError("MPS (the Apple-silicon GPU) is not available; use device cpu or auto")
    return torch.device(requested)


@dataclass(frozen=True)
class FeatureRoles:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    multi_valued: str | None


def feature_roles(table: pl.DataFrame, key: str) -> FeatureRoles:
    """How to encode each column, from its type: text is categorical, numbers are numeric and
    a list column (at most one) is multi-valued categorical."""
    categorical, numeric, lists = [], [], []
    for name, dtype in table.schema.items():
        if name == key:
            continue
        if dtype in (pl.String, pl.Categorical):
            categorical.append(name)
        elif dtype.is_numeric():
            numeric.append(name)
        elif isinstance(dtype, pl.List):
            lists.append(name)
        else:
            raise TypeError(f"Cannot encode feature column {name!r} of type {dtype}")
    if len(lists) > 1:
        raise ValueError(f"At most one list column is supported, got {lists}")
    return FeatureRoles(tuple(categorical), tuple(numeric), next(iter(lists), None))


def training_rows(loss: str, labels: np.ndarray) -> np.ndarray:
    """Which logged views a loss learns from: all of them for bce, the positives for softmax."""
    return labels.copy() if loss == "softmax" else np.ones(len(labels), dtype=bool)


@dataclass(frozen=True)
class TrainingExamples:
    """Training rows, already on the training device."""

    users: torch.Tensor
    videos: torch.Tensor
    labels: torch.Tensor  # bce: 0/1 per row (empty for softmax)
    log_q: torch.Tensor  # softmax: log sampling probability of each row's video (empty for bce)

    def __len__(self) -> int:
        return len(self.users)


def build_examples(
    loss: str,
    users: np.ndarray,
    videos: np.ndarray,
    labels: np.ndarray,
    n_videos: int,
    device: torch.device,
) -> TrainingExamples:
    rows = training_rows(loss, labels)
    users, videos = users[rows], videos[rows]
    if loss == "bce":
        label_values, log_q = torch.tensor(labels.astype(np.float32), device=device), EMPTY
    else:
        # In-batch negatives are drawn in proportion to how often each video was liked.
        liked_count = np.bincount(videos, minlength=n_videos)
        log_q = torch.tensor(np.log(liked_count[videos] / len(videos)).astype(np.float32))
        label_values, log_q = EMPTY, log_q.to(device)
    return TrainingExamples(
        torch.tensor(users, device=device), torch.tensor(videos, device=device), label_values, log_q
    )


def train_epoch(
    network: TwoTowerNetwork,
    examples: TrainingExamples,
    optimizer: torch.optim.Optimizer,
    shuffler: torch.Generator,
    *,
    batch_size: int,
    loss: str,
    temperature: float | None,
) -> float:
    """One pass over the examples in a fresh random order; returns the mean batch loss."""
    network.train()
    device = examples.users.device
    order = torch.randperm(len(examples), generator=shuffler).to(device)
    total = torch.zeros((), device=device)
    batches = 0
    for start in range(0, len(examples), batch_size):
        rows = order[start : start + batch_size]
        videos = examples.videos[rows]
        user_vectors = network.users(examples.users[rows])
        video_vectors = network.videos(videos)
        if loss == "bce":
            batch_loss = pointwise_loss(user_vectors, video_vectors, examples.labels[rows])
        else:
            batch_loss = in_batch_softmax_loss(
                user_vectors, video_vectors, videos, examples.log_q[rows], float(temperature)
            )
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        total += batch_loss.detach()  # summed on the device: no per-batch GPU sync
        batches += 1
    return float(total.item()) / batches

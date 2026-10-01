"""Two-tower neural recommender in PyTorch, trained on the CPU or the Apple-silicon GPU (MPS).

Two training objectives answer different questions:

* ``loss="bce"``: *will this user like this video, given it was shown?* Trained on every
  logged view (positives and non-positives). This is exactly the question the evaluation asks.
* ``loss="softmax"``: *which video did this user like?* Trained on positives only, with the
  batch's other positives as negatives and the logQ popularity correction: the standard
  retrieval objective of industrial two-tower models.

Checkpointing: pass ``checkpoint_path`` to :meth:`fit` and training state is saved after every
epoch; a later ``fit`` with the same settings and data resumes after the last saved epoch.
:meth:`save` and :meth:`load` store and restore a finished model, features included.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import torch

from recsys.models.two_tower.checkpoint import (
    FORMAT_VERSION,
    MODEL,
    TRAINING_STATE,
    entity_from_payload,
    entity_to_payload,
    frame_fingerprint,
    frame_from_tensor,
    frame_to_tensor,
    load_payload,
    save_atomic,
)
from recsys.models.two_tower.encoding import EntityFeatures, encode_entities
from recsys.models.two_tower.network import Tower, TwoTowerNetwork
from recsys.models.two_tower.training import (
    LOSSES,
    TrainingExamples,
    build_examples,
    feature_roles,
    resolve_device,
    train_epoch,
    training_rows,
)

logger = logging.getLogger(__name__)

SCORE_CHUNK_ROWS = 262_144  # pairs scored per vectorised step (bounds peak memory)
HYPERPARAMETERS = (
    "loss",
    "embedding_dim",
    "hidden_dim",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "temperature",
    "seed",
    "device",
)
# Changing these does not invalidate epochs already trained, so they stay out of the fingerprint.
RESUMABLE_CHANGES = ("epochs", "device")
EpochCallback = Callable[[int, float], None]

__all__ = ["TwoTowerRecommender", "resolve_device"]


def _trained_entities(size: int, index: np.ndarray) -> np.ndarray:
    trained = np.zeros(size, dtype=bool)
    trained[index] = True
    return trained


class TwoTowerRecommender:
    name = "two_tower"

    def __init__(
        self,
        user_features: pl.DataFrame,
        video_features: pl.DataFrame,
        *,
        loss: str,
        embedding_dim: int,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        device: str,
        temperature: float | None = None,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
        if loss == "softmax" and temperature is None:
            raise ValueError("The softmax loss needs a temperature")
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {batch_size}")
        self.user_features = user_features
        self.video_features = video_features
        self.loss = loss
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.seed = seed
        self.device = device
        self.train_losses: list[float] = []
        self._users: EntityFeatures | None = None
        self._videos: EntityFeatures | None = None
        self._network: TwoTowerNetwork | None = None

    @property
    def network(self) -> TwoTowerNetwork:
        if self._network is None:
            raise RuntimeError("Call fit() (or load a saved model) before using the model")
        return self._network

    def hyperparameters(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in HYPERPARAMETERS}

    def fit(
        self,
        train: pl.DataFrame,
        on_epoch_end: EpochCallback | None = None,
        checkpoint_path: Path | None = None,
    ) -> Self:
        """Train for ``epochs``; ``on_epoch_end(epoch, mean_loss)`` may call :meth:`score`.

        With ``checkpoint_path``, the training state is saved after every epoch (after the
        callback ran), and a matching checkpoint left by an interrupted run is resumed.
        """
        if not train["is_positive"].any():
            raise ValueError("Training data has no positive views to learn from")
        device = resolve_device(self.device)
        examples = self._build(train, device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        shuffler = torch.Generator().manual_seed(self.seed)
        self.train_losses = []
        fingerprint = self._fingerprint(train) if checkpoint_path is not None else ""
        first_epoch = 1
        if checkpoint_path is not None:
            first_epoch = self._resume(checkpoint_path, fingerprint, optimizer, shuffler)

        for epoch in range(first_epoch, self.epochs + 1):
            self.train_losses.append(self._train_one_epoch(examples, optimizer, shuffler))
            logger.info(
                "two_tower epoch %d/%d loss %.4f", epoch, self.epochs, self.train_losses[-1]
            )
            if on_epoch_end is not None:
                on_epoch_end(epoch, self.train_losses[-1])
            if checkpoint_path is not None:
                state = self._training_state(epoch, fingerprint, optimizer, shuffler)
                save_atomic(state, checkpoint_path)
        return self

    def _train_one_epoch(
        self,
        examples: TrainingExamples,
        optimizer: torch.optim.Optimizer,
        shuffler: torch.Generator,
    ) -> float:
        return train_epoch(
            self.network,
            examples,
            optimizer,
            shuffler,
            batch_size=self.batch_size,
            loss=self.loss,
            temperature=self.temperature,
        )

    def _build(self, train: pl.DataFrame, device: torch.device) -> TrainingExamples:
        """Encode the features, create the network and turn ``train`` into training examples."""
        self._users = self._encode(self.user_features, "user_id", train)
        self._videos = self._encode(self.video_features, "video_id", train)
        users = self._users.ids.encode(train["user_id"])
        videos = self._videos.ids.encode(train["video_id"])
        labels = train["is_positive"].to_numpy()
        # An entity is "trained" only if the loss actually sees it (softmax sees positives only);
        # every other entity keeps a zero id embedding and is described by its features alone.
        rows = training_rows(self.loss, labels)
        with torch.random.fork_rng(devices=[]):  # seed the weights, not the caller's RNG
            torch.manual_seed(self.seed)
            network = TwoTowerNetwork(
                self._tower(self._users, _trained_entities(self._users.size, users[rows])),
                self._tower(self._videos, _trained_entities(self._videos.size, videos[rows])),
                normalize=self.loss == "softmax",
            )
        self._network = network.to(device)
        return build_examples(self.loss, users, videos, labels, self._videos.size, device)

    @staticmethod
    def _encode(table: pl.DataFrame, key: str, train: pl.DataFrame) -> EntityFeatures:
        roles = feature_roles(table, key)
        return encode_entities(
            table,
            key,
            train[key].unique(),
            categorical=roles.categorical,
            numeric=roles.numeric,
            multi_valued=roles.multi_valued,
        )

    def _tower(self, features: EntityFeatures, trained: np.ndarray) -> Tower:
        return Tower(features, trained, self.embedding_dim, self.hidden_dim)

    def _fingerprint(self, train: pl.DataFrame) -> str:
        settings = {
            name: value
            for name, value in self.hyperparameters().items()
            if name not in RESUMABLE_CHANGES
        }
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for frame in (
            train.select("user_id", "video_id", "is_positive"),
            self.user_features,
            self.video_features,
        ):
            digest.update(frame_fingerprint(frame).encode())
        return digest.hexdigest()

    def _resume(
        self,
        path: Path,
        fingerprint: str,
        optimizer: torch.optim.Optimizer,
        shuffler: torch.Generator,
    ) -> int:
        """Restore a matching checkpoint; returns the first epoch still to train."""
        if not path.is_file():
            return 1
        state = load_payload(path)
        matches = (
            state.get("kind") == TRAINING_STATE
            and state.get("format_version") == FORMAT_VERSION
            and state.get("fingerprint") == fingerprint
            and int(state.get("epoch", 0)) <= self.epochs
        )
        if not matches:
            logger.warning("Not resuming from %s: it was made for other settings or data", path)
            return 1
        self.network.load_state_dict(state["network"])
        optimizer.load_state_dict(state["optimizer"])
        shuffler.set_state(state["shuffler"])
        self.train_losses = [float(loss) for loss in state["train_losses"]]
        logger.info("Resuming two_tower training from %s after epoch %d", path, state["epoch"])
        return int(state["epoch"]) + 1

    def _training_state(
        self,
        epoch: int,
        fingerprint: str,
        optimizer: torch.optim.Optimizer,
        shuffler: torch.Generator,
    ) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "kind": TRAINING_STATE,
            "fingerprint": fingerprint,
            "epoch": epoch,
            "network": self.network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "shuffler": shuffler.get_state(),
            "train_losses": list(self.train_losses),
        }

    def save(self, path: Path) -> None:
        """Store the fitted model with its encoded features, so :meth:`load` needs nothing else."""
        users, videos = self._entities()
        save_atomic(
            {
                "format_version": FORMAT_VERSION,
                "kind": MODEL,
                "hyperparameters": self.hyperparameters(),
                "users": entity_to_payload(users),
                "videos": entity_to_payload(videos),
                "user_features": frame_to_tensor(self.user_features),
                "video_features": frame_to_tensor(self.video_features),
                "network": self.network.state_dict(),
                "train_losses": list(self.train_losses),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> Self:
        payload = load_payload(path)
        if payload.get("kind") != MODEL or payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a saved two-tower model (format {FORMAT_VERSION})")
        model = cls(
            frame_from_tensor(payload["user_features"]),
            frame_from_tensor(payload["video_features"]),
            **(payload["hyperparameters"] | {"device": device}),
        )
        model._users = entity_from_payload(payload["users"])
        model._videos = entity_from_payload(payload["videos"])
        network = TwoTowerNetwork(
            model._tower(model._users, np.zeros(model._users.size, dtype=bool)),
            model._tower(model._videos, np.zeros(model._videos.size, dtype=bool)),
            normalize=model.loss == "softmax",
        )
        network.load_state_dict(payload["network"])
        model._network = network.to(resolve_device(device))
        model.train_losses = [float(loss) for loss in payload["train_losses"]]
        return model

    def _entities(self) -> tuple[EntityFeatures, EntityFeatures]:
        if self._users is None or self._videos is None:
            raise RuntimeError("Call fit() (or load a saved model) before using the model")
        return self._users, self._videos

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        network = self.network
        user_features, video_features = self._entities()
        users, user_rows = np.unique(
            user_features.ids.encode(pairs["user_id"]), return_inverse=True
        )
        videos, video_rows = np.unique(
            video_features.ids.encode(pairs["video_id"]), return_inverse=True
        )
        user_vectors = self._embed(network.users, users)
        video_vectors = self._embed(network.videos, videos)

        scores = np.empty(pairs.height, dtype=np.float64)
        for start in range(0, pairs.height, SCORE_CHUNK_ROWS):
            chunk = slice(start, start + SCORE_CHUNK_ROWS)
            scores[chunk] = np.einsum(
                "ij,ij->i", user_vectors[user_rows[chunk]], video_vectors[video_rows[chunk]]
            )
        return pl.Series("score", scores)

    def _embed(
        self, encoder: Callable[[torch.Tensor], torch.Tensor], index: np.ndarray
    ) -> np.ndarray:
        network = self.network
        network.eval()
        with torch.no_grad():
            vectors = encoder(torch.as_tensor(index, device=next(network.parameters()).device))
        return vectors.cpu().numpy().astype(np.float64)

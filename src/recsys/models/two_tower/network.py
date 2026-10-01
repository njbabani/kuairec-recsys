"""The two towers and the two training objectives.

Each tower turns one entity (a user or a video) into a vector from its id embedding plus its
features; a pair's score is the dot product of the two vectors. Because videos are encoded
independently of users, every video vector can be precomputed and searched with a nearest-
neighbour index, which is why this architecture is the standard first stage of large
recommenders.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (the conventional alias)
from torch import nn

from recsys.models.two_tower.encoding import UNKNOWN, EntityFeatures

FEATURE_EMBEDDING_DIM = 8
ID_EMBEDDING_INIT_STD = 0.1


class Tower(nn.Module):
    """id embedding + categorical embeddings + mean of multi-valued embeddings + numerics -> MLP.

    Entities with no training interactions keep an all-zero id embedding (they never receive a
    gradient), so their vector comes from their features alone: that is how a brand-new video
    still gets a sensible score.
    """

    def __init__(
        self, features: EntityFeatures, trained: np.ndarray, embedding_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        if trained.shape != (features.size,):
            raise ValueError(
                f"trained needs one flag per entity ({features.size}), got {trained.shape}"
            )

        # One shared table for every categorical column: column c's codes are shifted by offset c.
        # Feature arrays are derived data, not weights: copied in, and left out of state_dict().
        offsets = np.cumsum((0, *features.cardinalities[:-1]), dtype=np.int64)
        categorical = features.categorical + offsets
        self.register_buffer("categorical", torch.tensor(categorical), persistent=False)
        self.register_buffer("multi_valued", torch.tensor(features.multi_valued), persistent=False)
        self.register_buffer("numeric", torch.tensor(features.numeric), persistent=False)

        self.id_embedding = nn.Embedding(features.size, embedding_dim)
        nn.init.normal_(self.id_embedding.weight, std=ID_EMBEDDING_INIT_STD)
        untrained = ~torch.as_tensor(trained)
        untrained[UNKNOWN] = True
        with torch.no_grad():
            self.id_embedding.weight[untrained] = 0.0

        n_categorical = len(features.cardinalities)
        self.categorical_embedding = (
            nn.Embedding(sum(features.cardinalities), FEATURE_EMBEDDING_DIM)
            if n_categorical
            else None
        )
        self.multi_valued_embedding = (
            nn.Embedding(features.multi_valued_cardinality, FEATURE_EMBEDDING_DIM)
            if features.multi_valued_cardinality
            else None
        )
        _zero_rows_unused_by(self.categorical_embedding, categorical[trained])
        _zero_rows_unused_by(self.multi_valued_embedding, features.multi_valued[trained])
        input_dim = (
            embedding_dim
            + FEATURE_EMBEDDING_DIM * n_categorical
            + (FEATURE_EMBEDDING_DIM if self.multi_valued_embedding is not None else 0)
            + features.numeric.shape[1]
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        parts = [self.id_embedding(index)]
        if self.categorical_embedding is not None:
            parts.append(self.categorical_embedding(self.categorical[index]).flatten(1))
        if self.multi_valued_embedding is not None:
            codes = self.multi_valued[index]
            present = (codes != UNKNOWN).unsqueeze(-1).float()
            summed = (self.multi_valued_embedding(codes) * present).sum(dim=1)
            parts.append(summed / present.sum(dim=1).clamp(min=1.0))
        parts.append(self.numeric[index])
        return self.mlp(torch.cat(parts, dim=1))


def _zero_rows_unused_by(embedding: nn.Embedding | None, codes: np.ndarray) -> None:
    """Zero the rows no trained entity uses: they would never get a gradient, so a random
    start would only add noise to the vectors of the entities that do use them."""
    if embedding is None:
        return
    unused = np.ones(embedding.num_embeddings, dtype=bool)
    unused[np.unique(codes)] = False
    with torch.no_grad():
        embedding.weight[torch.from_numpy(unused)] = 0.0


class TwoTowerNetwork(nn.Module):
    """A user tower and a video tower; a pair's score is the dot product of their vectors."""

    def __init__(self, user_tower: Tower, video_tower: Tower, normalize: bool) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.video_tower = video_tower
        self.normalize = normalize  # unit vectors (cosine similarity), used with the softmax loss

    def users(self, index: torch.Tensor) -> torch.Tensor:
        return self._maybe_normalize(self.user_tower(index))

    def videos(self, index: torch.Tensor) -> torch.Tensor:
        return self._maybe_normalize(self.video_tower(index))

    def _maybe_normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(vectors, dim=1) if self.normalize else vectors


def pointwise_loss(users: torch.Tensor, videos: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Will this user like this video, given it was shown? Logistic loss on every logged view."""
    return F.binary_cross_entropy_with_logits((users * videos).sum(dim=1), labels)


def in_batch_softmax_loss(
    users: torch.Tensor,
    videos: torch.Tensor,
    video_index: torch.Tensor,
    log_q: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Which of these videos did this user like? The batch's other positives are the negatives.

    Popular videos turn up as negatives far more often than rare ones, which would teach the
    model to bury them; subtracting each video's log sampling probability (``log_q``, the logQ
    correction of Yi et al., 2019) removes that bias. Other rows holding the *same* video are
    masked out, so a video is never its own negative.
    """
    logits = users @ videos.T / temperature - log_q.unsqueeze(0)
    same_video = video_index.unsqueeze(0) == video_index.unsqueeze(1)
    own_row = torch.eye(len(video_index), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_video & ~own_row, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(video_index), device=logits.device))

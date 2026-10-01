"""Implicit-feedback matrix factorisation (ALS, Hu, Koren & Volinsky 2008) via ``implicit``.

Each user and video gets a vector of ``factors`` numbers; a pair's score is their dot product.
Training signal: each user's *positive* (duration-debiased) views. As in Hu et al., an observed
pair gets confidence ``1 + alpha * count`` and every other pair confidence 1, so ``alpha = 0``
fits the plain binary matrix and larger values trust observed positives more. (``implicit``
reads the matrix values as the full confidence, hence the explicit ``1 +``.)

Users without any positive training view, and users never seen in training, get the average
user vector: a popularity-like ranking from the same model instead of an all-zero vector that
would rank every candidate equally. Videos never seen in training rank last.
"""

from typing import Self

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy import sparse


class ALSRecommender:
    name = "als"

    def __init__(
        self, factors: int, regularization: float, alpha: float, iterations: int, seed: int
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self._users: pl.DataFrame | None = None
        self._videos: pl.DataFrame | None = None
        self._user_factors: np.ndarray | None = None
        self._video_factors: np.ndarray | None = None
        self._average_user: np.ndarray | None = None

    def fit(self, train: pl.DataFrame) -> Self:
        positives = train.filter(pl.col("is_positive")).group_by("user_id", "video_id").len()
        if positives.is_empty():
            raise ValueError("ALS needs at least one positive interaction to fit")

        users = _index(positives["user_id"], "user_row")  # only users with something to learn
        videos = _index(train["video_id"], "video_col")
        coded = positives.join(users, on="user_id").join(videos, on="video_id")
        confidence = sparse.csr_matrix(
            (
                (1 + self.alpha * coded["len"].to_numpy()).astype(np.float32),
                (coded["user_row"].to_numpy(), coded["video_col"].to_numpy()),
            ),
            shape=(users.height, videos.height),
        )
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            calculate_training_loss=False,
            use_gpu=False,  # factors must come back as numpy arrays
        )
        model.fit(confidence, show_progress=False)

        self._users, self._videos = users, videos
        self._user_factors = np.asarray(model.user_factors)
        self._video_factors = np.asarray(model.item_factors)
        self._average_user = self._user_factors.mean(axis=0)
        return self

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        if self._user_factors is None or self._video_factors is None:
            raise RuntimeError("Call fit() before score()")
        coded = (
            pairs.select("user_id", "video_id")
            .join(self._users, on="user_id", how="left", maintain_order="left")
            .join(self._videos, on="video_id", how="left", maintain_order="left")
        )
        known_video = coded["video_col"].is_not_null().to_numpy()
        rows = coded.filter(pl.Series(known_video))
        user_rows = rows["user_row"]
        user_vectors = np.where(
            user_rows.is_null().to_numpy()[:, None],
            self._average_user,
            self._user_factors[user_rows.fill_null(0).to_numpy()],
        )
        scores = np.full(pairs.height, -np.inf)
        scores[known_video] = np.einsum(
            "ij,ij->i", user_vectors, self._video_factors[rows["video_col"].to_numpy()]
        )
        return pl.Series("score", scores)


def _index(ids: pl.Series, position: str) -> pl.DataFrame:
    """Map each distinct id to a dense 0-based row/column position."""
    unique = ids.unique().sort()
    return pl.DataFrame({ids.name: unique, position: np.arange(unique.len())})

"""Several two-tower models trained with different seeds, scored by their average.

One training run of the two-tower model varies by about +-0.006 NDCG@10 from seed to seed, as
much as the gaps between the models being compared, so a single run is a noisy measurement.
Averaging a few runs removes most of that noise. The average of the members' dot products
equals one dot product of their concatenated vectors (divided by the member count), so the
ensemble is itself a two-tower model and can still be served from a nearest-neighbour index.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import polars as pl

from recsys.checkpointing import remove_if_empty
from recsys.models.two_tower.recommender import TwoTowerRecommender

MEMBER_PREFIX = "seed_"


def _member_file(directory: Path, seed: int) -> Path:
    return directory / f"{MEMBER_PREFIX}{seed}.pt"


class TwoTowerEnsemble:
    name = "two_tower"

    def __init__(
        self, members: Sequence[TwoTowerRecommender], checkpoint_dir: Path | None = None
    ) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        seeds = [member.seed for member in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Members need distinct seeds, got {seeds}")
        settings = [
            {name: value for name, value in member.hyperparameters().items() if name != "seed"}
            for member in members
        ]
        if any(other != settings[0] for other in settings[1:]):
            raise ValueError("Members must differ only in their seed")
        self.members = tuple(members)
        self.checkpoint_dir = checkpoint_dir

    def fit(self, train: pl.DataFrame) -> Self:
        """Train every member; with ``checkpoint_dir``, an interrupted fit resumes each member
        (finished ones are restored, not retrained) and the resume state is removed at the end."""
        for member in self.members:
            member.fit(train, checkpoint_path=self._checkpoint(member))
        for member in self.members:
            checkpoint = self._checkpoint(member)
            if checkpoint is not None:
                checkpoint.unlink(missing_ok=True)
        remove_if_empty(self.checkpoint_dir)
        return self

    def _checkpoint(self, member: TwoTowerRecommender) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return _member_file(self.checkpoint_dir, member.seed)

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        scores = [member.score(pairs) for member in self.members]
        return (sum(scores[1:], scores[0]) / len(scores)).alias("score")

    def save(self, directory: Path) -> None:
        for member in self.members:
            member.save(_member_file(directory, member.seed))

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> Self:
        paths = sorted(
            directory.glob(f"{MEMBER_PREFIX}*.pt"),
            key=lambda path: int(path.stem.removeprefix(MEMBER_PREFIX)),
        )
        if not paths:
            raise FileNotFoundError(f"No saved two-tower members in {directory}")
        return cls([TwoTowerRecommender.load(path, device=device) for path in paths])

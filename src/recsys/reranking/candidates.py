"""Inputs of the re-ranker: precomputed retrieval scores and the rows it learns from."""

import polars as pl

PAIR_KEYS = ["user_id", "video_id"]


def require_finite(scores: pl.Series, stage: str) -> None:
    """A NaN, null or infinite score would sort arbitrarily, so refuse it outright."""
    if scores.null_count() or not scores.is_finite().all():
        raise ValueError(f"{stage} scores must be finite (found NaN, null or infinite values)")


class PrecomputedScorer:
    """Scores looked up from a table another stage computed (``user_id``, ``video_id``, ``score``).

    The ``retrieval`` stage scores every pair the re-ranker needs with the two-tower model, so
    the re-ranker never has to load PyTorch. A pair missing from the table is an error: a
    silently invented score would quietly change the ranking.
    """

    def __init__(self, table: pl.DataFrame, name: str) -> None:
        if table.select(PAIR_KEYS).is_duplicated().any():
            raise ValueError(f"The {name} score table has duplicate (user_id, video_id) pairs")
        require_finite(table["score"], f"Precomputed {name}")
        self.table = table.select(*PAIR_KEYS, "score")
        self.name = name

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        scored = pairs.select(PAIR_KEYS).join(
            self.table, on=PAIR_KEYS, how="left", maintain_order="left"
        )
        missing = scored["score"].null_count()
        if missing:
            raise ValueError(f"No precomputed {self.name} score for {missing} pair(s)")
        return scored["score"]


def ranker_training_rows(valid: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Validation views shaped like the evaluation, which has one reaction per (user, video):

    * a pair viewed several times in the week keeps the reaction to its first view;
    * no re-watched training pairs and no videos unseen in training (the fully observed users
      have neither);
    * only users who both liked and skipped something (one label has nothing to order).
    """
    pairs_seen = train.select(PAIR_KEYS).unique()
    videos_seen = train.select("video_id").unique()
    rows = (
        valid.sort("event_time", *PAIR_KEYS)
        .unique(PAIR_KEYS, keep="first", maintain_order=True)
        .select(*PAIR_KEYS, "is_positive")
        .join(pairs_seen, on=PAIR_KEYS, how="anti")
        .join(videos_seen, on="video_id", how="semi")
    )
    mixed = (
        rows.group_by("user_id")
        .agg(pl.col("is_positive").n_unique().alias("labels"))
        .filter(pl.col("labels") == 2)
        .select("user_id")
    )
    return rows.join(mixed, on="user_id", how="semi")

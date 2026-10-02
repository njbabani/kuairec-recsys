"""Two-stage recommendation: retrieve a short list cheaply, then re-rank it carefully.

The retriever (the two-tower model) scores every candidate and keeps each user's top N; only
those N go through the feature pipeline and the LightGBM re-ranker. The re-ranker can only
reorder what retrieval surfaced, so :func:`retrieval_recall` measures that ceiling.
"""

from dataclasses import dataclass

import lightgbm as lgb
import polars as pl

from recsys.models.base import Scorer
from recsys.reranking.candidates import require_finite
from recsys.reranking.features import FEATURE_COLUMNS, RankingFeatures
from recsys.reranking.ranker import predict


class LightGBMReranker:
    """The re-ranker as a scorer: build the features for each pair, then predict."""

    name = "reranker"

    def __init__(self, features: RankingFeatures, booster: lgb.Booster) -> None:
        self.features = features
        self.booster = booster

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        matrix = self.features.transform(pairs).select(FEATURE_COLUMNS)
        return pl.Series("score", predict(self.booster, matrix))


def mark_top_n_per_user(frame: pl.DataFrame, score: str, top_n: int) -> pl.DataFrame:
    """Add ``retrieved``: True for each user's ``top_n`` best-scored rows (ties: lower video id).

    The rows come back grouped by user; callers that need the original order sort afterwards.
    """
    return frame.sort(["user_id", score, "video_id"], descending=[False, True, False]).with_columns(
        retrieved=pl.int_range(1, pl.len() + 1).over("user_id") <= top_n
    )


class TwoStageRecommender:
    """Each user's shortlist, in the re-ranker's order, then every other candidate in retrieval
    order. Scores are negative within-user positions: only the order inside a user matters."""

    name = "two_stage"

    def __init__(self, retriever: Scorer, reranker: Scorer, top_n: int) -> None:
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        self.retriever = retriever
        self.reranker = reranker
        self.top_n = top_n

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        candidates = pairs.select("user_id", "video_id")
        retrieval = self.retriever.score(candidates)
        require_finite(retrieval, "Retrieval")
        frame = mark_top_n_per_user(
            candidates.with_row_index("row").with_columns(retrieval=retrieval),
            "retrieval",
            self.top_n,
        )
        shortlist = frame.filter("retrieved")
        reranked = self.reranker.score(shortlist.select("user_id", "video_id"))
        require_finite(reranked, "Re-ranker")
        stages = pl.concat(
            [
                shortlist.select("row", "user_id", "video_id", stage=pl.lit(0), key=reranked),
                frame.filter(~pl.col("retrieved")).select(
                    "row", "user_id", "video_id", stage=pl.lit(1), key=pl.col("retrieval")
                ),
            ]
        )
        positions = stages.sort(
            ["user_id", "stage", "key", "video_id"], descending=[False, False, True, False]
        ).with_columns(position=pl.int_range(1, pl.len() + 1).over("user_id"))
        return positions.sort("row").select(score=-pl.col("position").cast(pl.Float64))["score"]


@dataclass(frozen=True)
class RetrievalRecall:
    recall: float  # mean share of each user's liked videos that made their top N
    ceiling: float  # mean of min(1, N / liked videos): the most a top-N list could hold


def retrieval_recall(labeled: pl.DataFrame, retriever: Scorer, top_n: int) -> RetrievalRecall:
    """Over users with at least one positive: how many of their positives the top N holds."""
    scores = retriever.score(labeled.select("user_id", "video_id"))
    require_finite(scores, "Retrieval")
    per_user = (
        mark_top_n_per_user(labeled.with_columns(retrieval=scores), "retrieval", top_n)
        .group_by("user_id")
        .agg(
            positives=pl.col("is_positive").sum(),
            found=(pl.col("is_positive") & pl.col("retrieved")).sum(),
        )
        .filter(pl.col("positives") > 0)
    )
    if per_user.is_empty():
        raise ValueError("No user has a positive, so retrieval recall is undefined")
    ceiling = pl.min_horizontal(pl.lit(1.0), top_n / pl.col("positives"))
    return RetrievalRecall(
        recall=float((per_user["found"] / per_user["positives"]).mean()),
        ceiling=float(per_user.select(ceiling.mean()).item()),
    )

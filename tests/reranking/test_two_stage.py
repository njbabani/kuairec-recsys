import polars as pl
import pytest

from recsys.reranking.two_stage import TwoStageRecommender, retrieval_recall


class TableScorer:
    """Scores from a {video_id: score} table; counts how many pairs it was asked to score."""

    name = "table"

    def __init__(self, table: dict[int, float]) -> None:
        self.table = table
        self.pairs_seen = 0

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        self.pairs_seen += pairs.height
        return pl.Series("score", [self.table[video] for video in pairs["video_id"]], pl.Float64)


# Retrieval prefers 1 > 2 > 3 > 4 > 5; the re-ranker prefers 3 > 2 > 1 (and dislikes 4, 5).
RETRIEVAL = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0}
RERANK = {1: 0.1, 2: 0.2, 3: 0.3, 4: 9.0, 5: 9.0}


def order(scores: pl.Series, pairs: pl.DataFrame) -> list[int]:
    return pairs.with_columns(score=scores).sort("score", descending=True)["video_id"].to_list()


def test_only_the_retrieved_top_n_are_reranked_and_they_come_first():
    pairs = pl.DataFrame({"user_id": [7] * 5, "video_id": [5, 4, 3, 2, 1]})
    reranker = TableScorer(RERANK)

    scores = TwoStageRecommender(TableScorer(RETRIEVAL), reranker, top_n=3).score(pairs)

    assert order(scores, pairs) == [3, 2, 1, 4, 5]
    assert reranker.pairs_seen == 3  # 4 and 5 were never retrieved, so never re-ranked


def test_each_user_gets_their_own_top_n_and_row_order_is_kept():
    pairs = pl.DataFrame({"user_id": [1, 2, 1, 2, 1], "video_id": [1, 1, 2, 2, 3]})

    scores = TwoStageRecommender(TableScorer(RETRIEVAL), TableScorer(RERANK), top_n=2).score(pairs)

    user_1 = pairs.with_row_index().filter(pl.col("user_id") == 1)
    ranked = order(scores.gather(user_1["index"]), user_1.drop("index"))
    assert ranked == [2, 1, 3]  # top 2 by retrieval (1, 2) re-ranked, then 3
    assert scores.len() == pairs.height


def test_top_n_must_be_positive():
    with pytest.raises(ValueError, match="top_n"):
        TwoStageRecommender(TableScorer(RETRIEVAL), TableScorer(RERANK), top_n=0)


def test_users_with_fewer_candidates_than_top_n_are_fully_reranked():
    pairs = pl.DataFrame({"user_id": [7, 7], "video_id": [1, 3]})

    scores = TwoStageRecommender(TableScorer(RETRIEVAL), TableScorer(RERANK), top_n=5).score(pairs)

    assert order(scores, pairs) == [3, 1]


@pytest.mark.parametrize("stage", ["retriever", "reranker"])
def test_non_finite_scores_are_rejected_instead_of_jumping_the_queue(stage):
    broken = {1: float("nan"), 2: 1.0, 3: 0.5}
    retriever = TableScorer(broken if stage == "retriever" else RETRIEVAL)
    reranker = TableScorer(broken if stage == "reranker" else RERANK)
    pairs = pl.DataFrame({"user_id": [7, 7, 7], "video_id": [1, 2, 3]})

    with pytest.raises(ValueError, match="finite"):
        TwoStageRecommender(retriever, reranker, top_n=2).score(pairs)


def test_retrieval_recall_reports_the_share_of_positives_retrieved_and_its_ceiling():
    labeled = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "video_id": [1, 2, 3, 1, 4, 5],
            "is_positive": [True, False, True, True, False, False],
        }
    )

    recall = retrieval_recall(labeled, TableScorer(RETRIEVAL), top_n=1)

    # user 1: 1 of 2 positives in a top 1 (ceiling 1/2); user 2: 1 of 1 (ceiling 1);
    # user 3 has no positives and is skipped.
    assert recall.recall == pytest.approx((0.5 + 1.0) / 2)
    assert recall.ceiling == pytest.approx((0.5 + 1.0) / 2)


def test_retrieval_recall_needs_at_least_one_user_with_a_positive():
    labeled = pl.DataFrame({"user_id": [1], "video_id": [1], "is_positive": [False]})

    with pytest.raises(ValueError, match="positive"):
        retrieval_recall(labeled, TableScorer(RETRIEVAL), top_n=1)

from typing import Self

import polars as pl

from recsys.evaluation.ranking import popularity_percentiles
from recsys.experiments.runner import fit_and_evaluate, load_split


class SpyModel:
    """Records which columns it was asked to score; scores everything equally."""

    name = "spy"

    def __init__(self) -> None:
        self.scored_columns: list[str] = []

    def fit(self, train: pl.DataFrame) -> Self:
        return self

    def score(self, pairs: pl.DataFrame) -> pl.Series:
        self.scored_columns = pairs.columns
        return pl.Series("score", [0.0] * pairs.height)


def test_fit_and_evaluate_never_shows_the_label_to_the_model(labeled_splits, evaluation_params):
    train, tune = load_split(labeled_splits, "train"), load_split(labeled_splits, "tune")
    model = SpyModel()

    result, fit_seconds = fit_and_evaluate(
        model, train, tune, popularity_percentiles(train, tune["video_id"]), evaluation_params
    )

    assert model.scored_columns == ["user_id", "video_id"]
    assert fit_seconds >= 0
    # Wall-clock time is not reproducible, so it stays out of the evaluation summary.
    assert "fit_seconds" not in result.summary

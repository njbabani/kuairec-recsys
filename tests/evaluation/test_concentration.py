import polars as pl
import pytest

from recsys.evaluation.concentration import gini, lorenz_curve, top_share


def test_gini_is_zero_when_every_item_is_equally_popular():
    assert gini(pl.Series([5, 5, 5, 5])) == pytest.approx(0.0)


def test_gini_is_maximal_when_one_item_takes_everything():
    assert gini(pl.Series([0, 0, 0, 10])) == pytest.approx(0.75)


def test_gini_matches_a_hand_computed_value_regardless_of_order():
    assert gini(pl.Series([4, 1, 3, 2])) == pytest.approx(0.25)


def test_lorenz_curve_runs_from_origin_to_one_through_cumulative_shares():
    curve = lorenz_curve(pl.Series([4, 1, 3, 2]))

    assert curve.columns == ["share_of_items", "share_of_interactions"]
    assert curve["share_of_items"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert curve["share_of_interactions"].to_list() == pytest.approx([0.0, 0.1, 0.3, 0.6, 1.0])


def test_top_share_is_the_fraction_held_by_the_most_popular_items():
    counts = pl.Series([1] * 9 + [11])

    assert top_share(counts, fraction=0.1) == pytest.approx(11 / 20)


def test_top_share_always_keeps_at_least_one_item():
    assert top_share(pl.Series([1, 2, 3]), fraction=0.1) == pytest.approx(3 / 6)


def test_top_share_uses_exactly_the_requested_number_of_items():
    # 0.29 * 100 is 28.999999999999996 in floating point; the top 29 items must still count.
    assert top_share(pl.Series(range(1, 101)), fraction=0.29) == pytest.approx(
        sum(range(72, 101)) / sum(range(1, 101))
    )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_top_share_rejects_fractions_outside_zero_to_one(fraction):
    with pytest.raises(ValueError, match="fraction"):
        top_share(pl.Series([1, 2, 3]), fraction=fraction)


@pytest.mark.parametrize(
    "measure", [gini, lorenz_curve, lambda counts: top_share(counts, fraction=0.1)]
)
@pytest.mark.parametrize(
    "counts",
    [
        pl.Series([], dtype=pl.Int64),
        pl.Series([0, 0]),
        pl.Series([3, -1]),
        pl.Series([1, None, 3]),
        pl.Series([1.0, float("nan"), 3.0]),
    ],
    ids=["empty", "all-zero", "negative", "null", "nan"],
)
def test_concentration_measures_reject_degenerate_counts(measure, counts):
    with pytest.raises(ValueError, match="count"):
        measure(counts)

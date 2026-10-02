import polars as pl

from recsys.demo import charts
from recsys.viz import style

READOUT = {
    "estimate": {"difference": 0.1, "ci_low": -0.2, "ci_high": 0.4},
    "cuped": {"difference": 0.15, "ci_low": -0.05, "ci_high": 0.35},
    "true_effect": 0.2,
}


def test_the_interval_chart_highlights_one_row():
    rows = pl.DataFrame(
        {"model": ["a", "b"], "value": [0.3, 0.2], "ci_low": [0.25, 0.1], "ci_high": [0.35, 0.3]}
    )

    spec = charts.interval_chart(rows, label="model", axis_title="NDCG@10", highlight="a")

    assert style.PRIMARY in str(spec.to_dict())
    assert style.INK_MUTED in str(spec.to_dict())


def test_every_chart_is_a_valid_vega_lite_spec():
    built = [
        charts.readout_chart(READOUT),
        charts.mde_chart(
            pl.DataFrame({"users": [100, 400], "mde": [0.8, 0.4]}),
            current_users=200,
            effects={"als_vs_popularity": 0.5, "a_vs_a": 0.0},  # zero cannot sit on a log axis
        ),
        charts.histogram_chart([10, 9, 11, 10, 10, 8, 12, 10, 10, 10]),
        charts.rates_chart({"naive": 0.2, "fixed": 0.05}, reference=0.05, axis_title="share"),
    ]

    for chart in built:
        spec = chart.to_dict()  # validates against the Vega-Lite schema
        assert "$schema" in spec


def test_degenerate_inputs_still_draw():
    built = [
        charts.histogram_chart([0] * 10),  # no simulated p-values at all
        charts.mde_chart(  # no spread, so nothing can be detected and nothing fits a log axis
            pl.DataFrame({"users": [100], "mde": [0.0]}), current_users=100, effects={"a_a": 0.0}
        ),
        charts.rates_chart({"naive": 0.4}, reference=None, axis_title="detected"),
        charts.readout_chart(READOUT, ci_level=0.9),
    ]

    for chart in built:
        assert "$schema" in chart.to_dict()


def test_confidence_levels_read_as_people_write_them():
    assert [charts.ci_label(level) for level in (0.95, 0.975, 0.9, 0.99)] == [
        "95%",
        "97.5%",
        "90%",
        "99%",
    ]

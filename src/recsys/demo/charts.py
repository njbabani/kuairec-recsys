"""Altair charts for the demo app, in the project's chart style (:mod:`recsys.viz.style`).

Every chart encodes one thing: a dot with its 95% interval, a curve, or bars against a dashed
reference line. Colours carry meaning only through the accent (``PRIMARY``) versus muted ink, and
every mark has a tooltip with its exact value.
"""

from collections.abc import Mapping

import altair as alt
import polars as pl

from recsys.viz import style

ROW_HEIGHT_PX = 38
AXIS_HEIGHT_PX = 40  # room for the x-axis and its title below the rows
CHART_HEIGHT_PX = 300
LABEL_LIMIT_PX = 260  # long enough for every policy and rule name
TICKS = 6
BAR_THICKNESS_PX = 22
LOG_PADDING = 1.25  # headroom around the data on a log axis
DASH = (4, 3)
DEFAULT_CI_LEVEL = 0.95


def ci_label(level: float) -> str:
    """A confidence level as people write it: 0.95 is "95%", 0.975 is "97.5%"."""
    return f"{level * 100:g}%"


def _reference(value: float, channel: str, color: str = style.INK_MUTED) -> alt.Chart:
    """A dashed line across the chart at ``value`` on the ``x`` or ``y`` channel."""
    data = pl.DataFrame({"reference": [value]})
    encoding = {channel: alt.X("reference:Q") if channel == "x" else alt.Y("reference:Q")}
    return alt.Chart(data).mark_rule(color=color, strokeDash=DASH).encode(**encoding)


def _rows_height(n_rows: int) -> int:
    return ROW_HEIGHT_PX * n_rows + AXIS_HEIGHT_PX


def _interval_layers(
    rows: pl.DataFrame,
    label: str,
    axis_title: str,
    highlight: str | None,
    value: str,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> list[alt.Chart]:
    """A dot per row with its interval (``ci_low`` to ``ci_high``), in the rows' order."""
    highlighted = pl.col(label) == highlight if highlight is not None else pl.lit(False)
    data = rows.with_columns(
        color=pl.when(highlighted).then(pl.lit(style.PRIMARY)).otherwise(pl.lit(style.INK_MUTED))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            f"{label}:N",
            sort=rows[label].to_list(),
            title=None,
            axis=alt.Axis(labelLimit=LABEL_LIMIT_PX),
        ),
        color=alt.Color("color:N", scale=None),
        tooltip=[
            alt.Tooltip(f"{label}:N"),
            alt.Tooltip(f"{value}:Q", format=".3f"),
            alt.Tooltip("ci_low:Q", title=f"{ci_label(ci_level)} CI low", format=".3f"),
            alt.Tooltip("ci_high:Q", title=f"{ci_label(ci_level)} CI high", format=".3f"),
        ],
    )
    intervals = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("ci_low:Q", title=axis_title, scale=alt.Scale(zero=False)), x2="ci_high:Q"
    )
    dots = base.mark_circle(size=110, opacity=1, stroke=style.SURFACE, strokeWidth=2).encode(
        x=f"{value}:Q"
    )
    return [intervals, dots]


def interval_chart(
    rows: pl.DataFrame,
    label: str,
    axis_title: str,
    highlight: str | None = None,
    value: str = "value",
    reference: float | None = None,
) -> alt.LayerChart:
    """A dot per row with its 95% interval; ``highlight`` gets the accent and ``reference``
    (e.g. zero for differences) a dashed line."""
    layers = _interval_layers(rows, label, axis_title, highlight, value)
    if reference is not None:
        layers = [_reference(reference, "x"), *layers]
    return alt.layer(*layers).properties(height=_rows_height(rows.height))


def readout_chart(result: Mapping, ci_level: float = DEFAULT_CI_LEVEL) -> alt.LayerChart:
    """An experiment's plain and CUPED estimates with their intervals, and the true effect."""
    rows = pl.DataFrame(
        [
            {
                "analysis": name,
                **{key: result[source][key] for key in ("difference", "ci_low", "ci_high")},
            }
            for name, source in (("plain", "estimate"), ("CUPED", "cuped"))
        ]
    )
    truth = (
        alt.Chart(pl.DataFrame({"true effect": [result["true_effect"]]}))
        .mark_rule(color=style.INK, strokeWidth=2)
        .encode(x="true effect:Q", tooltip=[alt.Tooltip("true effect:Q", format="+.3f")])
    )
    layers = _interval_layers(
        rows,
        label="analysis",
        axis_title=(
            "difference in videos liked per session "
            f"(treatment minus control, {ci_label(ci_level)} CI)"
        ),
        highlight="CUPED",
        value="difference",
        ci_level=ci_level,
    )
    return alt.layer(_reference(0.0, "x", style.BASELINE), *layers, truth).properties(
        height=_rows_height(rows.height)
    )


def mde_chart(
    curve: pl.DataFrame, current_users: int, effects: Mapping[str, float]
) -> alt.LayerChart:
    """The smallest detectable effect as the experiment grows, against real effect sizes.

    Both axes are logarithmic: the curve is then a straight line (the MDE falls with the square
    root of the users), and small effects are not squeezed together near zero."""
    # A log axis cannot show zero: drop zero effects (an A/A) and a zero-variance curve.
    effects = {name: abs(effect) for name, effect in effects.items() if effect}
    curve = curve.filter(pl.col("mde") > 0)
    shown = [*curve["mde"].to_list(), *effects.values()]
    domain = [min(shown) / LOG_PADDING, max(shown) * LOG_PADDING] if shown else alt.Undefined
    line = (
        alt.Chart(curve)
        .mark_line(color=style.PRIMARY, strokeWidth=style.LINE_WIDTH)
        .encode(
            x=alt.X("users:Q", scale=alt.Scale(type="log"), title="users in the experiment (log)"),
            y=alt.Y(
                "mde:Q",
                scale=alt.Scale(type="log", domain=domain, nice=False),
                title="smallest detectable effect (log)",
            ),
            tooltip=[alt.Tooltip("users:Q", format=","), alt.Tooltip("mde:Q", format=".3f")],
        )
    )
    sizes = pl.DataFrame({"experiment": list(effects), "effect": list(effects.values())})
    effect_lines = (
        alt.Chart(sizes)
        .mark_rule(color=style.INK_MUTED, strokeDash=DASH)
        .encode(y="effect:Q", tooltip=["experiment:N", alt.Tooltip("effect:Q", format=".3f")])
    )
    effect_labels = (
        alt.Chart(sizes)
        .mark_text(align="right", baseline="bottom", dy=-3, color=style.INK_SECONDARY)
        .encode(y="effect:Q", x=alt.value("width"), text="experiment:N")
    )
    now = (
        alt.Chart(pl.DataFrame({"users": [current_users]}))
        .mark_rule(color=style.INK, strokeWidth=1.5)
        .encode(x="users:Q", tooltip=[alt.Tooltip("users:Q", title="this experiment", format=",")])
    )
    return alt.layer(line, effect_lines, effect_labels, now).properties(height=CHART_HEIGHT_PX)


def histogram_chart(counts: list[int]) -> alt.LayerChart:
    """Simulated p-values in equal bins, against the even spread expected with no effect."""
    total, n_bins = max(sum(counts), 1), len(counts)  # all-empty bins draw as zeros
    bins = pl.DataFrame(
        {
            "from": [index / n_bins for index in range(n_bins)],
            "to": [(index + 1) / n_bins for index in range(n_bins)],
            "share": [count / total for count in counts],
        }
    )
    bars = (
        alt.Chart(bins)
        .mark_bar(color=style.PRIMARY, binSpacing=2)
        .encode(
            x=alt.X("from:Q", bin="binned", title="p-value"),
            x2="to:Q",
            y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="share of experiments"),
            tooltip=[alt.Tooltip("from:Q", title="p from"), alt.Tooltip("share:Q", format=".1%")],
        )
    )
    return alt.layer(bars, _reference(1 / n_bins, "y")).properties(height=CHART_HEIGHT_PX)


def rates_chart(
    rates: Mapping[str, float], reference: float | None, axis_title: str
) -> alt.LayerChart:
    """One bar per rule, with a dashed line at ``reference`` (e.g. the promised alpha)."""
    data = pl.DataFrame({"rule": list(rates), "rate": list(rates.values())})
    bars = (
        alt.Chart(data)
        .mark_bar(color=style.PRIMARY, cornerRadiusEnd=4, size=BAR_THICKNESS_PX)
        .encode(
            y=alt.Y(
                "rule:N", sort=list(rates), title=None, axis=alt.Axis(labelLimit=LABEL_LIMIT_PX)
            ),
            x=alt.X("rate:Q", axis=alt.Axis(format="%", tickCount=TICKS), title=axis_title),
            tooltip=["rule:N", alt.Tooltip("rate:Q", format=".1%")],
        )
    )
    labels = bars.mark_text(align="left", dx=4, color=style.INK_SECONDARY).encode(
        text=alt.Text("rate:Q", format=".1%")
    )
    layers = [bars, labels] if reference is None else [bars, labels, _reference(reference, "x")]
    return alt.layer(*layers).properties(height=_rows_height(data.height))

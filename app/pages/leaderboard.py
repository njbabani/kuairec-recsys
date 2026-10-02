"""Leaderboard: every model scored once on the untouched test users."""

import polars as pl
import streamlit as st

from recsys.demo import charts, ui
from recsys.demo.data import leaderboard, paired_comparisons

METRICS = {
    "ndcg_at_10": "NDCG@10",
    "precision_at_10": "Precision@10",
    "recall_at_50": "Recall@50",
    "map_at_10": "MAP@10",
}
SUMMARY_COLUMNS = {
    "ndcg_at_10": "NDCG@10",
    "recall_at_50": "Recall@50",
    "coverage_at_10": "catalogue coverage@10",
    "popularity_pct_at_10": "popularity percentile@10",
}

st.title("Leaderboard")
paths = ui.demo_paths()
if not ui.require(ui.status(paths)["test_report"]):
    st.stop()
test = ui.test_report(paths.test_report)
st.markdown(
    f"Every model was frozen before the **{test['users']:,} test users** were opened; each then "
    "ranked all of its candidate videos for every user, once. Dots are averages over users and "
    "lines their 95% bootstrap intervals."
)

metric = st.selectbox("Metric", list(METRICS), format_func=METRICS.__getitem__, key="metric")
board = leaderboard(test, metric)
st.altair_chart(
    charts.interval_chart(
        board, label="model", axis_title=f"{METRICS[metric]} (95% CI)", highlight=board["model"][0]
    ),
    width="stretch",
)

st.subheader("Paired comparisons")
st.markdown(
    "Users differ a lot, so the intervals of separate models overlap. Comparing two models **on "
    "the same users** cancels those differences out, which is what makes small gains visible. "
    "*Better for* is the share of users a model ranks better for (ties count half)."
)
reference = st.radio("Compared with", list(test["paired"]), horizontal=True, key="reference")
paired = paired_comparisons(test, reference)
compared = test["paired"][reference]["metric"]
st.altair_chart(
    charts.interval_chart(
        paired,
        label="model",
        axis_title=f"{METRICS.get(compared, compared)} minus {reference} (95% CI)",
        value="difference",
        reference=0.0,
    ),
    width="stretch",
)
st.dataframe(
    paired.with_columns(better_for=pl.col("win_rate") * 100).drop("win_rate"),
    hide_index=True,
    column_config={
        "difference": st.column_config.NumberColumn(format="%+.3f"),
        "ci_low": st.column_config.NumberColumn("95% CI low", format="%+.3f"),
        "ci_high": st.column_config.NumberColumn("95% CI high", format="%+.3f"),
        "better_for": st.column_config.NumberColumn("better for", format="%.0f%%"),
    },
)

st.subheader("All models")
st.caption(
    "Coverage is the share of the candidate videos that appear in anyone's top 10; the "
    "popularity percentile shows how much a model leans on popular videos (random: 50th)."
)
st.dataframe(
    pl.DataFrame(
        [
            {"model": name, **{label: summary[key] for key, label in SUMMARY_COLUMNS.items()}}
            for name, summary in test["models"].items()
        ]
    ),
    hide_index=True,
    column_config={
        label: st.column_config.NumberColumn(format="%.3f") for label in SUMMARY_COLUMNS.values()
    },
)

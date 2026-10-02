"""Overview: the project in brief, its headline numbers, and what the demo can show."""

import polars as pl
import streamlit as st

from recsys.demo import ui
from recsys.demo.data import leaderboard

st.title("KuaiRec recommender")
st.markdown(
    "A short-video recommender built on [KuaiRec](https://kuairec.com/), a dataset in which the "
    "test users reacted to almost every video. So for any recommendation we know what the user "
    "*actually did*, and every A/B test in this demo can be checked against the **true** effect."
)

paths = ui.demo_paths()
status = ui.status(paths)
best_card, gain_card, mde_card, aa_card = st.columns(4)
if status["test_report"].available:
    test = ui.test_report(paths.test_report)
    reference = next(iter(test["paired"]))
    metric = test["paired"][reference]["metric"]
    best = leaderboard(test, metric).row(0, named=True)
    gain = test["paired"]["two_tower"]["differences"]["two_stage"]
    best_card.metric(
        f"Best model: {best['model']}",
        f"{best['value']:.3f}",
        help=f"{metric} on the {test['users']:,} untouched test users",
        border=True,
    )
    gain_card.metric(
        "Re-ranking gain",
        f"{gain['mean']:+.3f}",
        help=(
            f"Two-stage minus two-tower {metric} on the same users, 95% CI "
            f"[{gain['ci_low']:+.3f}, {gain['ci_high']:+.3f}]"
        ),
        border=True,
    )
if status["ab_report"].available:
    ab = ui.ab_report(paths.ab_report)
    power, design, aa = ab["power"], ab["design"], ab["aa_test"]
    users = power["users_control"] + power["users_treatment"]
    mde_card.metric(
        "Detectable A/B lift",
        f"{power['mde']:.2f} likes",
        help=(
            f"Extra liked videos per {ab['session_length']}-video session that an experiment on "
            f"{users:,} users detects {design['power']:.0%} of the time "
            f"({power['mde_cuped']:.2f} with CUPED)"
        ),
        border=True,
    )
    aa_card.metric(
        "A/A false alarms",
        f"{aa['rejection_rate']:.1%}",
        help=(
            f"Share of {aa['simulations']:,} experiments comparing a model with itself that "
            f"came out 'significant'; the test promises {design['alpha']:.0%}"
        ),
        border=True,
    )

st.subheader("What you can do here")
st.markdown(
    "- **Leaderboard**: every model scored once on the untouched test users, with 95% intervals "
    "and paired comparisons.\n"
    "- **User explorer**: pick a test user and see what each model recommended and what the user "
    "did with it.\n"
    "- **A/B lab**: run an experiment yourself, plan its size, repeat it thousands of times, "
    "peek early or break the logging, and see how each method does against the truth."
)

st.subheader("Data")
st.dataframe(
    pl.DataFrame(
        [
            {"data": artifact.label, "ready": artifact.available, "build with": artifact.command}
            for artifact in status.values()
        ]
    ),
    hide_index=True,
    column_config={"ready": st.column_config.CheckboxColumn("ready")},
)
st.caption("The reports are in git; the data is rebuilt by the DVC pipeline (see the README).")

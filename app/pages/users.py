"""User explorer: what each model recommended to one test user, and what they did with it."""

import numpy as np
import polars as pl
import streamlit as st

from recsys.demo import ui
from recsys.demo.data import session_totals, user_session

st.title("User explorer")
st.markdown(
    "The test users reacted to almost every candidate video, so the reactions below are **what "
    "the user actually did**, not a model's guess. Pick a user to see each model's session for "
    "them, side by side."
)
paths = ui.demo_paths()
if not ui.require(ui.status(paths)["demo_tables"]):
    st.stop()
users = ui.users(paths.users)
videos = ui.session_videos(paths.session_videos)
ids = users["user_id"].to_list()

if st.button("Pick a random user", key="random_user", icon=":material/casino:"):
    st.session_state["user"] = int(np.random.default_rng().choice(ids))
user = st.selectbox("Test user", ids, key="user")
profile = users.filter(pl.col("user_id") == user).row(0, named=True)

views_card, rate_card, picky_card, arm_card = st.columns(4)
views_card.metric(
    "Training views",
    f"{profile['training_views']:,}",
    help="Videos this user watched in the logged feed before the experiment",
)
rate = profile["training_positive_rate"]
rate_card.metric(
    "Liked before the experiment",
    "n/a" if rate is None else f"{rate:.0%}",
    help="Share of the training views that count as liked: the covariate CUPED uses",
)
picky_card.metric(
    "Likes among candidates",
    f"{profile['liked_share']:.0%}",
    help="Share of all candidate videos this user liked: how picky they are",
)
arm_card.metric(
    "A/B arm",
    "treatment" if profile["in_treatment"] else "control",
    help="Assigned by a stable hash of the user id",
)
favourites = profile["favourite_categories"]
st.caption(
    "Favourite categories before the experiment: "
    + (", ".join(favourites) if favourites else "none recorded")
)

st.subheader("What each model showed this user")
length = int(videos["position"].max())
totals = session_totals(videos, user)
st.dataframe(
    totals,
    hide_index=True,
    column_config={
        "liked": st.column_config.NumberColumn(f"liked (of {length})"),
        "watched_s": st.column_config.NumberColumn("watched", format="%.0f s"),
    },
)
policies = totals["policy"].to_list()  # in the reports' order, unknown policies last
for tab, policy in zip(st.tabs(policies), policies, strict=True):
    with tab:
        st.dataframe(
            user_session(videos, user, policy),
            hide_index=True,
            column_config={
                "duration_s": st.column_config.NumberColumn("length", format="%.0f s"),
                "watched_s": st.column_config.NumberColumn("watched", format="%.1f s"),
                "liked": st.column_config.CheckboxColumn("liked"),
            },
        )

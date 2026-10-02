"""The KuaiRec recommender demo. Run it with ``make demo`` from the repository root."""

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="KuaiRec recommender", page_icon=":material/movie:", layout="wide")
st.navigation(
    [
        st.Page(PAGES / "overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page(PAGES / "leaderboard.py", title="Leaderboard", icon=":material/leaderboard:"),
        st.Page(PAGES / "users.py", title="User explorer", icon=":material/person_search:"),
        st.Page(PAGES / "ab_lab.py", title="A/B lab", icon=":material/science:"),
    ]
).run()

"""The Streamlit pages, run headlessly with Streamlit's own test harness (``AppTest``)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from recsys.demo.data import ROOT_ENV

APP_DIR = Path(__file__).resolve().parents[2] / "app"
PAGES = ("overview.py", "leaderboard.py", "users.py", "ab_lab.py")
PAGE_TIMEOUT_S = 120  # the A/B lab runs thousands of simulated experiments on first load


@pytest.fixture
def open_page(monkeypatch):
    def _open(root: Path, script: Path) -> AppTest:
        monkeypatch.setenv(ROOT_ENV, str(root))
        return AppTest.from_file(str(script), default_timeout=PAGE_TIMEOUT_S).run()

    return _open


def texts(app: AppTest) -> str:
    """Every piece of text a visitor sees on the page, as one string."""
    elements = [*app.markdown, *app.caption, *app.info, *app.warning, *app.title, *app.subheader]
    return "\n".join(str(element.value) for element in elements)


@pytest.mark.integration
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_the_pipelines_outputs(open_page, demo_root, page):
    app = open_page(demo_root, APP_DIR / "pages" / page)

    assert not app.exception
    assert not app.error
    assert not app.info  # no "build this first" notice: the page ran to the end


@pytest.mark.integration
@pytest.mark.parametrize(
    ("page", "command"),
    [
        ("overview.py", None),
        ("leaderboard.py", None),
        ("users.py", "make demo-data"),
        ("ab_lab.py", "make experiments"),
    ],
)
def test_every_page_renders_on_a_fresh_clone_and_says_what_to_run(
    open_page, reports_only_root, page, command
):
    app = open_page(reports_only_root, APP_DIR / "pages" / page)

    assert not app.exception
    if command is None:
        assert not app.info  # the reports in git are all these pages need
    else:
        assert command in texts(app)


@pytest.mark.integration
def test_the_entry_point_opens_on_the_overview(open_page, demo_root):
    app = open_page(demo_root, APP_DIR / "streamlit_app.py")

    assert not app.exception
    assert "KuaiRec" in texts(app)


@pytest.mark.integration
def test_the_ab_lab_reads_out_the_experiment_the_visitor_picks(open_page, demo_root):
    app = open_page(demo_root, APP_DIR / "pages" / "ab_lab.py")

    app.selectbox(key="lab_control").select("two_tower")
    app.selectbox(key="lab_treatment").select("two_tower")
    app.run()

    assert not app.exception
    assert "A/A" in texts(app)  # the same policy in both arms is flagged as an A/A test
    decision = next(metric for metric in app.metric if metric.label == "Decision")
    assert decision.value == "inconclusive"


@pytest.mark.integration
@pytest.mark.parametrize("share", [0.1, 0.3, 0.5, 0.9])
def test_the_ab_lab_survives_any_split_the_slider_allows(open_page, demo_root, share):
    app = open_page(demo_root, APP_DIR / "pages" / "ab_lab.py")

    app.slider(key="lab_share").set_value(share).run()
    looks = [slider for slider in app.slider if slider.key == "lab_looks"]
    if looks:
        looks[0].set_value(looks[0].max).run()

    assert not app.exception


@pytest.mark.integration
def test_the_ab_lab_accepts_any_alpha_and_power_the_params_allow(open_page, demo_root, tmp_path):
    root = tmp_path / "project"
    shutil.copytree(demo_root, root)
    report_path = root / "reports/metrics/ab_test.json"
    report = json.loads(report_path.read_text())
    report["design"] |= {"alpha": 0.025, "power": 0.85}
    report_path.write_text(json.dumps(report))

    app = open_page(root, APP_DIR / "pages" / "ab_lab.py")

    assert not app.exception
    assert app.select_slider(key="lab_alpha").value == 0.025
    assert "97.5%" in texts(app)  # intervals at 1 - alpha, not a fixed 95%


@pytest.mark.integration
def test_the_user_explorer_shows_the_chosen_users_sessions(open_page, demo_root, demo_world):
    app = open_page(demo_root, APP_DIR / "pages" / "users.py")

    app.selectbox(key="user").select(7)
    app.run()

    assert not app.exception
    assert app.selectbox(key="user").value == 7
    assert len(app.dataframe) == 1 + len(demo_world.policy_videos)  # totals, then one per policy


def test_the_app_never_loads_pytorch_or_lightgbm():
    # Their OpenMP runtimes crash when they share a process on macOS (see tests/conftest.py).
    probe = (
        "import sys; import recsys.demo.ui, recsys.demo.bundle; "
        "print(sorted(m for m in ('torch', 'lightgbm') if m in sys.modules))"
    )

    result = subprocess.run(  # noqa: S603 (fixed command, no user input)
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"

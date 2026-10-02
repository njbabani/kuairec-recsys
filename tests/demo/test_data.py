import polars as pl
import pytest

from recsys.demo.data import (
    ROOT_ENV,
    DemoPaths,
    artifacts,
    find_root,
    leaderboard,
    load_ab_report,
    load_session_videos,
    load_test_report,
    load_user_table,
    load_users,
    paired_comparisons,
    session_totals,
    user_session,
)

POLICIES = {"random", "popularity_engagement", "als", "two_tower", "two_stage"}


def test_paths_follow_the_params_of_the_given_root(demo_root):
    paths = DemoPaths.from_root(demo_root)

    assert paths.test_report == demo_root / "reports/metrics/test_evaluation.json"
    assert paths.user_outcomes == demo_root / "data/ab/user_outcomes.parquet"
    assert paths.session_videos == demo_root / "data/demo/sessions.parquet"


def test_every_artifact_is_available_once_the_pipeline_has_run(demo_root):
    status = artifacts(DemoPaths.from_root(demo_root))

    assert all(artifact.available for artifact in status.values())


def test_a_fresh_clone_has_the_reports_and_says_how_to_build_the_rest(reports_only_root):
    status = artifacts(DemoPaths.from_root(reports_only_root))

    assert status["test_report"].available
    assert status["ab_report"].available
    assert not status["ab_outcomes"].available
    assert status["ab_outcomes"].command == "make experiments"
    assert not status["demo_tables"].available
    assert status["demo_tables"].command == "make demo-data"


def test_the_root_is_the_nearest_folder_with_params_unless_the_environment_says(
    demo_root, tmp_path, monkeypatch
):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    assert find_root(demo_root / "data" / "ab") == demo_root

    monkeypatch.setenv(ROOT_ENV, str(demo_root))
    assert find_root(tmp_path) == demo_root

    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"params\.yaml"):
        find_root(demo_root)


def test_outside_any_project_the_root_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV, raising=False)

    with pytest.raises(FileNotFoundError, match=ROOT_ENV):
        find_root(tmp_path)


def test_loaders_read_what_the_pipeline_wrote(demo_root, demo_world):
    paths = DemoPaths.from_root(demo_root)

    assert set(load_test_report(paths.test_report)["models"]) == POLICIES
    assert load_ab_report(paths.ab_report)["design"]["alpha"] == 0.05
    assert load_user_table(paths.user_outcomes).users.len() == demo_world.n_users
    assert load_users(paths.users).height == demo_world.n_users
    assert set(load_session_videos(paths.session_videos)["policy"].unique()) == POLICIES


def test_loaders_reject_files_the_app_cannot_show(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"models": {}, "users": 3}')
    outcomes = tmp_path / "outcomes.parquet"
    pl.DataFrame({"user_id": [1], "policy": ["als"]}).write_parquet(outcomes)

    with pytest.raises(ValueError, match="paired"):
        load_test_report(report)
    with pytest.raises(ValueError, match="liked"):
        load_user_table(outcomes)


def test_the_leaderboard_ranks_models_with_their_intervals(demo_root):
    report = load_test_report(DemoPaths.from_root(demo_root).test_report)

    board = leaderboard(report, "ndcg_at_10")

    assert board.columns == ["model", "value", "ci_low", "ci_high"]
    assert board["model"][0] == "two_stage"  # the best model first
    assert board["value"].is_sorted(descending=True)
    assert (board["ci_low"] < board["value"]).all()  # real intervals, not a zero-width stand-in
    assert (board["value"] < board["ci_high"]).all()


def test_paired_comparisons_list_every_other_model_against_the_reference(demo_root):
    report = load_test_report(DemoPaths.from_root(demo_root).test_report)

    paired = paired_comparisons(report, "two_tower")

    assert set(paired["model"]) == POLICIES - {"two_tower"}
    two_stage = paired.filter(pl.col("model") == "two_stage").row(0, named=True)
    expected = report["paired"]["two_tower"]["differences"]["two_stage"]
    assert two_stage["difference"] == expected["mean"]
    assert two_stage["win_rate"] == expected["win_rate"]


def test_a_users_sessions_and_their_totals(demo_root, demo_world):
    videos = load_session_videos(DemoPaths.from_root(demo_root).session_videos)

    session = user_session(videos, user_id=3, policy="als")
    totals = session_totals(videos, user_id=3)

    assert session["video_id"].to_list() == demo_world.policy_videos["als"]
    assert session.columns == [
        "position",
        "video_id",
        "category",
        "duration_s",
        "watched_s",
        "liked",
    ]
    als = totals.filter(pl.col("policy") == "als").row(0, named=True)
    assert als["liked"] == session["liked"].sum()
    assert als["watched_s"] == pytest.approx(session["watched_s"].sum())
    assert set(totals["policy"]) == POLICIES


def test_metrics_without_an_interval_get_a_zero_width_one(demo_root):
    report = load_test_report(DemoPaths.from_root(demo_root).test_report)

    board = leaderboard(report, "coverage_at_10")

    assert board["ci_low"].to_list() == board["value"].to_list()


def test_totals_list_unknown_policies_after_the_known_ones():
    videos = pl.DataFrame(
        {
            "policy": ["experimental", "als", "random"],
            "user_id": [1, 1, 1],
            "liked": [True, False, True],
            "watched_s": [1.0, 2.0, 3.0],
        }
    )

    totals = session_totals(videos, user_id=1)

    assert totals["policy"].to_list() == ["random", "als", "experimental"]


def test_without_a_start_folder_the_root_falls_back_to_the_apps_own_repository(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # e.g. `streamlit run` from some other folder

    assert (find_root() / "app" / "streamlit_app.py").is_file()

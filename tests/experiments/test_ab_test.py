import json
import math

import numpy as np
import polars as pl
import pytest

from recsys.config import ABExperimentParams, ABTestParams
from recsys.experiments.ab_test import ABTestPaths, run_ab_test
from recsys.tracking import InMemoryTracker

N_USERS, N_VIDEOS, LENGTH = 300, 20, 3
# Each policy always shows the same three videos; later videos are liked more often.
POLICY_VIDEOS = {
    "random": [0, 7, 14],
    "popularity_engagement": [3, 4, 5],
    "als": [9, 10, 11],
    "two_tower": [15, 16, 17],
    "two_stage": [15, 16, 18],
}
PARAMS = ABTestParams(
    session_length=LENGTH,
    treatment_share=0.5,
    salt="test-ab",
    alpha=0.05,
    power=0.8,
    simulations=200,
    looks=5,
    srm_drop_shares=(0.05, 0.5),  # 300 users: only the large loss is reliably detectable
    seed=0,
    aa_policy="two_tower",
    experiments=(
        ABExperimentParams(name="two_tower_vs_als", control="als", treatment="two_tower"),
        ABExperimentParams(
            name="two_stage_vs_two_tower", control="two_tower", treatment="two_stage"
        ),
    ),
    primary_experiment="two_tower_vs_als",
)


@pytest.fixture
def paths(tmp_path) -> ABTestPaths:
    rng = np.random.default_rng(0)
    activity = rng.uniform(0.1, 0.6, size=N_USERS)  # each user's general tendency to like
    users = np.repeat(np.arange(N_USERS), N_VIDEOS)
    videos = np.tile(np.arange(N_VIDEOS), N_USERS)
    like_probability = np.clip(activity[users] + videos / 40, 0, 1)
    splits = tmp_path / "splits"
    splits.mkdir()
    pl.DataFrame(
        {
            "user_id": users,
            "video_id": videos,
            "is_positive": rng.random(users.size) < like_probability,
            "play_duration": rng.integers(1000, 20000, size=users.size),
        }
    ).write_parquet(splits / "test.parquet")
    pl.DataFrame(  # training weeks: the same users' earlier, independent views
        {
            "user_id": users,
            "video_id": videos + 100,
            "is_positive": rng.random(users.size) < activity[users],
        }
    ).write_parquet(splits / "train.parquet")
    pl.DataFrame(
        [
            (policy, user, video, position)
            for policy, chosen in POLICY_VIDEOS.items()
            for user in range(N_USERS)
            for position, video in enumerate(chosen, start=1)
        ],
        schema=["policy", "user_id", "video_id", "position"],
        orient="row",
    ).write_parquet(tmp_path / "sessions.parquet")
    return ABTestPaths(
        sessions_path=tmp_path / "sessions.parquet",
        splits_dir=splits,
        metrics_path=tmp_path / "ab_test.json",
        outcomes_path=tmp_path / "ab" / "user_outcomes.parquet",
    )


@pytest.mark.integration
def test_ab_test_reads_out_every_planned_experiment_against_the_truth(paths):
    report = run_ab_test(paths, PARAMS, InMemoryTracker())

    better = report["experiments"]["two_tower_vs_als"]
    truth = better["readout"]["true_effect"]
    # Videos 15-17 are each liked 6/40 more often than videos 9-11: +0.45 per session, up to the
    # noise of the users' recorded reactions (per user sd <= sqrt(6 * 0.25), 300 users).
    assert truth == pytest.approx(0.45, abs=4 * math.sqrt(6 * 0.25 / N_USERS))
    assert truth > report["power"]["mde"]  # so the test should detect it most of the time
    assert better["operating_characteristics"]["rejection_rate"] > PARAMS.power
    assert better["readout"]["decision"] == "ship"
    needed = better["users_needed"]
    low, high = needed["range"]
    assert low <= needed["at_true_effect"] <= high
    # Activity drives both the covariate and the outcome: correlation ~0.4, standard error ~0.05.
    assert report["power"]["covariate_correlation"] > 0.2
    assert report["design"]["alpha"] == PARAMS.alpha
    assert json.loads(paths.metrics_path.read_text()) == report


@pytest.mark.integration
def test_ab_test_validates_its_own_statistics(paths, rate_tolerance):
    report = run_ab_test(paths, PARAMS, InMemoryTracker())

    aa, tolerance = report["aa_test"], rate_tolerance(PARAMS.alpha, PARAMS.simulations)
    assert aa["true_effect"] == 0.0
    assert aa["rejection_rate"] == pytest.approx(PARAMS.alpha, abs=tolerance)
    assert sum(aa["p_value_histogram"]) == PARAMS.simulations
    peeking = report["peeking"]["aa"]
    # A naive run also looks at the end, so it never rejects less often than one final look;
    # rejecting strictly more often shows the earlier looks added false alarms.
    assert peeking["naive_rejection_rate"] > peeking["fixed_horizon_rejection_rate"]
    other_salts = report["assignment"]["other_salts"]
    assert other_salts["assignments"] == PARAMS.simulations
    assert other_salts["covariate_imbalance_rate"] == pytest.approx(PARAMS.alpha, abs=tolerance)


@pytest.mark.integration
def test_the_logging_bug_demo_fakes_a_lift_and_trips_the_sample_ratio_check(paths):
    srm = run_ab_test(paths, PARAMS, InMemoryTracker())["srm_demo"]

    small, large = srm["drops"]
    assert srm["true_effect"] == 0.0  # an A/A comparison: any lift is fake
    assert [small["drop_share"], large["drop_share"]] == list(PARAMS.srm_drop_shares)
    assert large["estimate"]["difference"] > srm["without_bug"]["estimate"]["difference"]
    assert large["decision"] == "invalid: sample ratio mismatch"


@pytest.mark.integration
def test_ab_test_writes_every_users_outcome_under_every_policy(paths):
    run_ab_test(paths, PARAMS, InMemoryTracker())

    outcomes = pl.read_parquet(paths.outcomes_path)

    assert outcomes.height == N_USERS * len(POLICY_VIDEOS)
    assert {"user_id", "policy", "liked", "watch_time_s", "covariate", "in_treatment"} <= set(
        outcomes.columns
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (pl.col("user_id") != 0, "als: 299 of 300 test users"),
        (pl.col("position") != 3, "als: sessions of 300 users are not 3 videos long"),
    ],
    ids=["missing-user", "short-session"],
)
def test_ab_test_refuses_sessions_that_do_not_cover_every_user_in_full(paths, broken, message):
    sessions = pl.read_parquet(paths.sessions_path)
    sessions.filter((pl.col("policy") != "als") | broken).write_parquet(paths.sessions_path)

    with pytest.raises(ValueError, match=message):
        run_ab_test(paths, PARAMS, InMemoryTracker())

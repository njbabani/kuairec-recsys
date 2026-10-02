import json
import math

import polars as pl
import pytest

from recsys.abtest.simulation import operating_characteristics, peeking, srm_demo
from recsys.abtest.stats import minimum_detectable_effect, required_sample_size
from recsys.demo.data import DemoPaths, load_ab_report, load_user_table
from recsys.demo.lab import (
    logging_bug,
    max_looks,
    mde_curve,
    peek,
    plan,
    rerandomise,
    run_experiment,
)
from recsys.experiments.ab_test import UserTable, assign

ALPHA, POWER, SHARE = 0.05, 0.8, 0.5


@pytest.fixture(scope="module")
def table(demo_root):
    return load_user_table(DemoPaths.from_root(demo_root).user_outcomes)


@pytest.mark.integration
def test_the_lab_reproduces_the_pipelines_readout_with_the_same_settings(demo_root, table):
    report = load_ab_report(DemoPaths.from_root(demo_root).ab_report)
    design = report["design"]

    result = run_experiment(
        table,
        "als",
        "two_tower",
        report["assignment"]["salt"],
        design["treatment_share"],
        design["alpha"],
    )

    expected = report["experiments"]["two_tower_vs_als"]["readout"]
    assert json.loads(json.dumps(result)) == expected


def test_a_new_salt_gives_a_new_split_but_the_same_truth(table):
    first = run_experiment(table, "als", "two_tower", "salt-a", SHARE, ALPHA)
    second = run_experiment(table, "als", "two_tower", "salt-b", SHARE, ALPHA)

    assert first["true_effect"] == second["true_effect"]
    assert first["estimate"] != second["estimate"]


def test_planning_uses_the_pipelines_power_formulas(table):
    result = plan(
        table, "als", "two_tower", n_users=400, treatment_share=SHARE, alpha=ALPHA, power=POWER
    )

    sd_control = float(table.values("als", "liked").std(ddof=1))
    sd_treatment = float(table.values("two_tower", "liked").std(ddof=1))
    expected_mde = minimum_detectable_effect(sd_control, 200, 200, ALPHA, POWER)
    assert result["mde"] == pytest.approx(expected_mde)
    assert result["mde_cuped"] < result["mde"]
    assert result["users_needed"]["at_true_effect"] == required_sample_size(
        sd_control, sd_treatment, result["true_effect"], SHARE, ALPHA, POWER
    )


def test_the_mde_halves_when_the_users_quadruple():
    curve = mde_curve(
        sd=2.0, users=[100, 400, 1600], treatment_share=SHARE, alpha=ALPHA, power=POWER
    )

    small, medium, large = curve["mde"].to_list()
    assert curve["users"].to_list() == [100, 400, 1600]
    assert small == pytest.approx(2 * medium)
    assert medium == pytest.approx(2 * large)


def test_rerandomising_the_same_policy_is_an_aa_test(table):
    result = rerandomise(
        table, "two_tower", "two_tower", n_sims=100, treatment_share=SHARE, alpha=ALPHA, seed=0
    )

    assert result["true_effect"] == 0.0
    assert result["simulations"] == 100


def test_peeking_runs_with_the_visitors_settings(table):
    result = peek(
        table,
        "als",
        "two_tower",
        looks=4,
        n_sims=100,
        treatment_share=SHARE,
        alpha=ALPHA,
        tau=0.3,
        seed=0,
    )

    assert result["looks"] == 4
    assert len(result["users_per_look"]) == 4


def test_a_large_logging_bug_is_caught_by_the_sample_ratio_check(table):
    result = logging_bug(
        table, "two_tower", salt="demo-test", treatment_share=SHARE, drop_share=0.8, alpha=ALPHA
    )  # 120 users: only a large loss is beyond chance at the strict alarm level

    assert result["true_effect"] == 0.0
    assert result["with_bug"]["decision"] == "invalid: sample ratio mismatch"
    assert (
        result["with_bug"]["estimate"]["difference"]
        > (result["without_bug"]["estimate"]["difference"])
    )


def test_every_setting_reaches_the_pipelines_functions(table):
    # Uneven split and alpha != 0.05, so swapped float arguments would change the result.
    share, alpha, seed = 0.3, 0.1, 3
    outcome = table.potential("als", "two_tower", "liked")

    assert rerandomise(table, "als", "two_tower", 100, share, alpha, seed) == (
        operating_characteristics(outcome, 100, share, alpha, seed)
    )
    assert peek(table, "als", "two_tower", 3, 100, share, alpha, 0.3, seed) == (
        peeking(outcome, 3, 100, share, alpha, 0.3, seed)
    )
    bug = logging_bug(table, "als", "salt", share, 0.2, alpha)
    expected = srm_demo(
        table.potential("als", "als", "liked"),
        assign(table.users, "salt", share),
        (0.2,),
        alpha,
        share,
    )
    assert bug["with_bug"] == expected["drops"][0]
    assert bug["without_bug"] == expected["without_bug"]


@pytest.mark.parametrize(
    ("n_users", "share", "expected"),
    [(1112, 0.5, 20), (1112, 0.1, 5), (1112, 0.9, 5), (120, 0.1, 0)],
    ids=["even-split-capped", "small-treatment", "small-control", "too-few-users"],
)
def test_peeking_looks_are_limited_so_every_first_look_has_enough_users_per_arm(
    n_users, share, expected
):
    assert max_looks(n_users, share) == expected


def test_planning_survives_an_outcome_that_never_varies():
    users = [1, 2, 3, 4]
    table = UserTable(
        pl.DataFrame(
            {
                "user_id": users * 2,
                "policy": ["a"] * 4 + ["b"] * 4,
                "liked": [2] * 4 + [3, 2, 4, 3],
                "watch_time_s": [10.0] * 8,
                "covariate": [0.1, 0.2, 0.3, 0.4] * 2,
                "in_treatment": [False, True] * 4,
            }
        )
    )

    result = plan(table, "a", "b", n_users=4, treatment_share=0.5, alpha=0.05, power=0.8)

    assert result["covariate_correlation"] == 0.0  # nothing to explain in a constant outcome
    assert not math.isnan(result["mde_cuped"])

import math

import numpy as np
import pytest

from recsys.abtest.simulation import (
    SRM_ALPHA,
    PotentialOutcomes,
    assignment_check,
    characteristics,
    decide,
    operating_characteristics,
    peeking,
    random_assignments,
    readout,
    srm_demo,
)
from recsys.abtest.stats import Estimate

N_USERS, ALPHA = 600, 0.05
_rng = np.random.default_rng(0)  # builds the fixed synthetic world below; no test draws from it
ACTIVITY = _rng.normal(size=N_USERS)  # pre-experiment covariate
BASE = 3 + 2 * ACTIVITY + _rng.normal(size=N_USERS)
NOISE = _rng.normal(scale=0.5, size=N_USERS)
ALTERNATING = np.arange(N_USERS) % 2 == 1  # an exact 50/50 split


def outcomes(effect: float) -> PotentialOutcomes:
    """Every user's outcome under both policies; the treatment adds ``effect`` on average."""
    return PotentialOutcomes(control=BASE, treatment=BASE + effect + NOISE, covariate=ACTIVITY)


SAME = PotentialOutcomes(control=BASE, treatment=BASE, covariate=ACTIVITY)  # an A/A comparison


def estimate(difference: float, p_value: float, ci_high: float) -> Estimate:
    return Estimate(1.0, 1.0 + difference, difference, ci_high - 1, ci_high, p_value, 0.0, 50, 50)


def test_potential_outcomes_must_describe_the_same_users():
    with pytest.raises(ValueError, match="same users"):
        PotentialOutcomes(control=BASE, treatment=BASE[:-1], covariate=ACTIVITY)


def test_readout_compares_each_arm_on_its_own_users_and_decides():
    potential = outcomes(effect=1.0)

    result = readout(potential, ALTERNATING, guardrail=None, alpha=ALPHA, treatment_share=0.5)

    expected = potential.treatment[ALTERNATING].mean() - potential.control[~ALTERNATING].mean()
    assert result["estimate"]["difference"] == pytest.approx(expected)
    assert result["true_effect"] == pytest.approx(potential.true_effect)
    assert result["cuped"]["ci_high"] - result["cuped"]["ci_low"] < (
        result["estimate"]["ci_high"] - result["estimate"]["ci_low"]
    )
    assert result["srm_p_value"] == pytest.approx(1.0)
    assert 0 < result["covariate_balance_p_value"] <= 1
    assert result["decision"] == "ship"


def test_a_significantly_worse_guardrail_blocks_shipping():
    worse = PotentialOutcomes(control=BASE, treatment=BASE - 1.0, covariate=ACTIVITY)

    result = readout(outcomes(1.0), ALTERNATING, guardrail=worse, alpha=ALPHA, treatment_share=0.5)

    assert result["decision"] == "do not ship: guardrail worse"


@pytest.mark.parametrize(
    ("primary", "guardrail", "srm_p_value", "decision"),
    [
        (estimate(1.0, 0.001, 2.0), None, SRM_ALPHA / 2, "invalid: sample ratio mismatch"),
        (estimate(1.0, 0.2, 2.0), None, 0.5, "inconclusive"),
        (estimate(-1.0, 0.001, -0.5), None, 0.5, "do not ship: worse"),
        (
            estimate(1.0, 0.001, 2.0),
            estimate(-1.0, 0.01, -0.1),
            0.5,
            "do not ship: guardrail worse",
        ),
        (estimate(1.0, 0.001, 2.0), estimate(-1.0, 0.3, 0.5), 0.5, "ship"),
    ],
    ids=["srm", "inconclusive", "worse", "guardrail", "ship"],
)
def test_the_ship_decision_checks_validity_then_evidence_then_harm(
    primary, guardrail, srm_p_value, decision
):
    assert decide(primary, guardrail, srm_p_value, ALPHA) == decision


def test_vectorised_characteristics_match_a_single_readout():
    potential = outcomes(effect=0.3)

    result = characteristics(potential, ALTERNATING[None, :], ALPHA)
    single = readout(potential, ALTERNATING, guardrail=None, alpha=ALPHA, treatment_share=0.5)

    for suffix, analysis in (("", "estimate"), ("_cuped", "cuped")):
        expected = single[analysis]
        assert result[f"mean_estimate{suffix}"] == pytest.approx(expected["difference"])
        width = expected["ci_high"] - expected["ci_low"]
        assert result[f"mean_ci_width{suffix}"] == pytest.approx(width)
        assert result[f"rejection_rate{suffix}"] == float(expected["p_value"] < ALPHA)


def test_an_aa_test_raises_false_alarms_at_the_chosen_rate(rate_tolerance):
    n_sims = 1000

    result = operating_characteristics(SAME, n_sims, treatment_share=0.5, alpha=ALPHA, seed=1)

    assert result["true_effect"] == 0.0
    for key in ("rejection_rate", "rejection_rate_cuped"):
        assert result[key] == pytest.approx(ALPHA, abs=rate_tolerance(ALPHA, n_sims))


def test_operating_characteristics_measure_power_bias_and_coverage_against_the_truth(
    rate_tolerance,
):
    n_sims = 1000

    result = operating_characteristics(outcomes(0.3), n_sims, 0.5, ALPHA, seed=2)

    # Each estimate's standard error is about a quarter of its 95% interval's width.
    standard_error = result["mean_ci_width"] / (2 * 1.96)
    assert abs(result["bias"]) < 4 * standard_error / math.sqrt(n_sims)
    assert result["ci_coverage"] >= 1 - ALPHA - rate_tolerance(1 - ALPHA, n_sims)
    assert result["rejection_rate_cuped"] > result["rejection_rate"]  # CUPED adds power
    assert result["mean_ci_width_cuped"] < result["mean_ci_width"]


def test_a_constant_covariate_leaves_cuped_equal_to_the_plain_analysis():
    flat = PotentialOutcomes(control=BASE, treatment=BASE + 0.3, covariate=np.ones(N_USERS))

    result = operating_characteristics(flat, n_sims=200, treatment_share=0.5, alpha=ALPHA, seed=4)

    assert result["mean_estimate_cuped"] == pytest.approx(result["mean_estimate"])
    assert result["rejection_rate_cuped"] == result["rejection_rate"]


def test_simulations_refuse_arms_too_small_to_analyse():
    tiny = PotentialOutcomes(control=BASE[:20], treatment=BASE[:20], covariate=ACTIVITY[:20])

    with pytest.raises(ValueError, match="at least 2 users"):
        operating_characteristics(tiny, n_sims=200, treatment_share=0.05, alpha=ALPHA, seed=0)


def test_random_assignments_follow_the_intended_share(rate_tolerance):
    assignments = random_assignments(n_sims=50, n_users=N_USERS, treatment_share=0.3, seed=0)

    assert assignments.shape == (50, N_USERS)
    assert assignments.mean() == pytest.approx(0.3, abs=rate_tolerance(0.3, assignments.size))


def test_the_assignment_check_passes_random_splits_and_flags_a_biased_one(rate_tolerance):
    n_sims = 1000
    fair = assignment_check(
        SAME, random_assignments(n_sims, N_USERS, 0.5, seed=5), ALPHA, treatment_share=0.5
    )
    biased = assignment_check(
        SAME, np.tile(np.median(ACTIVITY) < ACTIVITY, (10, 1)), ALPHA, treatment_share=0.5
    )

    for key in ("aa_rejection_rate", "covariate_imbalance_rate", "srm_rejection_rate"):
        assert fair[key] == pytest.approx(ALPHA, abs=rate_tolerance(ALPHA, n_sims))
    assert fair["assignments"] == n_sims
    assert biased["covariate_imbalance_rate"] == 1.0  # treatment got every active user


def test_peeking_inflates_false_alarms_and_always_valid_p_values_do_not(rate_tolerance):
    n_sims = 1000

    result = peeking(
        SAME, looks=10, n_sims=n_sims, treatment_share=0.5, alpha=ALPHA, tau=0.3, seed=3
    )

    tolerance = rate_tolerance(ALPHA, n_sims)
    assert result["fixed_horizon_rejection_rate"] == pytest.approx(ALPHA, abs=tolerance)
    assert result["naive_rejection_rate"] > ALPHA + tolerance  # ten looks, ten chances
    assert result["msprt_rejection_rate"] <= ALPHA + tolerance


def test_a_logging_bug_fakes_a_lift_that_the_sample_ratio_check_catches_once_large():
    result = srm_demo(SAME, ALTERNATING, drop_shares=(0.02, 0.4), alpha=ALPHA, treatment_share=0.5)

    small, large = result["drops"]
    assert result["true_effect"] == 0.0
    assert small["users_dropped"] == 6
    assert small["srm_p_value"] > SRM_ALPHA  # 294 vs 300 users: indistinguishable from chance
    assert large["users_dropped"] == 120
    assert large["estimate"]["difference"] > small["estimate"]["difference"] > 0
    assert large["decision"] == "invalid: sample ratio mismatch"

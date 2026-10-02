import math
import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from recsys.abtest.stats import (
    cuped_adjust,
    difference_in_means,
    mean_interval,
    minimum_detectable_effect,
    msprt_p_values,
    required_sample_size,
    sample_ratio_mismatch,
    sample_ratio_mismatch_rows,
)

ALPHA, POWER = 0.05, 0.8


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_difference_in_means_matches_welchs_t_test(rng):
    control, treatment = rng.normal(1.0, 1.0, size=400), rng.normal(1.3, 1.5, size=300)

    estimate = difference_in_means(control, treatment, alpha=ALPHA)

    welch = scipy_stats.ttest_ind(treatment, control, equal_var=False)
    interval = welch.confidence_interval(1 - ALPHA)
    assert estimate.difference == pytest.approx(treatment.mean() - control.mean())
    assert estimate.p_value == pytest.approx(welch.pvalue)
    assert (estimate.ci_low, estimate.ci_high) == pytest.approx((interval.low, interval.high))
    assert estimate.relative_lift == pytest.approx(estimate.difference / control.mean())
    assert (estimate.n_control, estimate.n_treatment) == (400, 300)


def test_difference_in_means_needs_two_users_per_arm(rng):
    with pytest.raises(ValueError, match="at least 2"):
        difference_in_means(np.array([1.0]), rng.normal(size=10))


def test_constant_arms_that_differ_are_a_certain_difference():
    differ = difference_in_means(np.ones(5), np.full(5, 2.0))
    same = difference_in_means(np.ones(5), np.ones(5))

    assert (differ.difference, differ.ci_low, differ.ci_high, differ.p_value) == (1, 1, 1, 0)
    assert same.p_value == 1.0


def test_cuped_keeps_the_mean_and_removes_the_variance_a_covariate_explains(rng):
    covariate = rng.normal(size=2000)
    outcome = 2.0 * covariate + rng.normal(scale=0.5, size=2000)

    adjusted, theta = cuped_adjust(outcome, covariate)

    # theta estimates the slope 2.0; its standard error is 0.5 / sqrt(2000) ~ 0.011
    assert theta == pytest.approx(2.0, abs=4 * 0.5 / math.sqrt(2000))
    assert adjusted.mean() == pytest.approx(outcome.mean())
    explained = np.corrcoef(outcome, covariate)[0, 1] ** 2
    assert adjusted.var() == pytest.approx(outcome.var() * (1 - explained), rel=1e-6)


def test_cuped_with_a_constant_covariate_changes_nothing(rng):
    outcome = rng.normal(size=50)

    adjusted, theta = cuped_adjust(outcome, np.ones(50))

    assert theta == 0.0
    np.testing.assert_allclose(adjusted, outcome)


def test_mean_interval_is_the_t_interval_for_a_mean(rng):
    values = rng.normal(0.2, 1.0, size=50)

    low, high = mean_interval(values, alpha=ALPHA)

    expected = scipy_stats.ttest_1samp(values, 0.0).confidence_interval(1 - ALPHA)
    assert (low, high) == pytest.approx((expected.low, expected.high))


def test_required_sample_size_inverts_the_minimum_detectable_effect():
    mde = minimum_detectable_effect(2.0, n_control=500, n_treatment=500, alpha=ALPHA, power=POWER)

    # (1.96 + 0.8416) * 2 * sqrt(2 / 500)
    assert mde == pytest.approx(2.8016 * 2.0 * math.sqrt(2 / 500), rel=1e-3)
    assert required_sample_size(2.0, 2.0, mde, treatment_share=0.5, alpha=ALPHA, power=POWER) == (
        1000
    )


def test_required_sample_size_reaches_the_target_power_with_unequal_arms(rng, rate_tolerance):
    sd_control, sd_treatment, effect, share, n_sims = 1.0, 2.0, 0.25, 0.3, 2000
    total = required_sample_size(sd_control, sd_treatment, effect, share, ALPHA, POWER)
    n_treatment = round(total * share)

    control = rng.normal(0.0, sd_control, size=(n_sims, total - n_treatment))
    treatment = rng.normal(effect, sd_treatment, size=(n_sims, n_treatment))
    p_values = scipy_stats.ttest_ind(treatment, control, axis=1, equal_var=False).pvalue

    assert (p_values < ALPHA).mean() == pytest.approx(POWER, abs=rate_tolerance(POWER, n_sims))


def test_power_analysis_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="effect"):
        required_sample_size(1.0, 1.0, effect=0.0)
    with pytest.raises(ValueError, match="share"):
        required_sample_size(1.0, 1.0, effect=0.1, treatment_share=1.0)
    with pytest.raises(ValueError, match="user"):
        minimum_detectable_effect(1.0, n_control=0, n_treatment=10)


def test_sample_ratio_mismatch_is_a_chi_square_test_of_the_split():
    expected = scipy_stats.chisquare([5000, 4700], f_exp=[4850, 4850]).pvalue

    assert sample_ratio_mismatch(5000, 5000, treatment_share=0.5) == pytest.approx(1.0)
    assert sample_ratio_mismatch(5000, 4700, treatment_share=0.5) == pytest.approx(expected)
    rows = sample_ratio_mismatch_rows(np.array([5000, 5000]), np.array([5000, 4700]), 0.5)
    np.testing.assert_allclose(rows, [1.0, expected])


def test_sample_ratio_mismatch_respects_an_uneven_intended_split():
    assert sample_ratio_mismatch(8000, 2000, treatment_share=0.2) == pytest.approx(1.0)
    assert sample_ratio_mismatch(5000, 5000, treatment_share=0.2) < 1e-6


def test_msprt_p_values_never_increase_and_stay_valid_under_the_null(rng, rate_tolerance):
    n_sims, looks = 300, np.arange(20, 401, 20)
    y = rng.normal(size=(n_sims, 2, 400))
    cumulative = np.cumsum(y, axis=2)[:, :, looks - 1] / looks
    differences = cumulative[:, 1] - cumulative[:, 0]

    p_values = msprt_p_values(differences, np.broadcast_to(2.0 / looks, differences.shape), 0.2)

    assert np.all(np.diff(p_values, axis=1) <= 0)
    # Always valid: 20 looks do not inflate false alarms above alpha.
    assert (p_values[:, -1] <= ALPHA).mean() <= ALPHA + rate_tolerance(ALPHA, n_sims)


def test_msprt_detects_a_real_effect_without_overflow_warnings():
    differences = np.full(10, 0.3)
    variances = 2.0 / np.arange(100, 1001, 100)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moderate = msprt_p_values(differences, variances, tau=0.3)
        overwhelming = msprt_p_values(np.full(3, 50.0), np.full(3, 1e-4), tau=0.3)

    assert moderate[-1] < 0.001
    assert moderate[0] > moderate[-1]  # the evidence builds up look by look
    np.testing.assert_array_equal(overwhelming, 0.0)

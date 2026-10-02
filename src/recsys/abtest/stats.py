"""The statistics of an A/B test, written out plainly.

* :func:`difference_in_means`: the treatment arm's average minus the control arm's, with a 95%
  confidence interval and a p-value from Welch's t-test (no equal-variance assumption).
* :func:`cuped_adjust`: CUPED (Deng et al., 2013). Part of why users differ in an experiment is
  how they behaved *before* it; subtracting the part a pre-experiment covariate predicts leaves
  the same average effect with less noise, so intervals get narrower.
* :func:`minimum_detectable_effect` / :func:`required_sample_size`: power analysis, done before
  the test. How big an effect could this many users reliably detect, and how many users would a
  given effect need? :func:`mean_interval` gives the range of effects worth planning for.
* :func:`sample_ratio_mismatch`: a chi-square test that the arms got the share of users they
  were meant to. A mismatch usually means a bug in assignment or logging, and invalidates the
  result whatever it says.
* :func:`msprt_p_values`: always-valid p-values (mixture sequential probability ratio test,
  Johari et al., 2017). Checking an ordinary p-value every day and stopping at the first
  "significant" one inflates false alarms; these p-values may be checked after every user.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

MIN_USERS_PER_ARM = 2


@dataclass(frozen=True)
class Estimate:
    control_mean: float
    treatment_mean: float
    difference: float  # treatment minus control
    ci_low: float
    ci_high: float
    p_value: float  # two-sided, Welch's t-test
    relative_lift: float  # difference / control mean
    n_control: int
    n_treatment: int


def difference_in_means(
    control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
) -> Estimate:
    if min(len(control), len(treatment)) < MIN_USERS_PER_ARM:
        raise ValueError(f"Each arm needs at least {MIN_USERS_PER_ARM} users")
    variance_c = control.var(ddof=1) / len(control)
    variance_t = treatment.var(ddof=1) / len(treatment)
    standard_error = math.sqrt(variance_c + variance_t)
    difference = float(treatment.mean() - control.mean())
    if standard_error == 0:  # both arms constant: any difference between them is certain
        p_value = 0.0 if difference else 1.0
        return _estimate(control, treatment, difference, (difference, difference), p_value)
    # Welch-Satterthwaite degrees of freedom
    dof = (variance_c + variance_t) ** 2 / (
        variance_c**2 / (len(control) - 1) + variance_t**2 / (len(treatment) - 1)
    )
    margin = stats.t.ppf(1 - alpha / 2, dof) * standard_error
    p_value = float(2 * stats.t.sf(abs(difference) / standard_error, dof))
    return _estimate(
        control, treatment, difference, (difference - margin, difference + margin), p_value
    )


def _estimate(
    control: np.ndarray,
    treatment: np.ndarray,
    difference: float,
    interval: tuple[float, float],
    p_value: float,
) -> Estimate:
    control_mean = float(control.mean())
    return Estimate(
        control_mean=control_mean,
        treatment_mean=float(treatment.mean()),
        difference=difference,
        ci_low=float(interval[0]),
        ci_high=float(interval[1]),
        p_value=p_value,
        relative_lift=difference / control_mean if control_mean else math.nan,
        n_control=len(control),
        n_treatment=len(treatment),
    )


def cuped_adjust(outcome: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """``outcome - theta * (covariate - mean)``, with theta = cov / var (pooled over both arms).

    The covariate must be measured before the experiment, so the treatment cannot move it."""
    variance = covariate.var()
    theta = float(np.cov(outcome, covariate, ddof=0)[0, 1] / variance) if variance else 0.0
    return outcome - theta * (covariate - covariate.mean()), theta


def _z(alpha: float, power: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def mean_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for the mean of ``values`` (a t-interval).

    Applied to each user's treatment-minus-control outcome, it says which average effects a
    *new* set of users could plausibly show, which is what a power analysis has to plan for."""
    if len(values) < MIN_USERS_PER_ARM:
        raise ValueError(f"An interval needs at least {MIN_USERS_PER_ARM} values")
    margin = (
        stats.t.ppf(1 - alpha / 2, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    )
    return float(values.mean() - margin), float(values.mean() + margin)


def minimum_detectable_effect(
    sd: float, n_control: int, n_treatment: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest true difference detected with probability ``power`` at significance ``alpha``."""
    if min(n_control, n_treatment) < 1:
        raise ValueError("Each arm needs at least one user")
    return _z(alpha, power) * sd * math.sqrt(1 / n_control + 1 / n_treatment)


def required_sample_size(
    sd_control: float,
    sd_treatment: float,
    effect: float,
    treatment_share: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Users needed in total (both arms) to detect ``effect``, with ``treatment_share`` of them
    in treatment. Each arm keeps its own spread: a policy can change how much users vary."""
    if effect == 0:
        raise ValueError("A zero effect cannot be detected with any number of users")
    if not 0 < treatment_share < 1:
        raise ValueError(f"treatment_share must be in (0, 1), got {treatment_share}")
    variance_per_user = sd_control**2 / (1 - treatment_share) + sd_treatment**2 / treatment_share
    exact = (_z(alpha, power) / effect) ** 2 * variance_per_user
    return math.ceil(exact - 1e-9)  # tolerate float noise just above a whole number


def sample_ratio_mismatch(n_control: int, n_treatment: int, treatment_share: float) -> float:
    """p-value that the observed split could come from the intended one."""
    return float(
        sample_ratio_mismatch_rows(np.array([n_control]), np.array([n_treatment]), treatment_share)[
            0
        ]
    )


def sample_ratio_mismatch_rows(
    n_control: np.ndarray, n_treatment: np.ndarray, treatment_share: float
) -> np.ndarray:
    """:func:`sample_ratio_mismatch` for many experiments at once (a chi-square test, 1 df)."""
    total = n_control + n_treatment
    expected_treatment = total * treatment_share
    gap = n_treatment - expected_treatment  # the control arm is off by the same amount
    chi_square = gap**2 * (1 / expected_treatment + 1 / (total - expected_treatment))
    return stats.chi2.sf(chi_square, df=1)


def msprt_p_values(differences: np.ndarray, variances: np.ndarray, tau: float) -> np.ndarray:
    """Always-valid p-values for a difference in means observed at successive looks.

    ``differences[i]`` is the estimate at look ``i`` and ``variances[i]`` its sampling variance;
    ``tau`` is the scale of effects the test is tuned to (here: the planned MDE). Works on the
    last axis, so a (simulations, looks) array is handled in one call.
    """
    with np.errstate(over="ignore"):  # overwhelming evidence: the ratio is inf, p is 0
        likelihood_ratio = np.sqrt(variances / (variances + tau**2)) * np.exp(
            tau**2 * differences**2 / (2 * variances * (variances + tau**2))
        )
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / likelihood_ratio), axis=-1)

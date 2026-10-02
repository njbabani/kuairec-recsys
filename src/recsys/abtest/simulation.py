"""Offline A/B experiments with a known right answer.

In a real experiment each user is seen under one arm only. Here every test user's reaction to
every video is recorded, so each user's outcome under *both* policies is known: the potential
outcomes. That gives the true average effect, and every statistical method can be checked
against it:

* :func:`readout`: one experiment as it would be run (users split by a stable hash), analysed
  with and without CUPED, with sample-ratio and covariate-balance checks, a guardrail and a
  ship decision;
* :func:`operating_characteristics`: thousands of re-randomised experiments: how often the test
  says "significant" (false alarms when there is no effect, power when there is), whether the
  estimate is unbiased and the 95% interval covers the truth 95% of the time;
* :func:`assignment_check`: the same for real assignments (e.g. one per hash salt): does the
  assignment itself behave like a fair coin?
* :func:`peeking`: the same experiment watched as users arrive: stopping at the first
  "significant" look vs always-valid (mSPRT) p-values;
* :func:`srm_demo`: a simulated logging bug that silently loses some treatment users.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

from recsys.abtest.stats import (
    MIN_USERS_PER_ARM,
    Estimate,
    cuped_adjust,
    difference_in_means,
    msprt_p_values,
    sample_ratio_mismatch,
    sample_ratio_mismatch_rows,
)

SRM_ALPHA = 0.001  # the usual, deliberately strict threshold for a sample ratio mismatch


@dataclass(frozen=True)
class PotentialOutcomes:
    """Every user's outcome under the control and the treatment policy, plus a covariate
    measured before the experiment (for CUPED). Arrays are aligned by user."""

    control: np.ndarray
    treatment: np.ndarray
    covariate: np.ndarray

    def __post_init__(self) -> None:
        if not self.control.shape == self.treatment.shape == self.covariate.shape:
            raise ValueError(
                "Control, treatment and covariate must describe the same users, got shapes "
                f"{self.control.shape}, {self.treatment.shape} and {self.covariate.shape}"
            )

    @property
    def true_effect(self) -> float:
        return float((self.treatment - self.control).mean())

    def observed(self, in_treatment: np.ndarray) -> np.ndarray:
        """What an experiment sees: each user's outcome under their own arm only."""
        return np.where(in_treatment, self.treatment, self.control)

    def subset(self, keep: np.ndarray) -> "PotentialOutcomes":
        return PotentialOutcomes(self.control[keep], self.treatment[keep], self.covariate[keep])


def _split(values: np.ndarray, in_treatment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values[~in_treatment], values[in_treatment]


def decide(primary: Estimate, guardrail: Estimate | None, srm_p_value: float, alpha: float) -> str:
    if srm_p_value < SRM_ALPHA:
        return "invalid: sample ratio mismatch"
    if primary.p_value >= alpha:
        return "inconclusive"
    if primary.difference < 0:
        return "do not ship: worse"
    if guardrail is not None and guardrail.ci_high < 0:
        return "do not ship: guardrail worse"
    return "ship"


def readout(
    outcomes: PotentialOutcomes,
    in_treatment: np.ndarray,
    guardrail: PotentialOutcomes | None,
    alpha: float,
    treatment_share: float,
) -> dict[str, Any]:
    """One experiment. The decision uses the CUPED estimate, the pre-registered analysis."""
    observed = outcomes.observed(in_treatment)
    estimate = difference_in_means(*_split(observed, in_treatment), alpha)
    adjusted, theta = cuped_adjust(observed, outcomes.covariate)
    cuped = difference_in_means(*_split(adjusted, in_treatment), alpha)
    srm_p_value = sample_ratio_mismatch(estimate.n_control, estimate.n_treatment, treatment_share)
    guard = (
        difference_in_means(*_split(guardrail.observed(in_treatment), in_treatment), alpha)
        if guardrail is not None
        else None
    )
    # The arms should not differ before the experiment: a pre-period A/A on the covariate.
    balance = difference_in_means(*_split(outcomes.covariate, in_treatment), alpha)
    return {
        "estimate": asdict(estimate),
        "cuped": {
            **asdict(cuped),
            "theta": theta,
            "variance_reduction": float(1 - adjusted.var() / observed.var()),
        },
        "srm_p_value": srm_p_value,
        "covariate_balance_p_value": balance.p_value,
        "guardrail": asdict(guard) if guard is not None else None,
        "true_effect": outcomes.true_effect,
        "true_guardrail_effect": guardrail.true_effect if guardrail is not None else None,
        "decision": decide(cuped, guard, srm_p_value, alpha),
    }


def _welch_rows(
    values: np.ndarray, in_treatment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference in means, its standard error and Welch's degrees of freedom, per row."""
    in_control = ~in_treatment
    n_t, n_c = in_treatment.sum(axis=1), in_control.sum(axis=1)
    if min(n_t.min(), n_c.min()) < MIN_USERS_PER_ARM:
        raise ValueError(
            f"Every simulated arm needs at least {MIN_USERS_PER_ARM} users; use more users or a "
            "treatment share closer to 0.5"
        )
    mean_t = (values * in_treatment).sum(axis=1) / n_t
    mean_c = (values * in_control).sum(axis=1) / n_c
    var_t = (((values - mean_t[:, None]) ** 2) * in_treatment).sum(axis=1) / (n_t - 1) / n_t
    var_c = (((values - mean_c[:, None]) ** 2) * in_control).sum(axis=1) / (n_c - 1) / n_c
    dof = (var_t + var_c) ** 2 / (var_t**2 / (n_t - 1) + var_c**2 / (n_c - 1))
    return mean_t - mean_c, np.sqrt(var_t + var_c), dof


def _cuped_rows(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """:func:`cuped_adjust` for every row of ``values`` at once."""
    centred = covariate - covariate.mean()
    if not centred.any():  # a constant covariate explains nothing
        return values
    theta = (values - values.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)
    return values - theta[:, None] * centred


def _summarise(
    values: np.ndarray, in_treatment: np.ndarray, alpha: float, truth: float
) -> dict[str, Any]:
    difference, standard_error, dof = _welch_rows(values, in_treatment)
    margin = stats.t.ppf(1 - alpha / 2, dof) * standard_error
    p_values = 2 * stats.t.sf(np.abs(difference) / standard_error, dof)
    covered = (difference - margin <= truth) & (truth <= difference + margin)
    return {
        "rejection_rate": float((p_values < alpha).mean()),
        "mean_estimate": float(difference.mean()),
        "bias": float(difference.mean() - truth),
        "ci_coverage": float(covered.mean()),
        "mean_ci_width": float((2 * margin).mean()),
        # Under no effect p-values are uniform: each tenth of [0, 1] should hold ~10% of them.
        "p_value_histogram": np.histogram(p_values, bins=10, range=(0, 1))[0].tolist(),
    }


def random_assignments(n_sims: int, n_users: int, treatment_share: float, seed: int) -> np.ndarray:
    """``n_sims`` independent coin-flip assignments of ``n_users`` (True: treatment)."""
    return np.random.default_rng(seed).random((n_sims, n_users)) < treatment_share


def characteristics(
    outcomes: PotentialOutcomes, assignments: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Run the experiment once per row of ``assignments`` and compare each analysis with the
    truth. With no true effect, ``rejection_rate`` is the false-alarm rate; with one, power."""
    observed = np.where(assignments, outcomes.treatment, outcomes.control)
    truth = outcomes.true_effect
    plain = _summarise(observed, assignments, alpha, truth)
    cuped = _summarise(_cuped_rows(observed, outcomes.covariate), assignments, alpha, truth)
    return {
        "simulations": int(assignments.shape[0]),
        "true_effect": truth,
        **plain,
        **{f"{name}_cuped": value for name, value in cuped.items()},
    }


def operating_characteristics(
    outcomes: PotentialOutcomes, n_sims: int, treatment_share: float, alpha: float, seed: int
) -> dict[str, Any]:
    """:func:`characteristics` over ``n_sims`` random re-assignments of the users."""
    assignments = random_assignments(n_sims, outcomes.control.size, treatment_share, seed)
    return characteristics(outcomes, assignments, alpha)


def assignment_check(
    outcomes: PotentialOutcomes, assignments: np.ndarray, alpha: float, treatment_share: float
) -> dict[str, Any]:
    """Judge real assignments (one per row, e.g. one per hash salt) as if they were coin flips.

    On an A/A comparison every rate should be close to ``alpha``: false alarms on the outcome,
    pre-experiment imbalance in the covariate, and sample-ratio alarms. Higher rates mean the
    assignment is not as random as the analysis assumes."""
    covariate = np.broadcast_to(outcomes.covariate, assignments.shape)
    n_treatment = assignments.sum(axis=1)
    srm_p_values = sample_ratio_mismatch_rows(
        assignments.shape[1] - n_treatment, n_treatment, treatment_share
    )
    aa = characteristics(outcomes, assignments, alpha)
    return {
        "assignments": int(assignments.shape[0]),
        "aa_rejection_rate": aa["rejection_rate"],
        "aa_rejection_rate_cuped": aa["rejection_rate_cuped"],
        "covariate_imbalance_rate": _summarise(covariate, assignments, alpha, 0.0)[
            "rejection_rate"
        ],
        "srm_rejection_rate": float((srm_p_values < alpha).mean()),
    }


def peeking(
    outcomes: PotentialOutcomes,
    looks: int,
    n_sims: int,
    treatment_share: float,
    alpha: float,
    tau: float,
    seed: int,
) -> dict[str, Any]:
    """Users arrive in random order and the result is checked ``looks`` times.

    * naive: stop at the first look whose ordinary p-value is below ``alpha``;
    * fixed horizon: look once, at the end (the ordinary test's assumption);
    * mSPRT: stop at the first look whose always-valid p-value is below ``alpha``.
    """
    rng = np.random.default_rng(seed)
    n_users = outcomes.control.size
    order = np.argsort(rng.random((n_sims, n_users)), axis=1)
    in_treatment = rng.random((n_sims, n_users)) < treatment_share
    observed = np.where(in_treatment, outcomes.treatment[order], outcomes.control[order])
    sizes = np.linspace(n_users / looks, n_users, looks).astype(int)
    columns = [_welch_rows(observed[:, :size], in_treatment[:, :size]) for size in sizes]
    differences = np.column_stack([difference for difference, _, _ in columns])
    errors = np.column_stack([error for _, error, _ in columns])
    p_values = np.column_stack([2 * stats.t.sf(np.abs(d) / e, dof) for d, e, dof in columns])
    always_valid = msprt_p_values(differences, errors**2, tau)
    return {
        "looks": looks,
        "users_per_look": sizes.tolist(),
        "simulations": n_sims,
        "true_effect": outcomes.true_effect,
        "naive_rejection_rate": float((p_values < alpha).any(axis=1).mean()),
        "fixed_horizon_rejection_rate": float((p_values[:, -1] < alpha).mean()),
        "msprt_rejection_rate": float((always_valid[:, -1] < alpha).mean()),
    }


def _without_least_engaged(
    outcomes: PotentialOutcomes,
    in_treatment: np.ndarray,
    drop_share: float,
    alpha: float,
    treatment_share: float,
) -> dict[str, Any]:
    treated = np.flatnonzero(in_treatment)
    least_engaged = treated[np.argsort(outcomes.observed(in_treatment)[treated], kind="stable")]
    n_dropped = round(drop_share * treated.size)
    logged = np.ones(in_treatment.size, dtype=bool)
    logged[least_engaged[:n_dropped]] = False
    result = readout(outcomes.subset(logged), in_treatment[logged], None, alpha, treatment_share)
    return {
        "drop_share": drop_share,
        "users_dropped": n_dropped,
        "srm_p_value": result["srm_p_value"],
        "estimate": result["cuped"],
        "decision": result["decision"],
    }


def srm_demo(
    outcomes: PotentialOutcomes,
    in_treatment: np.ndarray,
    drop_shares: tuple[float, ...],
    alpha: float,
    treatment_share: float,
) -> dict[str, Any]:
    """A logging bug loses the sessions of the treatment users who liked the fewest videos
    (say the new version only logs a session once something is liked).

    Meant for an A/A comparison, so any lift is fake. Each loss is analysed exactly like a real
    experiment (CUPED, then the sample-ratio check and the ship decision). CUPED cannot undo
    this bias: users are lost for how they reacted *during* the experiment, which no
    pre-experiment covariate predicts fully."""
    return {
        "true_effect": outcomes.true_effect,
        "without_bug": _without_least_engaged(outcomes, in_treatment, 0.0, alpha, treatment_share),
        "drops": [
            _without_least_engaged(outcomes, in_treatment, share, alpha, treatment_share)
            for share in drop_shares
        ],
    }

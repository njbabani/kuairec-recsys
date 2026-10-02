"""The A/B lab: the pipeline's own statistics, run with a visitor's settings.

Nothing here is new statistics. Each function sets up the inputs the ``ab_test`` stage would and
calls the same code (:mod:`recsys.abtest`), so what the app shows is what the pipeline does.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from recsys.abtest.outcomes import SESSION_METRICS
from recsys.abtest.simulation import operating_characteristics, peeking, readout, srm_demo
from recsys.abtest.stats import minimum_detectable_effect
from recsys.experiments.ab_test import UserTable, assign, users_needed

PRIMARY, GUARDRAIL = SESSION_METRICS
MAX_LOOKS = 20
# Users expected in the smaller arm at the first look. Fewer, and some of thousands of simulated
# experiments draw an arm with under two users, which a t-test cannot use.
MIN_FIRST_LOOK_ARM = 20


def _arm_sizes(n_users: int, treatment_share: float) -> tuple[int, int]:
    n_treatment = round(n_users * treatment_share)
    return n_users - n_treatment, n_treatment


def max_looks(n_users: int, treatment_share: float) -> int:
    """The most interim looks (up to ``MAX_LOOKS``) that still give the smaller arm about
    ``MIN_FIRST_LOOK_ARM`` users at the first look; below 2, peeking cannot be simulated."""
    smaller_arm = n_users * min(treatment_share, 1 - treatment_share)
    return min(MAX_LOOKS, math.floor(smaller_arm / MIN_FIRST_LOOK_ARM))


def run_experiment(
    table: UserTable,
    control: str,
    treatment: str,
    salt: str,
    treatment_share: float,
    alpha: float,
) -> dict[str, Any]:
    """One experiment: users split by a stable hash of ``salt``, read out like the pipeline."""
    return readout(
        table.potential(control, treatment, PRIMARY),
        assign(table.users, salt, treatment_share),
        table.potential(control, treatment, GUARDRAIL),
        alpha,
        treatment_share,
    )


def plan(
    table: UserTable,
    control: str,
    treatment: str,
    n_users: int,
    treatment_share: float,
    alpha: float,
    power: float,
) -> dict[str, Any]:
    """Before running: the smallest effect ``n_users`` can detect, and the users the true
    effect would need."""
    outcome = table.potential(control, treatment, PRIMARY)
    sd = float(outcome.control.std(ddof=1))
    mde = minimum_detectable_effect(sd, *_arm_sizes(n_users, treatment_share), alpha, power)
    varies = outcome.control.std() > 0 and outcome.covariate.std() > 0
    correlation = float(np.corrcoef(outcome.control, outcome.covariate)[0, 1]) if varies else 0.0
    return {
        "sd": sd,
        "mde": mde,
        "covariate_correlation": correlation,
        "mde_cuped": mde * math.sqrt(1 - correlation**2),
        "true_effect": outcome.true_effect,
        "users_needed": users_needed(outcome, treatment_share, alpha, power),
    }


def mde_curve(
    sd: float, users: Sequence[int], treatment_share: float, alpha: float, power: float
) -> pl.DataFrame:
    """The smallest detectable effect at each experiment size in ``users``."""
    return pl.DataFrame(
        {
            "users": list(users),
            "mde": [
                minimum_detectable_effect(sd, *_arm_sizes(n, treatment_share), alpha, power)
                for n in users
            ],
        }
    )


def rerandomise(
    table: UserTable,
    control: str,
    treatment: str,
    n_sims: int,
    treatment_share: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """The experiment repeated ``n_sims`` times with fresh random splits (the same policy in
    both arms makes it an A/A test)."""
    outcome = table.potential(control, treatment, PRIMARY)
    return operating_characteristics(outcome, n_sims, treatment_share, alpha, seed)


def peek(
    table: UserTable,
    control: str,
    treatment: str,
    looks: int,
    n_sims: int,
    treatment_share: float,
    alpha: float,
    tau: float,
    seed: int,
) -> dict[str, Any]:
    outcome = table.potential(control, treatment, PRIMARY)
    return peeking(outcome, looks, n_sims, treatment_share, alpha, tau, seed)


def logging_bug(
    table: UserTable,
    policy: str,
    salt: str,
    treatment_share: float,
    drop_share: float,
    alpha: float,
) -> dict[str, Any]:
    """An A/A test on ``policy`` read out with and without a bug that loses the least engaged
    ``drop_share`` of treatment users."""
    result = srm_demo(
        table.potential(policy, policy, PRIMARY),
        assign(table.users, salt, treatment_share),
        (drop_share,),
        alpha,
        treatment_share,
    )
    return {
        "true_effect": result["true_effect"],
        "without_bug": result["without_bug"],
        "with_bug": result["drops"][0],
    }

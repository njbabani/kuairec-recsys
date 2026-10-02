"""Offline A/B experiments on the fully observed test users.

Run as a DVC stage: ``python -m recsys.experiments.ab_test``. Each policy's session for every
test user (from ``final_evaluation``) is turned into what the user did with it: videos liked
(the primary metric) and watch time (a guardrail). Users are split into arms by a stable hash.
Because every user's outcome under every policy is known, each analysis is checked against the
true effect (see :mod:`recsys.abtest.simulation`):

* a readout per planned experiment (with CUPED on a pre-experiment covariate, sample-ratio and
  balance checks, the guardrail and a ship decision), its operating characteristics and the
  users it would need;
* a power analysis; an A/A test; a check of the hash assignment itself over many salts; the
  peeking demonstration; a simulated logging bug.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from recsys.abtest.outcomes import SESSION_METRICS, session_outcomes
from recsys.abtest.simulation import (
    PotentialOutcomes,
    assignment_check,
    operating_characteristics,
    peeking,
    readout,
    srm_demo,
)
from recsys.abtest.stats import (
    difference_in_means,
    mean_interval,
    minimum_detectable_effect,
    required_sample_size,
    sample_ratio_mismatch,
)
from recsys.config import ABExperimentParams, ABTestParams, load_params
from recsys.hashing import stable_fraction
from recsys.io import PARQUET_COMPRESSION, write_json
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

PRIMARY, GUARDRAIL = SESSION_METRICS  # videos liked per session; watch time per session
COVARIATE = "pre-experiment positive rate (each user's training weeks)"


@dataclass(frozen=True)
class ABTestPaths:
    sessions_path: Path
    splits_dir: Path
    metrics_path: Path
    outcomes_path: Path


def pre_experiment_covariate(train: pl.DataFrame) -> pl.DataFrame:
    """Each user's positive rate before the experiment: the treatment cannot have moved it."""
    return train.group_by("user_id").agg(covariate=pl.col("is_positive").mean())


def assign(users: pl.Series, salt: str, treatment_share: float) -> np.ndarray:
    """Each user's arm from a stable hash of ``salt:user_id`` (True: treatment)."""
    return np.array([stable_fraction(f"{salt}:{user}") < treatment_share for user in users])


def check_sessions(sessions: pl.DataFrame, users: pl.Series, length: int) -> None:
    """Every policy must give every test user a full session, or the arms of an experiment
    would compare different users or different amounts of exposure."""
    expected = set(users.to_list())
    for policy in sessions["policy"].unique().sort():
        sizes = sessions.filter(pl.col("policy") == policy).group_by("user_id").len()
        covered = set(sizes["user_id"].to_list())
        if covered != expected:
            raise ValueError(
                f"{policy}: {len(covered & expected)} of {len(expected)} test users have a "
                f"session, plus {len(covered - expected)} users outside the test set"
            )
        short = sizes.filter(pl.col("len") != length).height
        if short:
            raise ValueError(f"{policy}: sessions of {short} users are not {length} videos long")


def user_outcomes(
    sessions: pl.DataFrame,
    reactions: pl.DataFrame,
    covariates: pl.DataFrame,
    params: ABTestParams,
) -> pl.DataFrame:
    """One row per (test user, policy): the session's outcome, the covariate and the arm."""
    if covariates.is_empty():
        raise ValueError("No training views: the pre-experiment covariate cannot be computed")
    users = reactions["user_id"].unique().sort()
    check_sessions(sessions, users, params.session_length)
    outcomes = pl.concat(
        [
            session_outcomes(
                sessions.filter(pl.col("policy") == policy).drop("policy"), reactions
            ).with_columns(policy=pl.lit(policy))
            for policy in sessions["policy"].unique().sort()
        ]
    )
    assignment = pl.DataFrame(
        {"user_id": users, "in_treatment": assign(users, params.salt, params.treatment_share)}
    )
    joined = outcomes.join(assignment, on="user_id").join(covariates, on="user_id", how="left")
    missing = joined.filter(pl.col("covariate").is_null())["user_id"].n_unique()
    if missing:
        logger.warning("%d test users have no training views; using the average rate", missing)
    mean_rate = float(covariates["covariate"].mean())
    return joined.with_columns(pl.col("covariate").fill_null(mean_rate)).sort("policy", "user_id")


class UserTable:
    """Per-user outcome arrays aligned by user id, for any policy and metric."""

    def __init__(self, outcomes: pl.DataFrame) -> None:
        self.outcomes = outcomes
        first = outcomes.filter(pl.col("policy") == outcomes["policy"][0]).sort("user_id")
        self.users = first["user_id"]
        self.covariate = first["covariate"].to_numpy()
        self.in_treatment = first["in_treatment"].to_numpy()

    def values(self, policy: str, metric: str) -> np.ndarray:
        rows = self.outcomes.filter(pl.col("policy") == policy).sort("user_id")
        if not rows["user_id"].equals(self.users):
            raise ValueError(
                f"Policy {policy} covers {rows.height} users, not the {self.users.len()} "
                "every other policy covers"
            )
        return rows[metric].cast(pl.Float64).to_numpy()

    def potential(self, control: str, treatment: str, metric: str) -> PotentialOutcomes:
        return PotentialOutcomes(
            self.values(control, metric), self.values(treatment, metric), self.covariate
        )


def power_analysis(
    table: UserTable, experiment: ABExperimentParams, params: ABTestParams
) -> dict[str, Any]:
    """Before running: the smallest effect on the primary metric this many users can detect."""
    control = table.values(experiment.control, PRIMARY)
    sd = float(control.std(ddof=1))
    correlation = float(np.corrcoef(control, table.covariate)[0, 1])
    n_treatment = int(table.in_treatment.sum())
    n_control = int(table.in_treatment.size - n_treatment)
    mde = minimum_detectable_effect(sd, n_control, n_treatment, params.alpha, params.power)
    return {
        "metric": PRIMARY,
        "control_policy": experiment.control,
        "control_mean": float(control.mean()),
        "sd": sd,
        "users_control": n_control,
        "users_treatment": n_treatment,
        "mde": mde,
        "relative_mde": mde / float(control.mean()),
        "covariate_correlation": correlation,
        "mde_cuped": mde * float(np.sqrt(1 - correlation**2)),
    }


def users_needed(
    outcome: PotentialOutcomes, treatment_share: float, alpha: float, power: float
) -> dict[str, Any]:
    """Users (both arms together) for the planned power: at the true effect on these users, and
    across the 95% interval of effects a new set of users could plausibly show."""
    spreads = (float(outcome.control.std(ddof=1)), float(outcome.treatment.std(ddof=1)))

    def needed(effect: float) -> int | None:
        if effect == 0:
            return None
        return required_sample_size(*spreads, abs(effect), treatment_share, alpha, power)

    low, high = mean_interval(outcome.treatment - outcome.control, alpha)
    smaller, larger = sorted((abs(low), abs(high)))
    # The largest plausible effect needs the fewest users. If the interval reaches zero, no
    # number of users is guaranteed to be enough.
    most = None if low <= 0 <= high else needed(smaller)
    return {
        "at_true_effect": needed(outcome.true_effect),
        "effect_interval": [low, high],
        "range": [needed(larger), most],
    }


def analyse_experiment(
    table: UserTable, experiment: ABExperimentParams, params: ABTestParams, seed: int
) -> dict[str, Any]:
    outcome = table.potential(experiment.control, experiment.treatment, PRIMARY)
    guardrail = table.potential(experiment.control, experiment.treatment, GUARDRAIL)
    return {
        "control": experiment.control,
        "treatment": experiment.treatment,
        "readout": readout(
            outcome, table.in_treatment, guardrail, params.alpha, params.treatment_share
        ),
        "operating_characteristics": operating_characteristics(
            outcome, params.simulations, params.treatment_share, params.alpha, seed
        ),
        "users_needed": users_needed(outcome, params.treatment_share, params.alpha, params.power),
    }


def _assignment(table: UserTable, same: PotentialOutcomes, params: ABTestParams) -> dict[str, Any]:
    """The hash split every experiment used, and the hash itself judged over many other salts."""
    in_treatment, covariate = table.in_treatment, table.covariate
    n_treatment = int(in_treatment.sum())
    n_control = in_treatment.size - n_treatment
    salts = [f"{params.salt}-check-{index}" for index in range(params.simulations)]
    other = np.vstack([assign(table.users, salt, params.treatment_share) for salt in salts])
    balance = difference_in_means(covariate[~in_treatment], covariate[in_treatment], params.alpha)
    return {
        "salt": params.salt,
        "users_control": n_control,
        "users_treatment": n_treatment,
        "srm_p_value": sample_ratio_mismatch(n_control, n_treatment, params.treatment_share),
        "covariate_balance_p_value": balance.p_value,
        "other_salts": assignment_check(same, other, params.alpha, params.treatment_share),
    }


def _validation(table: UserTable, params: ABTestParams, tau: float) -> dict[str, Any]:
    """The A/A test, the assignment check, the peeking demonstration and the logging bug."""
    same = table.potential(params.aa_policy, params.aa_policy, PRIMARY)
    primary = params.experiment(params.primary_experiment)
    real = table.potential(primary.control, primary.treatment, PRIMARY)
    share, alpha, sims = params.treatment_share, params.alpha, params.simulations
    return {
        "aa_test": {
            "policy": params.aa_policy,
            **operating_characteristics(same, sims, share, alpha, params.seed),
        },
        "assignment": _assignment(table, same, params),
        "peeking": {
            "tau": tau,
            "aa": peeking(same, params.looks, sims, share, alpha, tau, params.seed + 1),
            "ab": peeking(real, params.looks, sims, share, alpha, tau, params.seed + 2),
        },
        "srm_demo": {
            "policy": params.aa_policy,
            **srm_demo(same, table.in_treatment, params.srm_drop_shares, alpha, share),
        },
    }


def _policy_means(table: UserTable) -> dict[str, dict[str, float]]:
    policies = table.outcomes["policy"].unique().sort().to_list()
    return {
        policy: {metric: float(table.values(policy, metric).mean()) for metric in SESSION_METRICS}
        for policy in policies
    }


def run_ab_test(paths: ABTestPaths, params: ABTestParams, tracker: Tracker) -> dict[str, Any]:
    reactions = pl.read_parquet(
        paths.splits_dir / "test.parquet",
        columns=["user_id", "video_id", "is_positive", "play_duration"],
    )
    train = pl.read_parquet(paths.splits_dir / "train.parquet", columns=["user_id", "is_positive"])
    outcomes = user_outcomes(
        pl.read_parquet(paths.sessions_path), reactions, pre_experiment_covariate(train), params
    )
    table = UserTable(outcomes)
    power = power_analysis(table, params.experiment(params.primary_experiment), params)
    with tracker.run("ab_test", {"ab_test": params.model_dump(mode="json")}) as run:
        experiments = {
            experiment.name: analyse_experiment(table, experiment, params, params.seed + index)
            for index, experiment in enumerate(params.experiments, start=10)
        }
        run.log_metrics(
            {
                f"{name}.difference_cuped": result["readout"]["cuped"]["difference"]
                for name, result in experiments.items()
            }
        )
    report = {
        "users": table.users.len(),
        "session_length": params.session_length,
        "design": params.model_dump(
            mode="json", include={"treatment_share", "alpha", "power", "simulations", "looks"}
        ),
        "metrics": {"primary": PRIMARY, "guardrail": GUARDRAIL},
        "covariate": COVARIATE,
        "policies": _policy_means(table),
        "power": power,
        "experiments": experiments,
        **_validation(table, params, tau=power["mde"]),
    }
    write_json(report, paths.metrics_path)
    paths.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.write_parquet(paths.outcomes_path, compression=PARQUET_COMPRESSION)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    data = params.data
    run_ab_test(
        ABTestPaths(
            sessions_path=data.ab_dir / "sessions.parquet",
            splits_dir=data.splits_dir,
            metrics_path=data.metrics_dir / "ab_test.json",
            outcomes_path=data.ab_dir / "user_outcomes.parquet",
        ),
        params.ab_test,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

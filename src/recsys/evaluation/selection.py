"""The winner's curse: picking the best of many trials on the same users that score it.

When a hyperparameter search selects its winner on a set of users and then reports the winner's
score on those same users, the score is optimistic: part of the winning margin is luck that
will not repeat on new users.
"""

from typing import Any

import numpy as np
import polars as pl

MIN_USERS = 2


def selection_optimism(scores: pl.DataFrame, *, n_samples: int, seed: int) -> float:
    """Bootstrap estimate of how much selecting the best trial inflates its score.

    ``scores`` has one row per user and one column per trial (each trial's per-user metric).
    Each bootstrap sample of users picks the trial with the best in-sample mean, then measures
    how much better that trial looks in-sample than on the users left out of the sample. The
    average gap is the optimism; subtract it from the reported best score.
    """
    if scores.height < MIN_USERS:
        raise ValueError(f"Need at least {MIN_USERS} users to estimate selection optimism")
    if sum(scores.null_count().row(0)):
        raise ValueError("Every user needs a score for every trial; found missing scores")

    matrix = scores.to_numpy().astype(float)
    n_users = matrix.shape[0]
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_samples):
        in_sample = rng.integers(0, n_users, size=n_users)
        left_out = np.setdiff1d(np.arange(n_users), in_sample)
        if left_out.size == 0:
            continue
        best = int(np.argmax(matrix[in_sample].mean(axis=0)))
        gaps.append(matrix[in_sample, best].mean() - matrix[left_out, best].mean())
    return float(np.mean(gaps))


def summarise_search(
    trials: list[dict[str, Any]],
    per_user_scores: dict[str, pl.Series],
    select_metric: str,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """The best trial plus how much picking it on the same users flatters its score.

    ``per_user_scores`` holds each trial's per-user ``select_metric`` (same users, same order).
    """
    best = max(trials, key=lambda trial: trial["metrics"][select_metric])
    optimism = selection_optimism(pl.DataFrame(per_user_scores), n_samples=n_samples, seed=seed)
    return {
        "select_metric": select_metric,
        "best": best,
        "selection_optimism": optimism,
        "best_score_optimism_corrected": best["metrics"][select_metric] - optimism,
        "trials": trials,
    }

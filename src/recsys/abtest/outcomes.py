"""What a user does with the videos a policy shows them.

A policy's *session* for a user is its top-``length`` videos by score. Because the test matrix is
fully observed, each of those videos already has the user's recorded reaction, so a session's
outcome is looked up, not modelled: how many of the videos the user liked (the primary metric)
and how long they watched in total (a guardrail).
"""

import polars as pl

PAIR_KEYS = ["user_id", "video_id"]
SESSION_METRICS = ("liked", "watch_time_s")
MS_PER_SECOND = 1000


def _require_rankable(scores: pl.Series) -> None:
    """Null and NaN sort *first*, so a broken model's junk would fill the session. -inf is fine:
    as in the evaluation harness, it ranks a video last (ALS does this for unknown videos)."""
    if scores.null_count() or (scores.dtype.is_float() and scores.is_nan().any()):
        raise ValueError("Session scores must not be null or NaN (use -inf to rank a video last)")


def sessions(scores: pl.DataFrame, score_column: str, length: int) -> pl.DataFrame:
    """Each user's top ``length`` videos by ``score_column`` (ties: lower video id first)."""
    _require_rankable(scores[score_column])
    return (
        scores.sort(["user_id", score_column, "video_id"], descending=[False, True, False])
        .with_columns(position=pl.int_range(1, pl.len() + 1).over("user_id"))
        .filter(pl.col("position") <= length)
        .select(*PAIR_KEYS, "position")
    )


def session_outcomes(session: pl.DataFrame, reactions: pl.DataFrame) -> pl.DataFrame:
    """Per user: ``liked`` (positives among the session's videos) and ``watch_time_s``."""
    joined = session.join(
        reactions.select(*PAIR_KEYS, "is_positive", "play_duration"), on=PAIR_KEYS, how="left"
    )
    missing = joined.select(
        pl.any_horizontal(pl.col("is_positive", "play_duration").is_null()).sum()
    ).item()
    if missing:
        raise ValueError(f"{missing} session video(s) have no complete recorded reaction")
    return (
        joined.group_by("user_id")
        .agg(
            liked=pl.col("is_positive").sum(),
            watch_time_s=pl.col("play_duration").sum() / MS_PER_SECOND,
        )
        .sort("user_id")
    )

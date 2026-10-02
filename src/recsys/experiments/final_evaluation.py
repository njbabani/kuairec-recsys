"""The final, one-time evaluation on the untouched test users, and their A/B sessions.

Run as a DVC stage: ``python -m recsys.experiments.final_evaluation``. Every model is frozen:
fitted on train with the settings chosen earlier (on the tune users; the re-ranker on held-out
validation users). This is the first and only time the test users' reactions score a model.
The same scores give each policy's *session* for every test user (its top videos), which the
``ab_test`` stage turns into simulated experiments. Like the re-ranker, it never loads PyTorch:
two-tower scores come precomputed.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from recsys.abtest.outcomes import PAIR_KEYS, sessions
from recsys.config import EvaluationParams, ModelParams, RerankerParams, load_params
from recsys.evaluation.ranking import EvaluationResult, evaluate, popularity_percentiles
from recsys.experiments.reranker import InputPaths, Inputs, fit_stage_one, load_inputs
from recsys.experiments.runner import compare_with_reference, load_split
from recsys.io import PARQUET_COMPRESSION, write_json
from recsys.models.base import Scorer
from recsys.models.popularity import RandomRecommender
from recsys.reranking.features import FeatureSources, RankingFeatures, StageOneScorers
from recsys.reranking.ranker import load_ranker
from recsys.reranking.two_stage import LightGBMReranker, TwoStageRecommender
from recsys.tracking import Tracker, build_tracker

logger = logging.getLogger(__name__)

TEST_SPLIT = "test"
POLICIES = ("random", "popularity_engagement", "als", "two_tower", "two_stage")
REFERENCES = ("two_tower", "als")


@dataclass(frozen=True)
class FinalEvaluationPaths:
    inputs: InputPaths
    reranker_model: Path
    metrics_path: Path
    sessions_path: Path


def policy_scorers(
    inputs: Inputs,
    stage_one: StageOneScorers,
    reranker: LightGBMReranker,
    models: ModelParams,
    top_n: int,
) -> dict[str, Scorer]:
    """Every policy that can be compared, in ``POLICIES`` order."""
    scorers: dict[str, Scorer] = {
        "random": RandomRecommender(models.random.seed),
        "popularity_engagement": stage_one.engagement,
        "als": stage_one.als,
        "two_tower": inputs.retrieval,
        "two_stage": TwoStageRecommender(inputs.retrieval, reranker, top_n),
    }
    return {name: scorers[name] for name in POLICIES}


def _score_policies(
    scorers: dict[str, Scorer],
    test: pl.DataFrame,
    popularity: pl.DataFrame,
    evaluation: EvaluationParams,
    session_length: int,
) -> tuple[dict[str, EvaluationResult], pl.DataFrame]:
    pairs = test.select(PAIR_KEYS)
    results, session_frames = {}, []
    for name, scorer in scorers.items():
        scored = test.with_columns(score=scorer.score(pairs))
        results[name] = evaluate(
            scored,
            ks=evaluation.ks,
            popularity=popularity,
            n_bootstrap=evaluation.bootstrap_samples,
            seed=evaluation.seed,
        )
        session = sessions(scored, "score", session_length)
        session_frames.append(session.select(pl.lit(name).alias("policy"), pl.all()))
        logger.info("%-22s test %s", name, results[name].summary)
    return results, pl.concat(session_frames)


def run_final_evaluation(
    paths: FinalEvaluationPaths,
    models: ModelParams,
    reranker_params: RerankerParams,
    evaluation: EvaluationParams,
    session_length: int,
    cutoff: date,
    tracker: Tracker,
) -> dict[str, Any]:
    inputs = load_inputs(paths.inputs)
    stage_one = fit_stage_one(inputs, models)
    sources = FeatureSources(
        inputs.train, inputs.user_features, inputs.video_features, inputs.daily_stats, cutoff
    )
    features = RankingFeatures(stage_one, reranker_params.prior_strength).fit(sources)
    reranker = LightGBMReranker(features, load_ranker(paths.reranker_model))
    test = load_split(paths.inputs.splits_dir, TEST_SPLIT)
    scorers = policy_scorers(inputs, stage_one, reranker, models, reranker_params.retrieve_top_n)

    with tracker.run("final_evaluation/test", {"split": TEST_SPLIT, "policies": POLICIES}) as run:
        results, all_sessions = _score_policies(
            scorers,
            test,
            popularity_percentiles(inputs.train, test["video_id"]),
            evaluation,
            session_length,
        )
        run.log_metrics(
            {
                f"{name}.{evaluation.select_metric}": result.summary[evaluation.select_metric]
                for name, result in results.items()
            }
        )
    report = {
        "split": TEST_SPLIT,
        "users": results["two_tower"].summary["users_evaluated"],
        "models": {name: result.summary for name, result in results.items()},
        "paired": {
            reference: compare_with_reference(results, reference, evaluation)
            for reference in REFERENCES
        },
    }
    write_json(report, paths.metrics_path)
    paths.sessions_path.parent.mkdir(parents=True, exist_ok=True)
    all_sessions.write_parquet(paths.sessions_path, compression=PARQUET_COMPRESSION)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    data = params.data
    run_final_evaluation(
        FinalEvaluationPaths(
            inputs=InputPaths.from_params(data),
            reranker_model=data.models_dir / "reranker" / "model.txt",
            metrics_path=data.metrics_dir / "test_evaluation.json",
            sessions_path=data.ab_dir / "sessions.parquet",
        ),
        params.models,
        params.reranker,
        params.evaluation,
        params.ab_test.session_length,
        params.split.valid_start,
        build_tracker(params.tracking, workdir=Path.cwd()),
    )


if __name__ == "__main__":
    main()

import json
import subprocess
import sys
from typing import get_args

import polars as pl
import pytest

from recsys.config import Policy
from recsys.experiments.final_evaluation import (
    POLICIES,
    FinalEvaluationPaths,
    run_final_evaluation,
)
from recsys.experiments.reranker import run_reranker
from recsys.tracking import InMemoryTracker

SESSION_LENGTH = 2


def evaluate(reranker_world, evaluation_params, toy_world, out_dir):
    paths = FinalEvaluationPaths(
        inputs=reranker_world.inputs,
        reranker_model=reranker_world.model_dir / "model.txt",
        metrics_path=out_dir / "test_evaluation.json",
        sessions_path=out_dir / "ab" / "sessions.parquet",
    )
    tracker = InMemoryTracker()
    report = run_final_evaluation(
        paths,
        toy_world.models,
        toy_world.reranker_params,
        evaluation_params,
        SESSION_LENGTH,
        toy_world.cutoff,
        tracker,
    )
    return {"report": report, "paths": paths, "tracker": tracker}


@pytest.fixture
def evaluated(reranker_world, evaluation_params, toy_world, tmp_path):
    """Train the toy re-ranker, then evaluate every policy on the toy test users."""
    run_reranker(
        reranker_world,
        toy_world.models,
        toy_world.reranker_params,
        evaluation_params,
        toy_world.cutoff,
        InMemoryTracker(),
    )
    return evaluate(reranker_world, evaluation_params, toy_world, tmp_path / "first")


def test_every_comparable_policy_is_evaluated():
    assert set(POLICIES) == set(get_args(Policy))


@pytest.mark.integration
def test_every_policy_is_evaluated_once_on_the_test_users(evaluated):
    report, paths = evaluated["report"], evaluated["paths"]

    assert set(report["models"]) == set(POLICIES)
    assert report["split"] == "test"
    assert set(report["paired"]) == {"two_tower", "als"}
    assert [run["name"] for run in evaluated["tracker"].runs] == ["final_evaluation/test"]
    assert json.loads(paths.metrics_path.read_text()) == report


@pytest.mark.integration
def test_each_policy_writes_a_session_per_test_user(evaluated):
    sessions = pl.read_parquet(evaluated["paths"].sessions_path)

    assert set(sessions["policy"].unique()) == set(POLICIES)
    assert sessions.columns == ["policy", "user_id", "video_id", "position"]
    per_user = sessions.group_by("policy", "user_id").len()
    assert per_user["len"].to_list() == [SESSION_LENGTH] * per_user.height
    assert set(sessions["user_id"].unique()) == {100}  # the toy world's only test user
    two_tower = sessions.filter(pl.col("policy") == "two_tower").sort("position")
    # Toy two-tower scores are video_id % 3: videos 11 and 14 score highest (ties: lower id).
    assert two_tower["video_id"].to_list() == [11, 14]


@pytest.mark.integration
def test_sessions_never_depend_on_the_test_users_reactions(
    evaluated, reranker_world, evaluation_params, toy_world, tmp_path
):
    test_split = reranker_world.inputs.splits_dir / "test.parquet"
    labels = pl.read_parquet(test_split)
    labels.with_columns(is_positive=~pl.col("is_positive")).write_parquet(test_split)

    flipped = evaluate(reranker_world, evaluation_params, toy_world, tmp_path / "flipped")

    original = pl.read_parquet(evaluated["paths"].sessions_path)
    assert pl.read_parquet(flipped["paths"].sessions_path).equals(original)
    assert flipped["report"]["models"] != evaluated["report"]["models"]  # the labels did change


def test_the_final_evaluation_never_imports_pytorch():
    # Also scans every project module it loads, so a lazy ``import torch`` inside a function
    # (which an import check alone would miss) fails the test too.
    probe = (
        "import re, sys, recsys.experiments.final_evaluation\n"
        "pattern = re.compile(r'^\\s*(import|from) torch\\b', re.MULTILINE)\n"
        "lazy = sorted(name for name, module in sys.modules.items() if name.startswith('recsys')\n"
        "              and pattern.search(open(module.__file__, encoding='utf-8').read()))\n"
        "print('torch' in sys.modules, lazy)"
    )

    result = subprocess.run(  # noqa: S603 (fixed command, no user input)
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False []"

"""Unit tests for Milestone 2 Plan 6 — Experiment Ranking."""

from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.knowledge import KnowledgeBase
from labpilot.experiments.models import (
    KnowledgeEffect,
    KnowledgeEntry,
)
from labpilot.experiments.ranking import RankingWeights, rank_candidates
from labpilot.experiments.manifest import RunManifest, StageStatus, save_manifest


def _seed_empty_run(runs_dir: Path, run_id: str, competition: str = "titanic") -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        competition=competition,
        status=StageStatus.COMPLETED,
        stages=[],
        metadata={},
    )
    save_manifest(run_dir, manifest)


def test_rank_prefers_known_good_loss_tag(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    runs = tmp_path / "runs"
    _seed_empty_run(runs, "base")

    kb = KnowledgeBase(knowledge, "titanic")
    kb._entries[("loss", "cv_accuracy")] = KnowledgeEntry(
        technique="loss",
        metric_key="cv_accuracy",
        effect=KnowledgeEffect.IMPROVES,
        delta_estimate=0.05,
        confidence=0.9,
        sample_size=3,
        evidence_run_ids=["base"],
        updated_at=datetime(2026, 1, 1),
    )
    kb._save()

    store = HypothesisStore(knowledge, "titanic")
    known = store.create(
        observation="Rare classes weak",
        reason="Imbalance",
        prediction="Focal loss helps",
        confidence=0.7,
        tags=["loss"],
    )
    novel = store.create(
        observation="New idea",
        reason="Guess",
        prediction="Totally novel trick",
        confidence=0.7,
        tags=["never-seen-before"],
    )

    ranked = rank_candidates("titanic", runs, knowledge)
    assert len(ranked) == 2
    assert ranked[0].hypothesis.id == known.id
    assert ranked[0].expected_gain > ranked[1].expected_gain
    assert ranked[0].risk < ranked[1].risk  # KB bonus lowers risk for known-good
    assert ranked[1].hypothesis.id == novel.id
    assert ranked[1].novelty >= ranked[0].novelty


def test_rank_weights_change_order(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    runs = tmp_path / "runs"
    run_dir = runs / "base"
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id="base",
        competition="titanic",
        status=StageStatus.COMPLETED,
        stages=[],
        metadata={},
    )
    save_manifest(run_dir, manifest)
    # Prior experiment already tried "loss"-like params so novelty is lower for that tag.
    (run_dir / "training_overrides.json").write_text(
        '{"model_params": {"learning_rate": 0.05}, "feature_recipes": ["loss"]}'
    )

    store = HypothesisStore(knowledge, "titanic")
    high_gain = store.create(
        observation="a",
        reason="b",
        prediction="high expected gain",
        confidence=0.9,
        tags=["loss"],
    )
    high_novelty = store.create(
        observation="c",
        reason="d",
        prediction="novel",
        confidence=0.9,  # same confidence — isolate novelty vs gain
        tags=["brand-new-technique"],
    )
    kb = KnowledgeBase(knowledge, "titanic")
    kb._entries[("loss", "cv_accuracy")] = KnowledgeEntry(
        technique="loss",
        metric_key="cv_accuracy",
        effect=KnowledgeEffect.IMPROVES,
        delta_estimate=0.1,
        confidence=0.95,
        sample_size=2,
        evidence_run_ids=["base"],
        updated_at=datetime(2026, 1, 1),
    )
    kb._save()

    gain_first = rank_candidates(
        "titanic",
        runs,
        knowledge,
        weights=RankingWeights(expected_gain=5.0, novelty=0.0, risk=0.0, gpu_cost=0.0),
    )
    assert gain_first[0].hypothesis.id == high_gain.id

    novelty_first = rank_candidates(
        "titanic",
        runs,
        knowledge,
        weights=RankingWeights(expected_gain=0.0, novelty=5.0, risk=0.0, gpu_cost=0.0),
    )
    assert novelty_first[0].hypothesis.id == high_novelty.id
    assert novelty_first[0].novelty > novelty_first[1].novelty


def test_rank_cli_empty_backlog(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "rank",
            "--competition",
            "titanic",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No proposed hypotheses" in result.output

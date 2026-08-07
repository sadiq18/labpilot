"""Unit tests for reflection critic, beliefs, pipeline (Plans 3–7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)
from labpilot.research_engine.reflection.beliefs import BeliefUpdater
from labpilot.research_engine.reflection.claims import ClaimPromoter
from labpilot.research_engine.reflection.critic import ExperimentCritic
from labpilot.research_engine.reflection.hypotheses import HypothesisEvaluator
from labpilot.research_engine.reflection.journal import JournalProjector
from labpilot.research_engine.reflection.pipeline import run_reflection


def test_critic_rule_engine_offline() -> None:
    critic = ExperimentCritic(llm_client=None)
    assessment = critic.assess(
        {
            "strength": "strong",
            "metrics": {"cv_accuracy": 0.8},
            "comparison": {"delta": 0.02, "verdict": "worth_keeping"},
            "config_summary": {"baseline_choice": {"template_name": "tabular_classification"}},
        }
    )
    assert assessment.belief_effect == "supports"
    assert assessment.hypothesis_outcome == "confirmed"
    assert assessment.generated_by == "llm"
    assert assessment.summary


def test_belief_updater_and_hypothesis_evaluator(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    store = HypothesisStore(knowledge, competition)
    hyp = store.create(
        observation="obs",
        reason="reason",
        prediction="mixup helps",
        confidence=0.6,
        created_by=HypothesisCreatedBy.MANUAL,
        generator=HypothesisGenerator.HUMAN,
        origin=HypothesisOrigin.USER,
    )
    store.update_status(hyp.id, HypothesisStatus.TESTING)
    hyp_id = hyp.id

    evidence = {
        "id": "EE-001",
        "execution_id": "E-001",
        "strength": "strong",
        "metrics": {"cv_accuracy": 0.81},
        "comparison": {"delta": 0.02},
        "config_summary": {"baseline_choice": {"template_name": "mixup"}},
    }
    critic = ExperimentCritic()
    assessment = critic.assess(evidence)
    updater = BeliefUpdater(knowledge, competition)
    try:
        result = updater.update_from_critic(assessment, evidence)
        assert result["new_confidence"] > result["prior_confidence"]
        assert result["belief_update_id"] >= 1
    finally:
        updater.close()

    evaluator = HypothesisEvaluator(knowledge, competition)
    hyp_result = evaluator.evaluate(
        assessment, hypothesis_id=hyp_id, evidence_run_id="E-001"
    )
    assert hyp_result is not None
    assert hyp_result["status"] == "confirmed"
    assert "reflection" in (store.get(hyp_id).reason or "")


def test_run_reflection_pipeline_and_journal(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metrics.json").write_text(
        json.dumps({"cv_accuracy": 0.83, "runtime_seconds": 3}),
        encoding="utf-8",
    )
    (workspace / "baseline_choice.json").write_text(
        json.dumps({"template_name": "tabular_classification", "problem_type": "tabular_classification"}),
        encoding="utf-8",
    )
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "comparison.json").write_text(
        json.dumps({"delta": 0.02, "verdict": "worth_keeping", "maximize": True}),
        encoding="utf-8",
    )

    result = run_reflection(
        knowledge,
        competition,
        workspace_path=workspace,
        execution_id="E-fixture",
        persist=True,
    )
    assert result["evidence"]["strength"] == "strong"
    assert result["belief"]["belief_id"]
    assert result["lesson"]["id"].startswith("L-")
    assert result["understanding"]["competition"] == competition

    # Force claim promotion path by raising confidence via second run.
    run_reflection(knowledge, competition, workspace_path=workspace, persist=True)
    promoter = ClaimPromoter(knowledge, competition)
    try:
        claims = promoter.promote_eligible(evidence_id=result["evidence"]["id"])
        assert isinstance(claims, list)
    finally:
        promoter.close()

    journal = JournalProjector(knowledge, competition)
    try:
        md = journal.render_markdown()
        assert "Research Journal" in md
        assert "Recommended next" in md
        data = journal.build()
        assert "evidence" in data and "recommended_next" in data
    finally:
        journal.close()

"""Tests for execution outcome recording and submit-learn."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from labpilot.accessor.kaggle.client import SubmissionResult
from labpilot.config import KaggleConfig
from labpilot.research_engine.execution.outcome import (
    experiment_artifact_id,
    package_execution_submission,
    record_successful_execution,
    submission_csv_path,
)
from labpilot.research_engine.execution.submit_learn import (
    SubmitLearnError,
    _detect_overfitting,
    resolve_submission_csv,
    submit_and_learn,
)
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import ResearchArtifactType
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore


def _seed_plan(knowledge: Path, competition: str = "demo") -> ResearchPlan:
    store = PlanStore(knowledge, competition)
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="",
        goal="test",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=TaskType.PREPARE_WORKSPACE,
                description="a",
                order=0,
            ),
        ],
        created_at=now,
        updated_at=now,
        metadata={"plan_kind": "baseline", "tags": ["lightgbm"]},
    )
    store.upsert_plan(plan)
    store.close()
    return plan


def test_package_submission_uses_execution_id(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "predictions.csv").write_text("id,pred\n1,0.5\n", encoding="utf-8")
    path = package_execution_submission(ws, "E-001")
    assert path.name == "submission_E-001.csv"
    assert path.is_file()
    assert (ws / "artifacts" / "submission.csv").is_file()


def test_resolve_submission_csv_by_execution(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    arts = ws / "artifacts"
    arts.mkdir(parents=True)
    (arts / "submission_E-001.csv").write_text("id,p\n0,1\n", encoding="utf-8")
    (arts / "submission_E-002.csv").write_text("id,p\n0,2\n", encoding="utf-8")
    resolved = resolve_submission_csv(ws, "E-002")
    assert resolved.name == "submission_E-002.csv"
    with pytest.raises(SubmitLearnError):
        resolve_submission_csv(ws, "E-999")


def test_detect_overfitting() -> None:
    assert _detect_overfitting(
        local_score=0.9,
        learning_gain=0.05,
        public_score=0.7,
        prior_public=0.8,
    )
    assert not _detect_overfitting(
        local_score=0.9,
        learning_gain=0.05,
        public_score=0.92,
        prior_public=0.8,
    )


def test_record_successful_execution_writes_artifact(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    plan = _seed_plan(knowledge, competition)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "metrics.json").write_text(
        json.dumps(
            {
                "cv_score": 0.81,
                "train_score": 0.9,
                "val_score": 0.8,
            }
        ),
        encoding="utf-8",
    )
    (ws / "comparison.json").write_text(
        json.dumps({"cv_delta": 0.02}),
        encoding="utf-8",
    )
    (ws / "predictions.csv").write_text("id,pred\n1,0.2\n", encoding="utf-8")

    hyp = HypothesisStore(knowledge, competition).create(
        observation="test",
        reason="r",
        prediction="p",
        confidence=0.5,
        tags=["lightgbm"],
    )
    proposed = HypothesisStore(knowledge, competition).create(
        observation="try target encoding on categoricals",
        reason="open idea",
        prediction="Target encoding improves CV vs baseline mean encoding",
        confidence=0.55,
        tags=["target_encoding"],
    )
    # Attach hyp to plan
    store = PlanStore(knowledge, competition)
    plan.hypothesis_id = hyp.id
    store.upsert_plan(plan)
    store.close()

    exec_store = ExecutionStore(knowledge, competition)
    execution = exec_store.create_execution("P-001", workspace_path=str(ws))
    exec_store.update_status(execution.id, "running")
    exec_store.update_status(execution.id, "succeeded")
    execution = exec_store.get_execution(execution.id)
    assert execution is not None
    exec_store.close()

    plan = PlanStore(knowledge, competition).get_plan("P-001")
    assert plan is not None

    summary = record_successful_execution(
        knowledge_dir=knowledge,
        competition=competition,
        execution=execution,
        plan=plan,
        workspace_root=ws,
        llm_client=None,
    )
    assert summary.execution_id == execution.id
    assert summary.learning_gain == pytest.approx(0.02)
    assert summary.train_vs_validation.get("train_score") == 0.9
    assert submission_csv_path(ws, execution.id).is_file()
    assert (ws / "artifacts" / "execution_outcome.json").is_file()
    # No generic follow-up mint without a worth-trying improvement signal.
    assert summary.follow_up_hypothesis_id is None

    with KnowledgeStore(knowledge, competition) as ks:
        art = ks.get_artifact(experiment_artifact_id(execution.id))
        assert art is not None
        assert art.type == ResearchArtifactType.EXPERIMENT
        assert art.metadata.get("learning_gain") == pytest.approx(0.02)

    updated = HypothesisStore(knowledge, competition).get(hyp.id)
    assert updated is not None
    assert updated.actual_outcome is not None
    assert "0.81" in updated.actual_outcome

    notified = HypothesisStore(knowledge, competition).get(proposed.id)
    assert notified is not None
    assert f"[experiment {execution.id}]" in (notified.reason or "")


def test_submit_and_learn_patches_lb_and_overfit(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    plan = _seed_plan(knowledge, competition)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "metrics.json").write_text(
        json.dumps({"cv_score": 0.9, "train_score": 0.95, "val_score": 0.9}),
        encoding="utf-8",
    )
    (ws / "comparison.json").write_text(json.dumps({"cv_delta": 0.05}), encoding="utf-8")
    (ws / "predictions.csv").write_text("id,pred\n1,0.2\n", encoding="utf-8")

    hyp = HypothesisStore(knowledge, competition).create(
        observation="local strong",
        reason="r",
        prediction="p",
        confidence=0.6,
        tags=["lightgbm"],
    )
    store = PlanStore(knowledge, competition)
    plan.hypothesis_id = hyp.id
    store.upsert_plan(plan)
    store.close()

    exec_store = ExecutionStore(knowledge, competition)
    execution = exec_store.create_execution("P-001", workspace_path=str(ws))
    exec_store.update_status(execution.id, "running")
    exec_store.update_status(execution.id, "succeeded")
    execution = exec_store.get_execution(execution.id)
    assert execution is not None
    exec_store.close()

    plan = PlanStore(knowledge, competition).get_plan("P-001")
    assert plan is not None
    record_successful_execution(
        knowledge_dir=knowledge,
        competition=competition,
        execution=execution,
        plan=plan,
        workspace_root=ws,
        llm_client=None,
    )

    mock_client = MagicMock()
    mock_client.count_todays_submissions.return_value = 0
    mock_client.fetch_competition_metadata.return_value = MagicMock(
        max_daily_submissions=5
    )
    mock_client.upload_submission.return_value = SubmissionResult(
        competition=competition,
        submission_path=str(submission_csv_path(ws, execution.id)),
        status="scored",
        public_score=0.7,
        message="labpilot",
        submissions_url="https://www.kaggle.com/c/demo/submissions",
    )

    # Seed a prior public score so delta is negative → overfit.
    other = HypothesisStore(knowledge, competition).create(
        observation="prior",
        reason="r",
        prediction="p",
        confidence=0.5,
    )
    HypothesisStore(knowledge, competition).update_outcome(other.id, public_score=0.85)

    summary = submit_and_learn(
        knowledge_dir=knowledge,
        competition=competition,
        execution_id=execution.id,
        workspace_root=ws,
        kaggle_config=KaggleConfig(),
        client=mock_client,
    )
    assert summary.leaderboard is not None
    assert summary.leaderboard.public_score == pytest.approx(0.7)
    assert summary.leaderboard.overfitting is True
    # Overfit → actionable improvement fork (not a baseline clone).
    assert summary.follow_up_hypothesis_id is not None
    minted = HypothesisStore(knowledge, competition).get(summary.follow_up_hypothesis_id)
    assert minted is not None
    assert minted.expected_impact > 0
    assert "overfitting" in minted.tags or "generalization" in minted.tags
    assert "baseline" not in minted.prediction.lower() or "regular" in minted.prediction.lower()

    updated = HypothesisStore(knowledge, competition).get(hyp.id)
    assert updated is not None
    assert updated.public_score == pytest.approx(0.7)
    assert updated.actual_outcome is not None
    assert "public_score" in updated.actual_outcome

    with KnowledgeStore(knowledge, competition) as ks:
        beliefs = ks.list_beliefs()
        assert any("overfit" in b["id"] or b.get("effect") == "negative" for b in beliefs)
        art = ks.get_artifact(experiment_artifact_id(execution.id))
        assert art is not None
        assert art.metadata["leaderboard"]["public_score"] == pytest.approx(0.7)

    mock_client.upload_submission.assert_called_once()


def test_aligned_submit_does_not_mint_useless_follow_up(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    plan = _seed_plan(knowledge, competition)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "metrics.json").write_text(json.dumps({"cv_score": 0.8}), encoding="utf-8")
    (ws / "comparison.json").write_text(json.dumps({"cv_delta": 0.01}), encoding="utf-8")
    (ws / "predictions.csv").write_text("id,pred\n1,0.2\n", encoding="utf-8")

    hyp = HypothesisStore(knowledge, competition).create(
        observation="ok",
        reason="r",
        prediction="p",
        confidence=0.5,
        tags=["lightgbm"],
    )
    store = PlanStore(knowledge, competition)
    plan.hypothesis_id = hyp.id
    store.upsert_plan(plan)
    store.close()

    exec_store = ExecutionStore(knowledge, competition)
    execution = exec_store.create_execution("P-001", workspace_path=str(ws))
    exec_store.update_status(execution.id, "running")
    exec_store.update_status(execution.id, "succeeded")
    execution = exec_store.get_execution(execution.id)
    assert execution is not None
    exec_store.close()

    plan = PlanStore(knowledge, competition).get_plan("P-001")
    assert plan is not None
    record_successful_execution(
        knowledge_dir=knowledge,
        competition=competition,
        execution=execution,
        plan=plan,
        workspace_root=ws,
        llm_client=None,
    )

    mock_client = MagicMock()
    mock_client.count_todays_submissions.return_value = 0
    mock_client.fetch_competition_metadata.return_value = MagicMock(
        max_daily_submissions=5
    )
    mock_client.upload_submission.return_value = SubmissionResult(
        competition=competition,
        submission_path=str(submission_csv_path(ws, execution.id)),
        status="scored",
        public_score=0.82,
        message="labpilot",
        submissions_url="https://www.kaggle.com/c/demo/submissions",
    )

    before = {h.id for h in HypothesisStore(knowledge, competition).list()}
    summary = submit_and_learn(
        knowledge_dir=knowledge,
        competition=competition,
        execution_id=execution.id,
        workspace_root=ws,
        kaggle_config=KaggleConfig(),
        client=mock_client,
    )
    assert summary.leaderboard is not None
    assert summary.leaderboard.overfitting is not True
    assert summary.follow_up_hypothesis_id is None
    after = {h.id for h in HypothesisStore(knowledge, competition).list()}
    assert after == before

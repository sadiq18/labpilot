"""Plan 4 — Planning Engine Micro Agent tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.experiments.models import Hypothesis
from labpilot.research_engine.planner import compile_research_plan, validate_plan
from labpilot.research_engine.planner.micro_agents.planning_engine import ResearchPlannerAgent
from labpilot.research_engine.planner.schemas.draft import DraftTask, ResearchPlanDraft
from labpilot.research_engine.planner.schemas.task_types import TaskType


def _hypothesis(tags: list[str] | None = None) -> Hypothesis:
    now = datetime.now(UTC)
    return Hypothesis(
        id="H-001",
        competition="demo",
        observation="No SpecAugment in the pipeline",
        reason="augmentation often helps audio",
        prediction="Add SpecAugment to improve CV",
        confidence=0.6,
        expected_impact=0.01,
        tags=tags or ["augmentation", "specaugment"],
        created_at=now,
        updated_at=now,
    )


class _MockLLM:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


def _valid_slim_draft() -> dict[str, Any]:
    return {
        "goal": "Add SpecAugment with a short validation loop",
        "current_state": "No SpecAugment",
        "expected_outcome": "CV improves vs baseline",
        "risk": "May overfit rare classes",
        "success_criteria": ["Smoke train succeeds", "1-epoch CV improves"],
        "rollback": "Revert augmentation.py",
        "artifacts": ["report.md"],
        "tasks": [
            {
                "key": "read",
                "type": "read_code",
                "description": "Inspect augmentation",
                "inputs": ["augmentation.py"],
                "outputs": ["notes"],
                "depends_on": [],
            },
            {
                "key": "write",
                "type": "write_code",
                "description": "Add SpecAugment",
                "inputs": ["augmentation.py"],
                "outputs": ["augmentation.py"],
                "depends_on": ["read"],
            },
            {
                "key": "unit",
                "type": "run_unit_test",
                "description": "Unit tests",
                "inputs": ["tests/"],
                "outputs": ["unit_report"],
                "depends_on": ["write"],
            },
            {
                "key": "smoke",
                "type": "run_smoke_test",
                "description": "Smoke run",
                "inputs": [],
                "outputs": ["smoke_log"],
                "depends_on": ["unit"],
            },
            {
                "key": "train",
                "type": "run_training",
                "description": "1-epoch train",
                "inputs": ["config.yaml"],
                "outputs": ["run_dir"],
                "depends_on": ["smoke"],
            },
            {
                "key": "evaluate",
                "type": "evaluate",
                "description": "Evaluate",
                "inputs": ["run_dir"],
                "outputs": ["metrics"],
                "depends_on": ["train"],
            },
            {
                "key": "compare",
                "type": "compare",
                "description": "Compare to baseline",
                "inputs": ["metrics"],
                "outputs": ["comparison"],
                "depends_on": ["evaluate"],
            },
            {
                "key": "report",
                "type": "generate_report",
                "description": "Write report",
                "inputs": ["comparison"],
                "outputs": ["report.md"],
                "depends_on": ["compare"],
            },
        ],
    }


def test_llm_none_identical_to_rule_engine(tmp_path: Path):
    kd = tmp_path / "knowledge"
    plan = compile_research_plan(
        _hypothesis(), knowledge_dir=kd, competition="demo", llm_client=None
    )
    validate_plan(plan)
    assert plan.generated_by == "rule_engine"
    assert plan.metadata["template"] == "augmentation"


def test_mock_llm_valid_draft_sets_generated_by_llm(tmp_path: Path):
    mock = _MockLLM(_valid_slim_draft())
    plan = compile_research_plan(
        _hypothesis(),
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        llm_client=mock,
    )
    validate_plan(plan)
    assert mock.calls == 1
    assert plan.generated_by == "llm"
    assert plan.goal == "Add SpecAugment with a short validation loop"
    assert plan.metadata.get("revised_by") == "llm"
    # Verification defaults filled after LLM draft (no verification in slim schema).
    assert any(t.verification.check for t in plan.tasks)
    types = {t.type for t in plan.tasks}
    assert TaskType.WRITE_CODE in types
    assert TaskType.COMPARE in types


def test_mock_llm_garbage_soft_falls_to_template(tmp_path: Path):
    mock = _MockLLM("this is not json at all {{{")
    plan = compile_research_plan(
        _hypothesis(),
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        llm_client=mock,
    )
    validate_plan(plan)
    assert mock.calls == 1
    assert plan.generated_by == "rule_engine"
    assert plan.metadata["template"] == "augmentation"


def test_mock_llm_invalid_dag_keeps_baseline(tmp_path: Path):
    bad = _valid_slim_draft()
    # Cycle: read ↔ write
    bad["tasks"][0]["depends_on"] = ["write"]
    bad["tasks"][1]["depends_on"] = ["read"]
    mock = _MockLLM(bad)
    plan = compile_research_plan(
        _hypothesis(),
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        llm_client=mock,
    )
    validate_plan(plan)
    assert mock.calls == 1
    assert plan.generated_by == "rule_engine"
    assert any("LLM revision rejected" in note for note in plan.notes)


def test_agent_rule_engine_returns_baseline():
    baseline = ResearchPlanDraft(
        goal="g",
        tasks=[
            DraftTask(key="read", type=TaskType.READ_CODE, description="look"),
        ],
    )
    agent = ResearchPlannerAgent(llm_client=None)
    from labpilot.accessor.common.micro_agents import StructuredContext

    result = agent.run(
        StructuredContext(
            competition="demo",
            question="g",
            data={"baseline_draft": baseline.model_dump(mode="json"), "goal": "g"},
        )
    )
    assert isinstance(result, ResearchPlanDraft)
    assert result.goal == "g"
    assert agent.last_used_llm is False
    assert len(result.tasks) == 1


def test_exactly_one_llm_call_per_compile(tmp_path: Path):
    mock = _MockLLM(_valid_slim_draft())
    compile_research_plan(
        _hypothesis(),
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        llm_client=mock,
    )
    assert mock.calls == 1

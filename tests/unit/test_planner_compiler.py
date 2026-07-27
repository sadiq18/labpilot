from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.shared.experiments.models import Hypothesis
from labpilot.research_engine.planner import compile_research_plan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.serializer import render_markdown
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import topological_levels, validate_plan


def _hypothesis(tags: list[str], prediction: str = "Improve CV") -> Hypothesis:
    now = datetime.now(UTC)
    return Hypothesis(
        id="H-001",
        competition="demo",
        observation="baseline observation",
        reason="because",
        prediction=prediction,
        confidence=0.6,
        expected_impact=0.01,
        tags=tags,
        created_at=now,
        updated_at=now,
    )


def test_augmentation_hypothesis_uses_augmentation_template(tmp_path: Path):
    kd = tmp_path / "knowledge"
    plan = compile_research_plan(
        _hypothesis(["augmentation", "specaugment"]),
        knowledge_dir=kd,
        competition="demo",
        llm_client=None,
    )
    validate_plan(plan)
    assert plan.status == PlanStatus.READY
    assert plan.generated_by == "rule_engine"
    assert plan.metadata["template"] == "augmentation"
    types = {t.type for t in plan.tasks}
    assert {
        TaskType.READ_CODE,
        TaskType.WRITE_CODE,
        TaskType.MODIFY_CONFIG,
        TaskType.RUN_TRAINING,
        TaskType.EVALUATE,
        TaskType.COMPARE,
    } <= types


def test_generic_hypothesis_falls_back_and_validates(tmp_path: Path):
    kd = tmp_path / "knowledge"
    plan = compile_research_plan(
        _hypothesis(["scheduler"], prediction="Try a cosine schedule"),
        knowledge_dir=kd,
        competition="demo",
    )
    validate_plan(plan)
    assert plan.metadata["template"] == "generic"
    # Topological levels are consistent with declared edges.
    levels = topological_levels(plan)
    assert levels[0] == [plan.tasks[0].id]


def test_projections_written_and_no_runs_dir(tmp_path: Path):
    kd = tmp_path / "knowledge"
    plan = compile_research_plan(
        _hypothesis(["augmentation"]), knowledge_dir=kd, competition="demo"
    )
    plans_dir = kd / "demo" / "research" / "plans"
    assert (plans_dir / f"{plan.id}.json").is_file()
    assert (plans_dir / f"{plan.id}.md").is_file()
    assert not (kd / "demo" / "research" / "runs").exists()


def test_persisted_plan_is_readable(tmp_path: Path):
    kd = tmp_path / "knowledge"
    plan = compile_research_plan(
        _hypothesis(["augmentation"]), knowledge_dir=kd, competition="demo"
    )
    store = PlanStore(kd, "demo")
    try:
        got = store.get_plan(plan.id)
        assert got is not None
        assert len(got.tasks) == len(plan.tasks)
    finally:
        store.close()


def test_markdown_is_derived_from_model(tmp_path: Path):
    # Markdown must be generated from model fields (goal + task ids), never
    # authored as the primary artifact.
    compiled = compile_research_plan(
        _hypothesis(["augmentation"], prediction="Add SpecAugment"),
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
    )
    md = render_markdown(compiled)
    assert compiled.goal in md
    assert compiled.tasks[0].id in md
    assert f"# Research Plan {compiled.id}" in md

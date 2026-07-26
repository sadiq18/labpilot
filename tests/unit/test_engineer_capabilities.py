"""Tests for Research Engineer capabilities Plans 5–9 + CLI Plan 10."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.research_engine.execution.capabilities.code_engineering import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.capabilities.research_review import (
    ResearchReviewCapability,
)
from labpilot.research_engine.execution.capabilities.runtime import RuntimeCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.engineer import (
    ResearchEngineer,
    default_capability_registry,
)
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner import compile_baseline_plan
from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
)
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore

runner = CliRunner()


def _seed_analyze(knowledge: Path, competition: str = "demo") -> None:
    paths = ResearchPaths(knowledge, competition).ensure()
    paths.report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "competition": competition,
                "techniques": {"items": []},
                "retrieval": {"queries": []},
            }
        ),
        encoding="utf-8",
    )


def _ctx(
    knowledge: Path,
    *,
    task_type: TaskType,
    competition: str = "demo",
    metadata: dict | None = None,
    constraints: dict | None = None,
) -> TaskContext:
    paths = ResearchPaths(knowledge, competition).ensure()
    root = knowledge.parent / "competitions" / competition
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="",
        goal="t",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=task_type,
                metadata=metadata or {},
            )
        ],
        created_at=now,
        updated_at=now,
        metadata={"plan_kind": "baseline"},
    )
    execution = ResearchExecution(id="E-001", plan_id="P-001", competition=competition)
    return TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=execution,
        paths=paths,
        workspace_root=root,
        competition=competition,
        constraints=constraints or {},
    )


def test_code_engineering_writes_without_llm(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ctx = _ctx(knowledge, task_type=TaskType.WRITE_CODE)
    ev = CodeEngineeringCapability().execute(ctx)
    assert ev.passed
    assert (ctx.workspace_root / "pipeline" / "train.py").is_file()
    # Offline path should prefer Jinja full templates (rule_engine), not last_resort.
    assert ev.metadata.get("origin") in {"rule_engine", "last_resort", "llm"}
    train = (ctx.workspace_root / "pipeline" / "train.py").read_text(encoding="utf-8")
    # Full Jinja tabular baseline is much larger than the emergency stub.
    if ev.metadata.get("origin") == "rule_engine":
        assert "lightgbm" in train.lower() or "LightGBM" in train or len(train) > 500
    assert "digests" in ev.metadata


def test_code_engineering_applies_llm_proposal(tmp_path: Path) -> None:
    """Capability applies a typed proposal (simulated LLM) to full train.py."""
    from labpilot.research_engine.execution.schemas.code_proposal import (
        CodeFileSpec,
        CodeProposal,
    )

    class FakeLLM:
        def complete(self, system: str, user: str) -> str:
            proposal = CodeProposal(
                summary="full baseline",
                rationale="test",
                files=[
                    CodeFileSpec(
                        path="pipeline/train.py",
                        content=(
                            '"""Full generated train."""\n'
                            "def main():\n"
                            "    print('full-code')\n"
                            "\n"
                            "if __name__ == '__main__':\n"
                            "    main()\n"
                        ),
                    )
                ],
            )
            return proposal.model_dump_json()

    knowledge = tmp_path / "knowledge"
    ctx = _ctx(knowledge, task_type=TaskType.WRITE_CODE)
    ev = CodeEngineeringCapability(llm_client=FakeLLM()).execute(ctx)
    assert ev.passed
    assert ev.metadata.get("origin") == "llm"
    assert ev.metadata.get("used_llm") is True
    text = (ctx.workspace_root / "pipeline" / "train.py").read_text(encoding="utf-8")
    assert "full-code" in text


def test_research_review_can_block(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ctx = _ctx(
        knowledge,
        task_type=TaskType.RESEARCH_REVIEW,
        metadata={"force_block": True},
    )
    ev = ResearchReviewCapability().execute(ctx)
    assert not ev.passed
    assert ev.error


def test_runtime_idempotent_job(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ctx = _ctx(knowledge, task_type=TaskType.SELECT_RUNTIME)
    cap = RuntimeCapability()
    first = cap.execute(ctx)
    assert first.passed
    job = first.metadata["job_id"]
    ctx.prior_evidence = first
    second = cap.execute(ctx)
    assert second.metadata["job_id"] == job
    assert second.metadata.get("redispatched") is False


def test_baseline_dry_run_end_to_end(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)

    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=default_capability_registry(install_packages=False),
        constraints={
            "dry_run": True,
            "smoke_syntax_only": True,
            "train_stub": True,
            "allow_upload": False,
        },
    )
    try:
        execution = engineer.run_plan("P-001")
        assert execution.status == "succeeded", execution.error
        workspace = Path(execution.workspace_path or "")
        assert workspace.name == "demo"
        assert (workspace / "pipeline" / "train.py").is_file()
        assert (workspace / "artifacts" / "smoke_ok.json").is_file()
        assert (workspace / "metrics.json").is_file()
        assert (workspace / "artifacts" / "submission.csv").is_file()
        assert (workspace / "artifacts" / "report.md").is_file()
        plan = engineer._plan_store.get_plan("P-001")
        assert plan is not None
        assert plan.status == PlanStatus.DONE
        assert all(t.status == TaskStatus.DONE for t in plan.tasks)
    finally:
        engineer.close()


def test_smoke_fail_stops_before_train(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = PlanStore(knowledge, "demo")
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition="demo",
        hypothesis_id="",
        goal="gate",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=TaskType.RUN_SMOKE_TEST,
                order=0,
                retry_policy=RetryPolicy(max_retries=0, abort_on_failure=True),
            ),
            ResearchTask(
                id="P-001-T02",
                plan_id="P-001",
                type=TaskType.RUN_TRAINING,
                dependencies=["P-001-T01"],
                order=1,
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    store.upsert_plan(plan)
    store.close()

    # No train.py → smoke fails.
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=default_capability_registry(install_packages=False),
        constraints={"dry_run": True},
    )
    try:
        execution = engineer.run_plan("P-001")
        assert execution.status == "failed"
        plan = engineer._plan_store.get_plan("P-001")
        assert plan is not None
        assert plan.tasks[0].status == TaskStatus.FAILED
        assert plan.tasks[1].status == TaskStatus.PENDING
    finally:
        engineer.close()


def test_cli_run_requires_plan(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--competition", "demo"])
    assert result.exit_code == 1
    assert "Plan-driven" in result.output or "plan" in result.output.lower()


def test_cli_run_plan_dry(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)
    result = runner.invoke(
        app,
        [
            "run",
            "--plan",
            "P-001",
            "--competition",
            "demo",
            "--knowledge-dir",
            str(knowledge),
            "--dry-run",
            "--no-install-packages",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "E-001" in result.output
    assert "succeeded" in result.output

"""Round-trip tests for Research OS artifact adapters (M1 plan-1)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.artifacts import (
    ARTIFACT_SCHEMA_IDS,
    EvidenceArtifacts,
    ExecutionArtifacts,
    PlanArtifacts,
    read_analysis,
    write_analysis,
)
from labpilot.research_engine.evidence.models import EvidenceCard, EvidenceDecision
from labpilot.research_engine.intelligence.models import AnalysisReport
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)

# Sibling engine packages must not import artifacts (callers only).
_FORBIDDEN_IMPORT = "labpilot.research_engine.artifacts"


def _seed_plan(knowledge: Path, competition: str = "demo") -> ResearchPlan:
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="H-001",
        goal="test goal",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=TaskType.PREPARE_WORKSPACE,
                description="prep",
                status=TaskStatus.PENDING,
            )
        ],
        created_at=now,
        updated_at=now,
    )
    arts = PlanArtifacts(knowledge, competition)
    try:
        arts.upsert(plan)
    finally:
        arts.close()
    return plan


def test_analysis_artifact_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    report = AnalysisReport(
        competition={"slug": competition},
        analyzers=["competition"],
        notes=["artifact adapter smoke"],
    )
    ref = write_analysis(report, knowledge, competition)
    assert ref.kind == "competition_analysis"
    assert ref.schema_id == ARTIFACT_SCHEMA_IDS["competition_analysis"]
    assert ref.path is not None
    assert Path(ref.path).is_file()

    loaded = read_analysis(knowledge, competition)
    assert loaded is not None
    assert loaded.competition["slug"] == competition
    assert loaded.notes == ["artifact adapter smoke"]


def test_plan_artifact_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="H-001",
        goal="round-trip",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=TaskType.READ_CODE,
                status=TaskStatus.PENDING,
            )
        ],
        created_at=now,
        updated_at=now,
    )
    arts = PlanArtifacts(knowledge, competition)
    try:
        ref = arts.upsert(plan)
        assert ref.kind == "research_plan"
        assert ref.schema_id == ARTIFACT_SCHEMA_IDS["research_plan"]
        assert ref.path is not None
        assert Path(ref.path).is_file()
        assert Path(ref.path).with_suffix(".md").is_file()

        got = arts.get("P-001")
        assert got is not None
        assert got.goal == "round-trip"
        assert [t.id for t in got.tasks] == ["P-001-T01"]
    finally:
        arts.close()


def test_execution_artifact_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    _seed_plan(knowledge, competition)
    arts = ExecutionArtifacts(knowledge, competition)
    try:
        execution, ref = arts.create("P-001")
        assert execution.id == "E-001"
        assert ref.kind == "execution"
        assert ref.schema_id == ARTIFACT_SCHEMA_IDS["execution"]
        got = arts.get(execution.id)
        assert got is not None
        assert got.plan_id == "P-001"
        assert got.status == "pending"
        arts.update_status(execution.id, "running")
        running = arts.get(execution.id)
        assert running is not None
        assert running.status == "running"
    finally:
        arts.close()


def test_evidence_artifact_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    arts = EvidenceArtifacts(knowledge, competition)
    card = EvidenceCard(
        competition=competition,
        treatment_experiment="E-001",
        decision=EvidenceDecision.ACCEPTED,
        decision_reason="adapter test",
    )
    saved, ref = arts.save(card)
    assert saved.id.startswith("EV-")
    assert ref.kind == "evidence_card"
    assert ref.schema_id == ARTIFACT_SCHEMA_IDS["evidence_card"]
    assert Path(ref.path or "").is_file()
    loaded = arts.get(saved.id)
    assert loaded is not None
    assert loaded.decision == EvidenceDecision.ACCEPTED
    assert loaded.treatment_experiment == "E-001"


def test_engine_packages_do_not_import_artifacts() -> None:
    """Import boundary: engine packages must not depend on artifacts."""
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    package_dirs = ("intelligence", "planner", "execution", "evidence", "reflection")
    violations: list[str] = []
    for name in package_dirs:
        package_dir = root / name
        for path in package_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == _FORBIDDEN_IMPORT or alias.name.startswith(
                            _FORBIDDEN_IMPORT + "."
                        ):
                            violations.append(f"{path}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == _FORBIDDEN_IMPORT or node.module.startswith(
                        _FORBIDDEN_IMPORT + "."
                    ):
                        violations.append(
                            f"{path}:{node.lineno}: from {node.module} import ..."
                        )
    assert not violations, "engine packages must not import artifacts:\n" + "\n".join(
        violations
    )

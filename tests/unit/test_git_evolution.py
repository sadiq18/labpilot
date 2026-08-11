"""Tests for GitTool (GitPython) and experiment git orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.agents import (
    AgentTask,
    build_default_specialist_registry,
    execute_agent_sync,
    find_experiment_record,
    revert_to_commit,
)
from labpilot.research_engine.agents.git_evolution import snapshot_before_experiment
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.git import GitPythonTool, open_git_tool
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _seed_plan(knowledge: Path, competition: str) -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="mini",
            status=PlanStatus.READY,
            tasks=[
                ResearchTask(
                    id="P-001-T01",
                    plan_id="P-001",
                    type=TaskType.WRITE_CODE,
                    description="code",
                    order=0,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        store.upsert_plan(plan)
        return plan.id
    finally:
        store.close()


def test_git_tool_branch_commit_structured(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "git-tool", "git-tool")
    tool = open_git_tool(client.root)
    assert isinstance(tool, GitPythonTool)
    train = client.root / "pipeline" / "train.py"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("v1 = 1\n", encoding="utf-8")

    branch = tool.create_branch("research/S-001/E-042", checkout=True)
    assert branch.name == "research/S-001/E-042"
    snap = tool.commit("experiment: baseline + specaugment")
    assert snap is not None
    assert snap.commit
    assert snap.short
    assert "pipeline/train.py" in snap.files_changed or snap.files_changed
    assert snap.message.startswith("experiment:")

    train.write_text("v2 = 2\n", encoding="utf-8")
    tool.checkout(snap.commit, paths=["pipeline"])
    assert train.read_text(encoding="utf-8") == "v1 = 1\n"


def test_git_tool_code_commit_skips_knowledge(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "code-only", "code-only")
    tool = open_git_tool(client.root)
    (client.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (client.root / "pipeline" / "train.py").write_text("x=1\n", encoding="utf-8")
    secret = client.root / "knowledge" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("do-not-commit\n", encoding="utf-8")

    snap = tool.commit("experiment: code only")
    assert snap is not None
    tracked = tool.execute("ls-files", "knowledge/secret.txt")
    assert tracked.strip() == ""


def test_snapshot_and_experiment_record_via_git_tool(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "exp-git", "exp-git")
    ws = Workspace.from_client(client).ensure_roots()
    plan_id = _seed_plan(ws.knowledge_dir, ws.competition)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("pass\n", encoding="utf-8")

    snap = snapshot_before_experiment(
        ws.root,
        session_id="S-009",
        experiment_key="E-042",
        message="experiment: baseline + smoke",
    )
    assert snap is not None
    assert snap.branch == "research/S-009/E-042"

    registry = build_default_specialist_registry(
        dry_run_default=True, install_subscribers=False
    )
    execute_agent_sync(
        registry.require("experiment").agent,
        AgentTask(
            id="T-git",
            capability="run_experiment",
            description="baseline + smoke",
            metadata={
                "plan_id": plan_id,
                "dry_run": True,
                "session_id": "S-009",
                "execution_id": "E-042",
            },
        ),
        ws,
        ContextBundle(request=ContextRequest(competition=ws.competition, goal="g")),
    )
    record = find_experiment_record(ws.effective_runs_dir, "E-042")
    assert record is not None
    assert record.get("git_commit")
    assert record.get("git_branch") == "research/S-009/E-042"
    assert "files_changed" in record


def test_git_tool_execute_fallback(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "exec", "exec")
    tool = open_git_tool(client.root)
    out = tool.execute("rev-parse", "--is-inside-work-tree")
    assert "true" in out

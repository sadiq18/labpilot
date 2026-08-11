"""Drive `ExperimentSpecialist.execute` without git branching or a real run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from labpilot.research_engine.agents import experiment as experiment_mod
from labpilot.research_engine.agents.models import AgentTask
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace

COMPETITION = "demo"


def experiment_workspace(tmp_path: Path) -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", COMPETITION, code_root=tmp_path / "ws"
    ).ensure_roots()


def bundle() -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition=COMPETITION, goal="test"))


def training_task(**metadata: Any) -> AgentTask:
    return AgentTask(
        id="T-1", capability="run_training", description="train", metadata=metadata
    )


def stub_git_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop `execute` branching and committing. Install once per test."""

    def _no_snapshot(
        workspace_root: Path, *, session_id: str, experiment_key: str, message: str
    ) -> None:
        # Signature-faithful: `lambda *a, **k` would swallow a renamed kwarg.
        del workspace_root, session_id, experiment_key, message

    monkeypatch.setattr(experiment_mod, "snapshot_before_experiment", _no_snapshot)


def stub_run_plan(monkeypatch: pytest.MonkeyPatch, **run_plan_data: Any) -> None:
    """Report `run_plan_data` as the run's outcome. Safe to call repeatedly."""

    def _fake_run_plan(*_a: object, **_k: object) -> ToolResult:
        return ToolResult(data=dict(run_plan_data), refs=[])

    monkeypatch.setattr("labpilot.research_engine.tools.handlers.run.run_plan", _fake_run_plan)


def stub_experiment_io(monkeypatch: pytest.MonkeyPatch, **run_plan_data: Any) -> None:
    """Both stubs at once, for a test that sets its outcome only once."""
    stub_git_snapshot(monkeypatch)
    stub_run_plan(monkeypatch, **run_plan_data)

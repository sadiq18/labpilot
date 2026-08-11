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


def workspace(tmp_path: Path) -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", COMPETITION, code_root=tmp_path / "ws"
    ).ensure_roots()


def bundle() -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition=COMPETITION, goal="test"))


def training_task() -> AgentTask:
    return AgentTask(id="T-1", capability="run_training", description="train")


def stub_experiment_io(monkeypatch: pytest.MonkeyPatch, **run_plan_data: Any) -> None:
    """Replace the git snapshot and the plan runner with the given outcome."""

    def _no_snapshot(
        workspace_root: Path, *, session_id: str, experiment_key: str, message: str
    ) -> None:
        # Signature-faithful: `lambda *a, **k` would swallow a renamed kwarg.
        del workspace_root, session_id, experiment_key, message

    def _fake_run_plan(*_a: object, **_k: object) -> ToolResult:
        return ToolResult(data=dict(run_plan_data), refs=[])

    monkeypatch.setattr(experiment_mod, "snapshot_before_experiment", _no_snapshot)
    monkeypatch.setattr("labpilot.research_engine.tools.handlers.run.run_plan", _fake_run_plan)

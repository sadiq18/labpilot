"""Specialist tool handlers — thin wrappers that invoke agents."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.agents import (
    AgentTask,
    build_default_specialist_registry,
    execute_agent_sync,
)
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def _bundle(workspace: Workspace, *, goal: str = "") -> ContextBundle:
    return ContextBundle(
        request=ContextRequest(
            competition=workspace.competition,
            goal=goal or workspace.goal or "",
            knowledge_dir=workspace.knowledge_dir,
        )
    )


def implement(
    workspace: Workspace,
    *,
    description: str = "",
    capability: str = "implement",
    force_rewrite: bool = False,
    llm_client: Any | None = None,
    **extra: Any,
) -> ToolResult:
    """Route to the Implementation specialist."""
    registry = build_default_specialist_registry(llm_client=llm_client)
    candidates = registry.candidates(capability=capability)
    if not candidates:
        raise KeyError(f"no specialist for capability: {capability}")
    meta = dict(extra)
    if force_rewrite:
        meta["force_rewrite"] = True
    task = AgentTask(
        id=str(meta.pop("task_id", "T-implement")),
        capability=capability,
        description=description,
        metadata=meta,
    )
    refs = execute_agent_sync(
        candidates[0].agent,
        task,
        workspace,
        _bundle(workspace, goal=description),
    )
    return ToolResult(
        refs=refs,
        data={
            "specialist": candidates[0].name,
            "capability": capability,
            "paths": [r.path for r in refs],
        },
    )


def run_experiment(
    workspace: Workspace,
    *,
    plan_id: str,
    dry_run: bool = True,
    description: str = "",
    llm_client: Any | None = None,
    **extra: Any,
) -> ToolResult:
    """Route to the Experiment specialist (never live-submits)."""
    registry = build_default_specialist_registry(
        llm_client=llm_client,
        dry_run_default=dry_run,
    )
    candidates = registry.candidates(capability="run_experiment")
    if not candidates:
        raise KeyError("no experiment specialist registered")
    meta = {"plan_id": plan_id, "dry_run": dry_run, **extra}
    # Hard gate: specialist path never uploads.
    meta["submit"] = False
    task = AgentTask(
        id=str(meta.pop("task_id", "T-experiment")),
        capability="run_experiment",
        description=description or f"run experiment for {plan_id}",
        metadata=meta,
    )
    refs = execute_agent_sync(
        candidates[0].agent,
        task,
        workspace,
        _bundle(workspace, goal=description),
    )
    metrics_ref = next((r for r in refs if r.kind == "metrics"), None)
    experiment_ref = next((r for r in refs if r.kind == "experiment"), None)
    return ToolResult(
        refs=refs,
        data={
            "specialist": candidates[0].name,
            "plan_id": plan_id,
            "dry_run": dry_run,
            "experiment_path": experiment_ref.path if experiment_ref else None,
            "metrics_path": metrics_ref.path if metrics_ref else None,
            "submit": False,
        },
    )

"""Specialist tool handlers — thin wrappers that invoke agents."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from labpilot.research_engine.agents.catalog import build_default_specialist_registry
from labpilot.research_engine.agents.facade import execute_agent_sync
from labpilot.research_engine.agents.models import AgentTask
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


class ImplementProducedNothingError(RuntimeError):
    """An implementation step that wrote no files.

    Sibling of `ExperimentProducedNoMetricsError`, for the same reason and with
    the same remedy: raised rather than returned, so the Conductor records a
    failed task instead of a successful one. A silent no-op that reports success
    is an invitation to repeat it.
    """


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
    if not refs:
        # "A tool that changed nothing did not succeed" — M9's own rule, and
        # this was the clearest violation of it in the system.
        #
        # Measured on rogii 2026-08-09: `implement` reported `completed` five
        # times while `pipeline/train.py` went untouched. Reporting success is
        # what made the policy keep choosing it — from its side the tool had
        # just worked — and the campaign spent all eight steps producing
        # nothing. Raised rather than returned, so the Conductor records a
        # failed task, exactly as `run_experiment` does for the same failure.
        raise ImplementProducedNothingError(
            f"{capability} produced no files. Nothing was written, so there is "
            "nothing to run or evaluate."
        )

    return ToolResult(
        refs=refs,
        data={
            "specialist": candidates[0].name,
            "capability": capability,
            "paths": [r.path for r in refs],
        },
    )


class ExperimentProducedNoMetricsError(RuntimeError):
    """A non-dry experiment finished without producing metrics.

    Raised rather than returned so the Conductor records a failed task instead
    of a successful one, which is what let a silent no-op repeat indefinitely.
    """


def _metrics_written_since(metrics_ref: Any, started_at: float) -> bool:
    """Whether the metrics artifact was written by the run that just finished.

    A missing file and a stale file are the same answer to the only question
    that matters — did this execution produce a result — but they need different
    messages, so the caller distinguishes them.
    """
    if metrics_ref is None:
        return False
    path = Path(str(getattr(metrics_ref, "path", "") or ""))
    if not path.is_file():
        return False
    # A second of slack: some filesystems round mtime, and the artifact is
    # written moments before the ref is built.
    return path.stat().st_mtime >= started_at - 1.0


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
    started_at = time.time()
    refs = execute_agent_sync(
        candidates[0].agent,
        task,
        workspace,
        _bundle(workspace, goal=description),
    )
    metrics_ref = next((r for r in refs if r.kind == "metrics"), None)
    experiment_ref = next((r for r in refs if r.kind == "experiment"), None)
    # A real experiment that produced no metrics did not happen. Reporting
    # success here is how a dry run masqueraded as training for an entire
    # campaign: the policy saw "completed", chose it again, and looped.
    #
    # Presence is not enough: `metrics.json` from an earlier successful run sits
    # at the workspace root and survives a failure, so this guard found a file
    # and passed while the execution had died at `import catboost`. Measured on
    # rogii 2026-08-07 — eight consecutive failed executions, every one recorded
    # `run_experiment completed`, and the campaign looked healthy for 20 steps.
    # Ask whether *this run* wrote it.
    if not dry_run and not _metrics_written_since(metrics_ref, started_at):
        stale = metrics_ref is not None
        raise ExperimentProducedNoMetricsError(
            f"run_experiment for {plan_id} completed without writing metrics"
            + (
                " — the metrics file on disk predates this run, so it belongs to "
                "an earlier execution"
                if stale
                else ". The plan may have been a no-op (already done, or nothing to train)."
            )
        )
    return ToolResult(
        refs=refs,
        data={
            "specialist": candidates[0].name,
            "plan_id": plan_id,
            "dry_run": dry_run,
            # The id the specialist already stamped on its own artifacts. Without
            # it the conductor has nothing to record a score against, so every
            # experiment run through this path was invisible to `metric_target`
            # and `plateau` however well it trained — measured 2026-08-20, two
            # campaigns chose this tool six times each and appended nothing to
            # the series.
            "execution_id": _execution_id_from(experiment_ref, metrics_ref),
            "experiment_path": experiment_ref.path if experiment_ref else None,
            "metrics_path": metrics_ref.path if metrics_ref else None,
            "submit": False,
        },
    )


def _execution_id_from(*refs: Any) -> str | None:
    """The execution id the specialist encoded in its ref ids.

    `experiment:E-007` / `metrics:E-007` — the agent builds both from one
    `execution_id`, so either answers. Read from the refs rather than threaded
    through as a second parameter, because the refs are what the specialist
    actually returns and a parameter could disagree with them.
    """
    for ref in refs:
        raw = str(getattr(ref, "id", "") or "")
        _, _, execution_id = raw.partition(":")
        if execution_id.strip():
            return execution_id.strip()
    return None

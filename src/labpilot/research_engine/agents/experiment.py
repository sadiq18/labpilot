"""Experiment specialist — run plans, collect metrics, emit completion hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio

from labpilot.research_engine.agents.events import (
    EXPERIMENT_COMPLETED,
    MODEL_FAILED,
    EventEmitter,
    noop_emit,
)
from labpilot.research_engine.agents.git_evolution import (
    short_commit,
    snapshot_before_experiment,
    write_experiment_git_record,
)
from labpilot.research_engine.agents.models import as_agent_task
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.workspace_facade import Workspace

_EXPERIMENT_SCHEMA = "labpilot.artifact.experiment/v1"
_METRICS_SCHEMA = "labpilot.artifact.metrics/v1"


def _load_metrics(root: Path) -> dict[str, Any]:
    path = root / "metrics.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {"value": data}


class ExperimentSpecialist:
    """Run experiments via existing run_plan path; never submits live / peers."""

    name = "experiment"
    capabilities = [
        "experiment",
        "run_experiment",
        "run_training",
        "evaluate",
        "compare",
    ]

    def __init__(
        self,
        *,
        dry_run_default: bool = True,
        on_event: EventEmitter | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._dry_run_default = dry_run_default
        self._emit = on_event or noop_emit
        self._llm = llm_client

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        del context  # reserved for ranking / evidence selection
        agent_task = as_agent_task(task)
        meta = dict(agent_task.metadata)
        plan_id = str(meta.get("plan_id") or "P-001")
        dry_run = bool(meta["dry_run"]) if "dry_run" in meta else self._dry_run_default
        # Submit stays Conductor-gated — specialist never uploads.
        submit = False

        session_id = str(meta.get("session_id") or "local")
        experiment_key = str(
            meta.get("execution_id") or meta.get("experiment_key") or agent_task.id
        )
        description = agent_task.description or plan_id
        snapshot = await anyio.to_thread.run_sync(
            lambda: snapshot_before_experiment(
                workspace.root,
                session_id=session_id,
                experiment_key=experiment_key,
                message=f"experiment: {description}",
            )
        )
        git_branch = snapshot.branch if snapshot else None
        git_commit = snapshot.commit if snapshot else None
        files_changed = list(snapshot.files_changed) if snapshot else []

        # Lazy import avoids agents ↔ tools package cycle at import time.
        from labpilot.research_engine.tools.handlers.run import run_plan

        result = await anyio.to_thread.run_sync(
            lambda: run_plan(
                workspace,
                plan_id=plan_id,
                dry_run=dry_run,
                submit=submit,
                llm_client=self._llm,
                constraints=dict(meta.get("constraints") or {}),
            )
        )

        metrics = _load_metrics(workspace.root)
        execution_id = str(result.data.get("execution_id") or f"E-agent-{agent_task.id}")
        status = str(result.data.get("status") or "unknown")
        experiment_id = f"exp_{workspace.competition}_{execution_id}"
        record_payload = {
            "experiment_id": experiment_id,
            "task_id": agent_task.id,
            "execution_id": execution_id,
            "plan_id": plan_id,
            "competition": workspace.competition,
            "status": status,
            "metrics": metrics,
            "git_commit": git_commit,
            "git_commit_short": short_commit(git_commit),
            "git_branch": git_branch,
            "files_changed": files_changed,
            "aliases": [experiment_key],
        }
        record_path = write_experiment_git_record(workspace.root, record_payload)

        refs = list(result.refs)
        refs.append(
            ArtifactRef(
                kind="experiment",
                id=f"experiment:{execution_id}",
                schema_id=_EXPERIMENT_SCHEMA,
                path=str(record_path),
                competition=workspace.competition,
            )
        )
        metrics_path = workspace.root / "metrics.json"
        if metrics_path.is_file():
            refs.append(
                ArtifactRef(
                    kind="metrics",
                    id=f"metrics:{execution_id}",
                    schema_id=_METRICS_SCHEMA,
                    path=str(metrics_path),
                    competition=workspace.competition,
                )
            )

        ref_payload = [
            {"kind": r.kind, "id": r.id, "path": r.path, "schema_id": r.schema_id}
            for r in refs
        ]
        event_payload = {
            "task_id": agent_task.id,
            "experiment_id": experiment_id,
            "execution_id": execution_id,
            "plan_id": plan_id,
            "competition": workspace.competition,
            "workspace_root": str(workspace.root),
            "metrics": metrics,
            "status": status,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "files_changed": files_changed,
            "paths": [r.path for r in refs if r.path],
            "refs": ref_payload,
        }
        if status == "failed" or result.data.get("error"):
            event_payload["error"] = result.data.get("error")
            self._emit(MODEL_FAILED, event_payload)
        self._emit(EXPERIMENT_COMPLETED, event_payload)
        return refs

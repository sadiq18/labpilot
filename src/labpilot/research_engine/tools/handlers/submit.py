"""submit / submit_learn tool handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.artifacts.submission import write_submission_record
from labpilot.research_engine.execution.outcome import package_execution_submission
from labpilot.research_engine.execution.submit_learn import submit_and_learn
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def submit(
    workspace: Workspace,
    *,
    execution_id: str,
) -> ToolResult:
    """Package ``submission_<E-id>.csv`` under the workspace artifacts root."""
    csv_path = package_execution_submission(workspace.root, execution_id)
    record, ref = write_submission_record(
        workspace.root,
        execution_id,
        {"status": "packaged", "csv_path": str(csv_path)},
        competition=workspace.competition,
        produced_by="submit",
    )
    return ToolResult(
        refs=[ref],
        data={
            "execution_id": execution_id,
            "csv_path": str(csv_path),
            "result_path": record.result_path,
        },
    )


def submit_learn(
    workspace: Workspace,
    *,
    execution_id: str,
    submission_path: Path | str | None = None,
    message: str | None = None,
    kaggle_config: Any | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> ToolResult:
    """Upload a packaged submission and apply LB learning updates."""
    summary = submit_and_learn(
        knowledge_dir=workspace.knowledge_dir,
        competition=workspace.competition,
        execution_id=execution_id,
        workspace_root=workspace.root,
        submission_path=Path(submission_path) if submission_path else None,
        message=message,
        kaggle_config=kaggle_config,
        dry_run=dry_run,
        client=client,
    )
    payload = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else {}
    _, ref = write_submission_record(
        workspace.root,
        execution_id,
        payload if isinstance(payload, dict) else {"status": "submitted"},
        competition=workspace.competition,
        produced_by="submit_learn",
    )
    lb = summary.leaderboard
    return ToolResult(
        refs=[ref],
        data={
            "execution_id": execution_id,
            "submission_path": summary.submission_path,
            "public_score": lb.public_score if lb else None,
            "follow_up_hypothesis_id": summary.follow_up_hypothesis_id,
            "dry_run": dry_run,
        },
    )

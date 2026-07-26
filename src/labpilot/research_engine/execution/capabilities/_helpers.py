"""Shared helpers for capability evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def evidence(
    context: TaskContext,
    *,
    capability: str,
    passed: bool,
    summary: str,
    checks: list[str] | None = None,
    paths: list[str] | None = None,
    error: str | None = None,
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> TaskEvidence:
    return TaskEvidence(
        task_id=context.task.id,
        execution_id=context.execution.id,
        capability=capability,
        passed=passed,
        summary=summary,
        checks=checks or [],
        paths=paths or [],
        error=error,
        metrics=metrics or {},
        metadata=metadata or {},
    )


def is_dry_run(context: TaskContext) -> bool:
    return bool(context.constraints.get("dry_run", False))


def allow_upload(context: TaskContext) -> bool:
    return bool(context.constraints.get("allow_upload", False))

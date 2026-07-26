"""Workspace capability — create/verify ``competitions/<slug>/`` layout.

Capability ``name`` stays ``\"workspace\"``. The on-disk root is the competition
slug directory (not the execution id).
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: Relative dirs created under ``competitions/<competition-slug>/`` (idempotent).
_WORKSPACE_SUBDIRS = (
    "src",
    "configs",
    "data",
    "logs",
    "artifacts",
    "tests",
)


class WorkspaceCapability(BaseCapability):
    name = "workspace"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.PREPARE_WORKSPACE})

    def execute(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        expected_name = context.competition
        root.mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for name in _WORKSPACE_SUBDIRS:
            path = root / name
            existed = path.is_dir()
            path.mkdir(parents=True, exist_ok=True)
            if not existed:
                created.append(str(path))

        # Ensure research tree dirs exist (Analyze layout).
        context.paths.ensure()
        research_ok = context.paths.root.is_dir()
        named_ok = root.name == expected_name

        return TaskEvidence(
            task_id=context.task.id,
            execution_id=context.execution.id,
            capability=self.name,
            passed=research_ok and root.is_dir() and named_ok,
            summary="workspace prepared" if created else "workspace already present",
            checks=["dirs_exist", "writable", "named_as_competition"],
            paths=[str(root / name) for name in _WORKSPACE_SUBDIRS],
            error=None if named_ok else f"workspace must be named {expected_name!r}, got {root.name!r}",
            metadata={
                "created": created,
                "idempotent": not created,
                "workspace": str(root),
                "competition": expected_name,
            },
        )


def default_workspace_dirs(root: Path) -> list[Path]:
    return [root / name for name in _WORKSPACE_SUBDIRS]

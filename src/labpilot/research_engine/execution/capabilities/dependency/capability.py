"""Dependency capability — install/verify packages (deterministic)."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


def _requirements_candidates(context: TaskContext) -> list[Path]:
    roots = [
        context.workspace_root,
        context.paths.base_dir,  # workspace knowledge/ or legacy knowledge/
        context.paths.data_root,  # flat knowledge/ or knowledge/<slug>
        context.paths.root,
    ]
    names = ("requirements.txt", "requirements-lock.txt", "constraints.txt")
    found: list[Path] = []
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                found.append(path)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _packages_already_satisfied(req_path: Path) -> bool:
    """Best-effort: treat empty/missing importable pins as not satisfied."""
    text = req_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return True
    # Only check simple bare package names (no extras/urls) for no-op path.
    for line in lines:
        name = (
            line.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split("[")[0]
            .strip()
        )
        if not name or name.startswith("-") or "/" in name or name.startswith("git+"):
            continue
        mod = name.replace("-", "_").lower()
        try:
            importlib.import_module(mod)
        except Exception:
            return False
    return True


class DependencyCapability(BaseCapability):
    name = "dependency"

    def __init__(self, *, install: bool = True) -> None:
        """If ``install`` is False, only record requirement digests (tests)."""
        self._install = install

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.INSTALL_PACKAGE})

    def execute(self, context: TaskContext) -> TaskEvidence:
        reqs = _requirements_candidates(context)
        if not reqs:
            return TaskEvidence(
                task_id=context.task.id,
                execution_id=context.execution.id,
                capability=self.name,
                passed=True,
                summary="no requirements file; skipped install",
                checks=["no_requirements"],
                metadata={"skipped": True},
            )

        primary = reqs[0]
        digest = _file_digest(primary)
        if _packages_already_satisfied(primary):
            return TaskEvidence(
                task_id=context.task.id,
                execution_id=context.execution.id,
                capability=self.name,
                passed=True,
                summary=f"dependencies already satisfied ({primary.name})",
                checks=["already_satisfied"],
                paths=[str(primary)],
                metadata={"digest": digest, "idempotent": True},
            )

        if not self._install:
            return TaskEvidence(
                task_id=context.task.id,
                execution_id=context.execution.id,
                capability=self.name,
                passed=True,
                summary=f"install disabled; recorded {primary.name}",
                checks=["install_disabled"],
                paths=[str(primary)],
                metadata={"digest": digest, "would_install": True},
            )

        cmd = [sys.executable, "-m", "pip", "install", "-r", str(primary)]
        proc = subprocess.run(  # noqa: S603 - intentional pip install
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        return TaskEvidence(
            task_id=context.task.id,
            execution_id=context.execution.id,
            capability=self.name,
            passed=ok,
            summary="pip install ok" if ok else "pip install failed",
            checks=["pip_install"],
            paths=[str(primary)],
            error=None if ok else (proc.stderr or proc.stdout)[:2000],
            metadata={
                "digest": digest,
                "returncode": proc.returncode,
                "cmd": cmd,
            },
        )

"""Dependency capability — install/verify packages (deterministic)."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import (
    failure_excerpt,
    stream_text,
)
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.execution.training.environment import child_environment
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
                checks=["no_requirements", "no_verification"],
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
                checks=["already_satisfied", "no_verification"],
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
                checks=["install_disabled", "no_verification"],
                paths=[str(primary)],
                metadata={"digest": digest, "would_install": True},
            )

        cmd = [sys.executable, "-m", "pip", "install", "-r", str(primary)]
        try:
            proc = subprocess.run(  # noqa: S603 - intentional pip install
                cmd,
                capture_output=True,
                text=True,
                check=False,
                # Installing a package *runs* it: `setup.py` or a PEP 517 backend
                # executes during the build. The requirements file is written into
                # the workspace, where codegen chooses the paths it writes, so the
                # package names here are model-chosen — and this inherited the
                # operator's whole environment. Of the three places that execute
                # model-written code this was the most exposed and the last to be
                # fixed; the other two are the verification gates. M20 criterion 2,
                # reported reviewing PR #124.
                env=child_environment(),
                # Longer than either gate: a source build of a large wheel is
                # legitimately slow, and being killed mid-build is a worse failure
                # than waiting. The bound is for a build that hangs on a prompt or
                # a dead index, which otherwise stalls the campaign for good.
                timeout=int(context.constraints.get("install_timeout_s", 900)),
            )
        except subprocess.TimeoutExpired as expired:
            message = f"pip install timed out after {expired.timeout:.0f}s"
            # pip's partial output names the package it was collecting or
            # building when the clock ran out; without it a hung build reports
            # no package at all, while the branch below carries stderr into
            # `error` on an ordinary failure. Reported reviewing PR #124.
            streams = [stream_text(expired.output), stream_text(expired.stderr)]
            # Both streams: see `VerificationCapability._timed_out` — a timeout
            # has no traceback, so stderr-or-stdout would drop the line naming
            # the package pip was building.
            excerpt = failure_excerpt("", "\n".join(p for p in streams if p.strip()))
            return TaskEvidence(
                task_id=context.task.id,
                execution_id=context.execution.id,
                capability=self.name,
                passed=False,
                summary=message,
                checks=["pip_install", "timeout"],
                paths=[str(primary)],
                error=(
                    f"{message}. Nothing was verified about these dependencies — "
                    "an install that did not finish leaves the environment in an "
                    "unknown state rather than a satisfied one."
                    + (f"\nLast output before it was stopped:\n{excerpt}" if excerpt else "")
                ),
                metadata={"digest": digest, "timeout_s": expired.timeout, "cmd": cmd},
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

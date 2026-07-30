"""Verification — unit tests and ★ smoke gate (deterministic)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class VerificationCapability(BaseCapability):
    name = "verification"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.RUN_UNIT_TEST, TaskType.RUN_SMOKE_TEST})

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.type == TaskType.RUN_UNIT_TEST:
            return self._unit(context)
        return self._smoke(context)

    def _unit(self, context: TaskContext) -> TaskEvidence:
        tests_dir = context.workspace_root / "tests"
        log_path = context.workspace_root / "logs" / "unit_tests.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not tests_dir.is_dir() or not any(tests_dir.glob("test_*.py")):
            log_path.write_text("no tests; skipped\n", encoding="utf-8")
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary="no unit tests; skipped",
                checks=["no_tests"],
                paths=[str(log_path)],
                metadata={"skipped": True},
            )

        if is_dry_run(context):
            log_path.write_text("dry_run: unit tests not executed\n", encoding="utf-8")
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary="unit tests dry-run skip",
                checks=["dry_run"],
                paths=[str(log_path)],
                metadata={"dry_run": True},
            )

        started = time.monotonic()
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
            capture_output=True,
            text=True,
            check=False,
            cwd=context.workspace_root,
        )
        duration = time.monotonic() - started
        log_path.write_text(
            f"returncode={proc.returncode}\n{proc.stdout}\n{proc.stderr}\n",
            encoding="utf-8",
        )
        ok = proc.returncode == 0
        return evidence(
            context,
            capability=self.name,
            passed=ok,
            summary="unit tests passed" if ok else "unit tests failed",
            checks=["pytest"],
            paths=[str(log_path)],
            error=None if ok else (proc.stderr or proc.stdout)[:1500],
            metadata={"returncode": proc.returncode, "duration_s": duration},
        )

    def _smoke(self, context: TaskContext) -> TaskEvidence:
        """★ Production-shaped gate before full training."""
        train = context.workspace_root / "pipeline" / "train.py"
        log_path = context.workspace_root / "logs" / "smoke.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not train.is_file():
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="smoke failed: missing train.py",
                checks=["smoke_gate"],
                error="missing pipeline/train.py",
                paths=[str(log_path)],
            )

        # Always syntax-check.
        from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import validate_python_syntax

        syntax_errors = validate_python_syntax(train)
        if syntax_errors:
            log_path.write_text("\n".join(syntax_errors), encoding="utf-8")
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="smoke failed: syntax",
                checks=["smoke_gate", "syntax"],
                error="; ".join(syntax_errors),
                paths=[str(log_path)],
            )

        if is_dry_run(context) or context.constraints.get("smoke_syntax_only"):
            log_path.write_text("smoke syntax-only ok\n", encoding="utf-8")
            marker = context.workspace_root / "artifacts" / "smoke_ok.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('{"passed": true, "mode": "syntax_only"}\n', encoding="utf-8")
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary="smoke gate passed (syntax-only)",
                checks=["smoke_gate", "syntax"],
                paths=[str(log_path), str(marker)],
                metadata={"mode": "syntax_only"},
            )

        started = time.monotonic()
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(train)],
            capture_output=True,
            text=True,
            check=False,
            # Workspace root so scripts can open pipeline/config.yaml and write
            # metrics.json / submission.csv at the competition root.
            cwd=context.workspace_root,
            timeout=int(context.constraints.get("smoke_timeout_s", 120)),
            env={
                **dict(__import__("os").environ),
                "LABPILOT_SMOKE": "1",
            },
        )
        duration = time.monotonic() - started
        log_path.write_text(
            f"returncode={proc.returncode}\nduration_s={duration}\n"
            f"{proc.stdout}\n{proc.stderr}\n",
            encoding="utf-8",
        )
        ok = proc.returncode == 0
        if ok:
            marker = context.workspace_root / "artifacts" / "smoke_ok.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                f'{{"passed": true, "duration_s": {duration}}}\n',
                encoding="utf-8",
            )
        return evidence(
            context,
            capability=self.name,
            passed=ok,
            summary="smoke gate passed" if ok else "smoke gate failed",
            checks=["smoke_gate"],
            paths=[str(log_path)],
            error=None if ok else (proc.stderr or proc.stdout)[:1500],
            metadata={"returncode": proc.returncode, "duration_s": duration},
        )

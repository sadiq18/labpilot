"""Verification — unit tests and ★ smoke gate (deterministic)."""

from __future__ import annotations

import subprocess
import sys
import time

from labpilot.research_engine.execution.capabilities._helpers import (
    evidence,
    failure_excerpt,
    is_dry_run,
)
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.execution.training.environment import (
    child_environment,
    training_command,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: How much of a failure to keep. Same budget as before; the change is *which*
#: end of the output it comes from.


class VerificationCapability(BaseCapability):
    name = "verification"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.RUN_UNIT_TEST, TaskType.RUN_SMOKE_TEST})

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.type == TaskType.RUN_UNIT_TEST:
            return self._unit(context)
        return self._smoke(context)

    def _timed_out(
        self,
        context: TaskContext,
        log_path,
        expired: subprocess.TimeoutExpired,
        *,
        check: str,
    ) -> TaskEvidence:
        """A verdict for a step that ran out of time, rather than an exception.

        `subprocess.run` raises `TimeoutExpired`, and `engineer.py` calls
        `capability.execute` unwrapped — so without this the bound would trade a
        hang for a vanish: no evidence file, no verdict, nothing recorded as
        having decided anything. A gate that disappears is the failure this
        milestone is named after. Reported reviewing PR #124.
        """
        limit = expired.timeout
        message = f"{check} timed out after {limit:.0f}s"
        log_path.write_text(f"{message}\n", encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            passed=False,
            summary=message,
            checks=[check, "timeout"],
            paths=[str(log_path)],
            error=(
                f"{message}. The step was stopped, so nothing here says whether it "
                "would have passed — a run that does not finish has not been verified."
            ),
            metadata={"timeout_s": limit, "cmd": list(expired.cmd)},
        )

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
                checks=["no_tests", "no_verification"],
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
                checks=["dry_run", "no_verification"],
                paths=[str(log_path)],
                metadata={"dry_run": True},
            )

        started = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
                capture_output=True,
                text=True,
                check=False,
                cwd=context.workspace_root,
                # These are the workspace's tests — written by a model, importing
                # the same generated pipeline training runs. `TrainingRunner` strips
                # credentials before running that code and this gate did not, so the
                # check that runs first was the more permissive of the two. M20
                # criterion 2.
                env=child_environment(),
                # Longer than the smoke gate's 120s: a real generated suite may
                # legitimately take minutes, and this bound is here to catch a
                # `while True:` that would otherwise block the campaign forever
                # with no verdict at all — not to police slowness.
                timeout=int(context.constraints.get("unit_timeout_s", 600)),
            )
        except subprocess.TimeoutExpired as expired:
            return self._timed_out(context, log_path, expired, check="pytest")
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
            error=None if ok else failure_excerpt(proc.stderr or "", proc.stdout or ""),
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
        from labpilot.research_engine.execution.capabilities.code_engineering.syntax import (
            validate_python_syntax,
        )

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

        # Run it the way training will. This gate calls itself
        # "production-shaped", and invoking `python train.py` directly was not:
        # `training_command` uses `uv run --script` whenever the file declares
        # dependencies, so the two paths differ exactly where dependencies are
        # concerned — which is the failure this gate exists to catch first.
        #
        # Measured on rogii 2026-08-08: a `train.py` whose PEP 723 block was
        # unterminated **passed smoke** (bare `python` ignores the block, and a
        # docstring plus comments exits 0) and then failed training, where uv
        # refused the whole script. The gate reported success for a file that
        # could not run.
        started = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603
                training_command(train, python=sys.executable),
                capture_output=True,
                text=True,
                check=False,
                # Workspace root so scripts can open pipeline/config.yaml and write
                # metrics.json / submission.csv at the competition root.
                cwd=context.workspace_root,
                timeout=int(context.constraints.get("smoke_timeout_s", 120)),
                # The command was shared and the environment was not: this passed
                # the operator's `os.environ` straight through, so the first thing
                # to execute model-written code gave it the provider and Kaggle keys
                # that `child_environment` exists to withhold. `LABPILOT_SMOKE` is
                # the one difference this gate is entitled to — it is how a script
                # knows to take its short path, and production must not set it.
                env={**child_environment(), "LABPILOT_SMOKE": "1"},
            )
        except subprocess.TimeoutExpired as expired:
            return self._timed_out(context, log_path, expired, check="smoke_gate")
        duration = time.monotonic() - started
        log_path.write_text(
            f"returncode={proc.returncode}\nduration_s={duration}\n{proc.stdout}\n{proc.stderr}\n",
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
            error=None if ok else failure_excerpt(proc.stderr or "", proc.stdout or ""),
            metadata={"returncode": proc.returncode, "duration_s": duration},
        )

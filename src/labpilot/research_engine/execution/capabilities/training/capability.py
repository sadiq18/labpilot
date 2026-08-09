"""Training capability — wrap TrainingRunner; dry-run writes stub metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: Marker for "exited 0 and produced no result". Imported by the Engineer to
#: tell that apart from a training run that genuinely crashed.
METRICS_NOT_WRITTEN = "did not write metrics.json"


class TrainingCapability(BaseCapability):
    name = "training"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.RUN_TRAINING})

    def execute(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        train = root / "pipeline" / "train.py"
        if not train.is_file():
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="training failed: missing train.py",
                checks=["train_script"],
                error="missing pipeline/train.py",
            )

        # Smoke gate soft-check (deps usually enforce; belt-and-suspenders).
        smoke_ok = root / "artifacts" / "smoke_ok.json"
        if not smoke_ok.is_file() and not is_dry_run(context):
            # Allow if prior smoke task marked done via plan deps only.
            pass

        started = time.monotonic()
        # Wall clock, because mtime is compared against it; monotonic is not.
        wall_started = time.time()
        if is_dry_run(context) or context.constraints.get("train_stub", False):
            metrics = {
                "cv_accuracy": 0.5,
                "status": "dry_run_stub",
                "execution_id": context.execution.id,
                "plan_id": context.plan.id,
            }
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / ".stub").write_text("dry_run\n", encoding="utf-8")
            duration = time.monotonic() - started
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary="training dry-run stub metrics written",
                checks=["train_stub"],
                paths=[str(metrics_path)],
                metrics=metrics,
                metadata={"duration_s": duration, "dry_run": True},
            )

        from labpilot.research_engine.execution.training.runner import TrainingRunner

        runner = TrainingRunner(root)
        try:
            result = runner.run(
                timeout=context.constraints.get("train_timeout_s"),
            )
            log_path = runner.save_run_log(result)
            artifacts = runner.collect_artifacts()
            duration = time.monotonic() - started
            metrics_path = Path(artifacts["metrics"]) if artifacts.get("metrics") else root / "metrics.json"
            metrics: dict = {}
            if metrics_path.is_file():
                try:
                    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metrics = loaded
                except json.JSONDecodeError:
                    metrics = {}

            ok = result.returncode == 0
            error = None if ok else (result.stderr or result.stdout)[:2000]
            # Exit 0 alone is not enough — codegen sometimes emits a broken
            # ``__main__`` guard so the script no-ops without writing metrics.
            #
            # And presence alone is not enough either. `metrics.json` from an
            # earlier successful run sits at the workspace root and survives
            # every failure after it, so "is there a file?" answers yes for a
            # training run that wrote nothing. Measured on rogii 2026-08-09:
            # execution E-227 reported **succeeded** and plan P-025 went
            # **done** against a `metrics.json` written the previous evening —
            # a green plan with no result, and the number on it belonged to a
            # different experiment.
            #
            # `run_experiment` closed this hole with `_metrics_written_since`;
            # the Engineer path never had the guard. Ask whether *this run*
            # wrote it. A second of slack because some filesystems round mtime.
            fresh = metrics_path.is_file() and metrics_path.stat().st_mtime >= wall_started - 1.0
            if ok and not fresh:
                ok = False
                error = (
                    f"training exited 0 but {METRICS_NOT_WRITTEN} "
                    + (
                        "— the file on disk predates this run, so it belongs to an "
                        "earlier execution"
                        if metrics_path.is_file()
                        else "(check ``if __name__ == '__main__':`` and that main() runs)"
                    )
                )
            return evidence(
                context,
                capability=self.name,
                passed=ok,
                summary="training completed" if ok else "training failed",
                checks=["train_runner", "metrics_json"],
                paths=[str(p) for p in artifacts.values()] + [str(log_path)],
                metrics=metrics,
                error=error,
                metadata={"returncode": result.returncode, "duration_s": duration},
            )
        except Exception as exc:
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="training error",
                checks=["train_runner"],
                error=str(exc),
            )

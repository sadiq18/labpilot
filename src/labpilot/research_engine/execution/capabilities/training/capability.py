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
            if ok and not metrics_path.is_file():
                ok = False
                error = (
                    "training exited 0 but did not write metrics.json "
                    "(check ``if __name__ == '__main__':`` and that main() runs)"
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

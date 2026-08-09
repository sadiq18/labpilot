"""Training capability — wrap TrainingRunner; dry-run writes stub metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import (
    evidence,
    failure_excerpt,
    is_dry_run,
)
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: Markers for "exited 0 and produced no result". Imported by the Engineer to
#: tell those apart from a training run that genuinely crashed.
#:
#: Two of them, because there are two ways to finish with nothing: never
#: writing the file, and writing one that holds no metrics. The second was
#: added without a marker, so `_training_produced_nothing` — which matches on
#: these — returned "" for it, `code_is_suspect` stayed False, and the retry
#: reran the identical script to write the identical empty file. Reported on
#: PR #117.
METRICS_NOT_WRITTEN = "did not write metrics.json"
METRICS_EMPTY = "holds no metrics"
PRODUCED_NOTHING_MARKERS: tuple[str, ...] = (METRICS_NOT_WRITTEN, METRICS_EMPTY)


def _misplaced_note(root: Path, since: float) -> str:
    """Name a `metrics.json` this run wrote *somewhere else*.

    "It did not write metrics.json" is true and unhelpful when the script wrote
    one enthusiastically into a directory it invented. Measured on rogii
    2026-08-09: `./workspace/metrics.json`, created by the script's own
    `makedirs`, three retries in a row — each one told what was missing and
    never where its output had gone, so each edited something else.

    Cheap: one shallow glob, only on the failure path.
    """
    try:
        found = [
            path
            for path in root.rglob("metrics.json")
            if path.parent != root
            and ".venv" not in path.parts
            and path.stat().st_mtime >= since - 1.0
        ]
    except OSError:  # pragma: no cover - a listing failure must not mask the real error
        return ""
    if not found:
        return ""
    listed = ", ".join(str(p.relative_to(root)) for p in sorted(found)[:3])
    return (
        f". This run did write one at {listed} — move the output to the workspace "
        "root and do not create a directory for it"
    )


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
            metrics_path = (
                Path(artifacts["metrics"]) if artifacts.get("metrics") else root / "metrics.json"
            )
            metrics: dict = {}
            if metrics_path.is_file():
                try:
                    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metrics = loaded
                except json.JSONDecodeError:
                    metrics = {}

            ok = result.returncode == 0
            # The same tail-preferring, tqdm-collapsing excerpt the smoke gate
            # uses. This was still a raw head slice, so a crash whose stderr
            # opened with 2000 characters of `Loading train: 96%|` handed the
            # retry a progress bar and discarded the `KeyError` underneath —
            # for training crashes specifically, while the smoke path was
            # fixed. Reported on PR #117.
            error = None if ok else failure_excerpt(result.stderr, result.stdout, limit=2000)
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
            # Freshness is about *when*; this is about *whether there is a
            # result at all*. A run that exits 0 and writes `{}` passes every
            # timing check and reports success with nothing measured — the same
            # "reported succeeded, the number belongs to nothing real" class,
            # reached through content instead of staleness. Reported on PR #117.
            if ok and fresh and not metrics:
                ok = False
                error = (
                    f"training exited 0 and wrote {metrics_path.name}, but it "
                    f"{METRICS_EMPTY} — an empty or unparseable object is not a result"
                )
            if ok and not fresh:
                ok = False
                error = (
                    f"training exited 0 but {METRICS_NOT_WRITTEN} at "
                    f"{metrics_path.name} (workspace root)"
                    + (
                        " — the file there predates this run, so it belongs to an earlier execution"
                        if metrics_path.is_file()
                        else " (check ``if __name__ == '__main__':`` and that main() runs)"
                    )
                    + _misplaced_note(root, wall_started)
                )
            return evidence(
                context,
                capability=self.name,
                passed=ok,
                summary="training completed" if ok else "training failed",
                checks=["train_runner", "metrics_json"],
                paths=[str(p) for p in artifacts.values()] + [str(log_path)],
                # Nothing this run wrote, nothing it reports. A stale
                # `metrics.json` was still loaded and returned beside
                # `passed=False`, so any reader that trusted `evidence.metrics`
                # without checking `passed` first saw a plausible number
                # belonging to an earlier execution — the same file, and the
                # same confusion, the freshness guard above exists to end.
                metrics=metrics if fresh else {},
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

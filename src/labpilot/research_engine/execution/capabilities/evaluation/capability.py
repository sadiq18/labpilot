"""Inference, evaluation, and compare capabilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from labpilot.accessor.common.json_utils import dumps
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class EvaluationCapability(BaseCapability):
    name = "evaluation"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(
            {
                TaskType.RUN_INFERENCE,
                TaskType.EVALUATE,
                TaskType.COMPARE,
            }
        )

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.type == TaskType.RUN_INFERENCE:
            return self._infer(context)
        if context.task.type == TaskType.EVALUATE:
            return self._evaluate(context)
        return self._compare(context)

    def _infer(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        pred = root / "predictions.csv"
        if not pred.is_file():
            submission = root / "submission.csv"
            if submission.is_file():
                pred.write_text(submission.read_text(encoding="utf-8"), encoding="utf-8")
            elif is_dry_run(context):
                # A dry run checks the wiring, and the placeholder is the wiring.
                pred.write_text("id,prediction\n0,0\n", encoding="utf-8")
            else:
                # This wrote `id,prediction\n0,0` and then reported
                # `passed=pred.is_file()` — a verdict about a file it had just
                # fabricated, so a run that inferred nothing was indistinguishable
                # from one that inferred correctly. The same defect as
                # `submission`, in a different capability, found while writing a
                # rejection test for this site rather than by reading it. M20.
                return evidence(
                    context,
                    capability=self.name,
                    passed=False,
                    summary="nothing to infer from",
                    checks=["inference"],
                    error=(
                        "no predictions.csv and no submission.csv in the workspace. "
                        "Writing a placeholder row would produce a file that predicts "
                        "nothing and report it as inference."
                    ),
                )
        return evidence(
            context,
            capability=self.name,
            passed=pred.is_file(),
            summary="predictions ready",
            checks=["inference"],
            paths=[str(pred)],
            metadata={"dry_run": is_dry_run(context)},
        )

    def _evaluate(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        metrics_path = root / "metrics.json"
        if not metrics_path.is_file():
            if is_dry_run(context):
                metrics_path.write_text(
                    json.dumps(
                        {
                            "cv_accuracy": 0.5,
                            "cv_fold_scores": [0.48, 0.52],
                            "cv_mean": 0.5,
                            "cv_std": 0.02,
                            "train_time_s": 1.0,
                            "status": "dry_run_eval",
                            "execution_id": context.execution.id,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                return evidence(
                    context,
                    capability=self.name,
                    passed=False,
                    summary="metrics.json missing",
                    checks=["evaluate"],
                    error="metrics.json not found",
                )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment_id = self._upsert_experiment(context, metrics)
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary="metrics recorded",
            checks=["evaluate"],
            paths=[str(metrics_path)],
            metrics=metrics if isinstance(metrics, dict) else {},
            metadata={"experiment_id": experiment_id},
        )

    def _compare(self, context: TaskContext) -> TaskEvidence:
        from labpilot.research_engine.evidence.compare_service import (
            run_compare_and_build_card,
        )

        card = run_compare_and_build_card(context)
        root = context.workspace_root
        paths = [
            str(root / "comparison.json"),
            str(root / "artifacts" / "comparison.json"),
        ]
        # A comparison that compared nothing is not a comparison. `passed=True`
        # was unconditional, so a card built with no control, or over placeholder
        # metrics, reported success — and the card it wrote is what COMPARE
        # exists to produce. M20: the verdict has to be about the card's
        # substance, not about having reached the end of the function.
        compared = card.observed.cv_gain is not None
        return evidence(
            context,
            capability=self.name,
            passed=compared,
            error=(
                None
                if compared
                else (
                    f"Evidence {card.id} compared nothing: no cv_gain. "
                    f"{card.decision_reason or 'no control to compare against'}"
                )
            ),
            summary=(
                f"Evidence {card.id}: {card.decision.value} "
                f"(cv_gain={card.observed.cv_gain})"
            ),
            checks=["compare", "evidence_card"],
            paths=paths,
            metrics={
                "cv_delta": card.observed.cv_gain,
                "primary_delta": card.observed.cv_gain,
            },
            metadata={
                "evidence_card_id": card.id,
                "decision": card.decision.value,
                "control_experiment": card.control_experiment,
                "technique_attribution": card.technique_attribution,
                **card.to_comparison_dict(),
            },
        )

    def _upsert_experiment(self, context: TaskContext, metrics: dict) -> str:
        """Write a durable row into DB ``experiments`` (best-effort)."""
        exp_id = f"exp_{context.competition}_{context.execution.id}"
        now = datetime.now(UTC).isoformat()
        try:
            client = SqliteClient(context.paths.db_path)
            try:
                client.conn.execute(
                    """
                    INSERT INTO experiments (
                        id, competition_slug, summary, outcome, metrics,
                        techniques, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        metrics=excluded.metrics,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at
                    """,
                    (
                        exp_id,
                        context.competition,
                        f"execution {context.execution.id}",
                        "baseline"
                        if context.plan.metadata.get("plan_kind") == "baseline"
                        else "completed",
                        dumps(metrics),
                        dumps([]),
                        dumps(
                            {
                                "plan_id": context.plan.id,
                                "execution_id": context.execution.id,
                                "hypothesis_id": context.plan.hypothesis_id,
                            }
                        ),
                        now,
                        now,
                    ),
                )
                client.conn.commit()
            finally:
                client.close()
        except Exception:
            pass
        return exp_id

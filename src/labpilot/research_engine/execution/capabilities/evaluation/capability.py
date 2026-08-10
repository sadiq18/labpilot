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
                    # The same shape as every other return here. The first
                    # version dropped both, so failure evidence for this step
                    # differed from success evidence exactly when something had
                    # already gone wrong. Reported reviewing PR #121.
                    paths=[str(pred)],
                    metadata={"dry_run": is_dry_run(context)},
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
        # A comparison that compared nothing is not a comparison — but *no
        # control to compare against* is a different thing from *a control was
        # there and this produced nothing*. The first version tested
        # `cv_gain is not None`, which is False by construction on a
        # `missing_control` card, so every campaign's **first** COMPARE failed:
        # a baseline has no prior execution, and the baseline plan runs COMPARE
        # on it. Reported reviewing PR #121 — the same shape as the credential
        # gate on #120, right about the pathological input and wrong about the
        # ordinary one.
        #
        # With no control this step verified nothing, which is a state to
        # declare rather than a failure to report, so it takes the same
        # `no_verification` stamp the other such branches take.
        had_control = bool(card.control_experiment)
        compared = card.observed.cv_gain is not None
        checks = ["compare", "evidence_card"]
        if not had_control:
            checks.append("no_verification")
        return evidence(
            context,
            capability=self.name,
            passed=compared or not had_control,
            error=(
                None
                if compared or not had_control
                else (
                    f"Evidence {card.id} compared nothing: a control was available "
                    f"({card.control_experiment}) and the comparison still produced "
                    f"no cv_gain. {card.decision_reason or ''}".strip()
                )
            ),
            summary=(
                f"Evidence {card.id}: {card.decision.value} (cv_gain={card.observed.cv_gain})"
            ),
            checks=checks,
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

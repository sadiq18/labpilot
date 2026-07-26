"""Reporting & Memory — report, reflect, belief, hypothesis writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class ReportingCapability(BaseCapability):
    name = "reporting"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(
            {
                TaskType.GENERATE_REPORT,
                TaskType.REFLECT,
                TaskType.UPDATE_BELIEF,
                TaskType.CREATE_HYPOTHESIS,
            }
        )

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.type == TaskType.GENERATE_REPORT:
            return self._report(context)
        if context.task.type == TaskType.REFLECT:
            return self._reflect(context)
        if context.task.type == TaskType.UPDATE_BELIEF:
            return self._belief(context)
        return self._hypothesis(context)

    def _load_metrics(self, root: Path) -> dict:
        metrics_path = root / "metrics.json"
        if metrics_path.is_file():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        return {}

    def _report(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        metrics = self._load_metrics(root)
        reports_dir = context.paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{context.execution.id}_report.md"
        lines = [
            f"# Experiment report — {context.competition}",
            "",
            f"- plan: `{context.plan.id}`",
            f"- execution: `{context.execution.id}`",
            f"- plan_kind: `{context.plan.metadata.get('plan_kind', '')}`",
            f"- hypothesis: `{context.plan.hypothesis_id or '—'}`",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2),
            "```",
            "",
            f"Generated at {datetime.now(UTC).isoformat()}",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        # Also copy under workspace artifacts.
        local = root / "artifacts" / "report.md"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary="report written",
            checks=["generate_report"],
            paths=[str(report_path), str(local)],
            metrics=metrics,
        )

    def _reflect(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        metrics = self._load_metrics(root)
        reflection = {
            "plan_id": context.plan.id,
            "execution_id": context.execution.id,
            "competition": context.competition,
            "metrics": metrics,
            "notes": [
                "Deterministic reflection (no LLM inventing metrics).",
                "Next: create hypothesis plans against this baseline if plan_kind=baseline.",
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = root / "artifacts" / "reflection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reflection, indent=2) + "\n", encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary="reflection captured",
            checks=["reflect"],
            paths=[str(path)],
            metadata=reflection,
        )

    def _belief(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        metrics = self._load_metrics(root)
        belief = {
            "technique": "baseline",
            "competition": context.competition,
            "plan_id": context.plan.id,
            "execution_id": context.execution.id,
            "metrics": metrics,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path = root / "artifacts" / "belief_update.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(belief, indent=2) + "\n", encoding="utf-8")
        # Best-effort DB belief row is out of scope if schema requires more fields;
        # durable JSON is enough for MVP memory hook.
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary="belief update recorded",
            checks=["update_belief"],
            paths=[str(path)],
            metadata=belief,
        )

    def _hypothesis(self, context: TaskContext) -> TaskEvidence:
        path = context.workspace_root / "artifacts" / "next_hypothesis.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suggestion": "Propose an improvement hypothesis against P-001 metrics.",
            "plan_id": context.plan.id,
            "execution_id": context.execution.id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary="hypothesis suggestion recorded",
            checks=["create_hypothesis"],
            paths=[str(path)],
            metadata=payload,
        )

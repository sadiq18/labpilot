"""Reporting & Memory — report + reflection library cutover."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.memory.hooks import persist_experience_from_completion
from labpilot.research_engine.planner.schemas.task_types import TaskType
from labpilot.research_engine.reflection.pipeline import run_reflection

logger = logging.getLogger(__name__)


class ReportingCapability(BaseCapability):
    """Report, reflect, update beliefs, propose the next hypothesis.

    Every one of the four used to return `passed=True` unconditionally, because
    each ends by writing a file and `write_text` either works or raises. That is
    the M20 shape exactly: the verdict answered *"did I write something"* while
    the step promises *"this execution was reported on"*. A run that produced no
    metrics still got a report, a reflection with no assessment still "completed
    the pipeline", and a hypothesis step still recorded a suggestion — the canned
    fallback string, which is a default, not a suggestion.

    So each now reports whether it had **anything to say**. Writing the artifact
    is not the achievement; having content in it is.
    """

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

    def _run_reflection_pipeline(self, context: TaskContext) -> dict:
        llm = None
        if not context.constraints.get("offline"):
            llm = context.constraints.get("llm_client")
        return run_reflection(
            context.paths.base_dir,
            context.competition,
            execution_id=context.execution.id,
            workspace_path=context.workspace_root,
            plan_id=context.plan.id,
            hypothesis_id=context.plan.hypothesis_id or None,
            llm_client=llm,
            persist=True,
        )

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
        local = root / "artifacts" / "report.md"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            # A report of an execution that produced no metrics is a heading and
            # an empty JSON block. The run it describes did not happen.
            passed=bool(metrics),
            error=None if metrics else "no metrics to report — the run produced none",
            summary="report written" if metrics else "report written with no metrics",
            checks=["generate_report"],
            paths=[str(report_path), str(local)],
            metrics=metrics,
        )

    def _persist_experience_fallback(
        self, context: TaskContext, *, reflection: dict
    ) -> None:
        """Write-only memory upsert when Blinker completion may not have fired."""
        metrics = self._load_metrics(context.workspace_root)
        experiment_id = getattr(context.execution, "experiment_id", None)
        persist_experience_from_completion(
            {
                "competition": context.competition,
                "knowledge_dir": str(context.paths.base_dir),
                "workspace_root": str(context.workspace_root),
                "execution_id": context.execution.id,
                "experiment_id": experiment_id or context.execution.id,
                "plan_id": context.plan.id,
                "hypothesis_id": context.plan.hypothesis_id or None,
                "status": getattr(context.execution, "status", None) or "completed",
                "metrics": metrics,
                "reflection": reflection,
            }
        )

    def _reflect(self, context: TaskContext) -> TaskEvidence:
        result = self._run_reflection_pipeline(context)
        root = context.workspace_root
        path = root / "artifacts" / "reflection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        projection = {
            "plan_id": context.plan.id,
            "execution_id": context.execution.id,
            "competition": context.competition,
            "evidence_id": (result.get("evidence") or {}).get("id"),
            "assessment": result.get("assessment"),
            "belief": result.get("belief"),
            "hypothesis": result.get("hypothesis"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
        try:
            self._persist_experience_fallback(context, reflection=result)
        except Exception:  # noqa: BLE001 — never fail reflect for memory
            logger.exception("experience memory fallback after reflect failed")
        return evidence(
            context,
            capability=self.name,
            # "Completed" was true of the call, not of the pipeline. A run that
            # returns neither an assessment nor a card reflected on nothing.
            passed=bool(result.get("assessment") or result.get("evidence")),
            error=(
                None
                if (result.get("assessment") or result.get("evidence"))
                else "the reflection pipeline returned neither an assessment nor a card"
            ),
            summary="reflection pipeline completed",
            checks=["reflect"],
            paths=[str(path)],
            metadata=projection,
        )

    def _belief(self, context: TaskContext) -> TaskEvidence:
        # Belief mutation is owned by REFLECT pipeline; this task re-runs updater
        # path via full pipeline for idempotent DAG plans that list both tasks.
        result = self._run_reflection_pipeline(context)
        root = context.workspace_root
        path = root / "artifacts" / "belief_update.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.get("belief") or {}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            # An empty payload is not a belief update; it is an empty file named
            # after one.
            passed=bool(payload),
            error=None if payload else "no belief update was produced",
            summary="belief update recorded",
            checks=["update_belief"],
            paths=[str(path)],
            metadata=payload,
        )

    def _hypothesis(self, context: TaskContext) -> TaskEvidence:
        result = self._run_reflection_pipeline(context)
        path = context.workspace_root / "artifacts" / "next_hypothesis.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        assessment = result.get("assessment") or {}
        # The fallback is a *default*, not a suggestion — it says nothing about
        # this execution and would read identically after every run. Kept so the
        # file is well-formed, but it no longer counts as having proposed
        # something.
        recommendation = assessment.get("recommendation")
        payload = {
            "suggestion": recommendation
            or "Propose an improvement hypothesis against baseline metrics.",
            "plan_id": context.plan.id,
            "execution_id": context.execution.id,
            "hypothesis_evaluation": result.get("hypothesis"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return evidence(
            context,
            capability=self.name,
            passed=bool(recommendation),
            error=(
                None
                if recommendation
                else "no recommendation was produced; the file holds the default text"
            ),
            summary="hypothesis suggestion recorded",
            checks=["create_hypothesis"],
            paths=[str(path)],
            metadata=payload,
        )

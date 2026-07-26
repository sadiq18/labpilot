"""Research Review — research-correctness gate (LLM optional; deterministic policy)."""

from __future__ import annotations

import ast
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class ResearchReviewCapability(BaseCapability):
    """Gate after code/config changes.

    Without an LLM: deterministic checks (train.py exists + parses). Critical
    findings block. With ``force_block`` in task metadata (tests), always fail.
    """

    name = "research_review"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.RESEARCH_REVIEW})

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.metadata.get("force_block"):
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="blocked by force_block metadata",
                checks=["force_block"],
                error="critical research finding (forced)",
            )

        train = context.workspace_root / "pipeline" / "train.py"
        findings: list[str] = []
        if not train.is_file():
            findings.append("critical: missing pipeline/train.py")
        else:
            try:
                ast.parse(train.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                findings.append(f"critical: train.py syntax error: {exc}")

        # Optional LLM judgement slice (soft): never invent metrics; may add notes.
        llm_notes: list[str] = []
        if self._llm is not None and not is_dry_run(context):
            llm_notes.append("llm_client present; deterministic checks still authoritative")

        critical = [f for f in findings if f.startswith("critical:")]
        passed = not critical
        return evidence(
            context,
            capability=self.name,
            passed=passed,
            summary="review passed" if passed else "review blocked",
            checks=["train_exists", "syntax"] + (["llm_note"] if llm_notes else []),
            paths=[str(train)] if train.is_file() else [],
            error="; ".join(critical) if critical else None,
            metadata={
                "findings": findings,
                "llm_notes": llm_notes,
                "decision": "allow" if passed else "block",
            },
        )

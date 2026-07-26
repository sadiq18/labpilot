"""Typed recovery policies (stub for Plan 2 — retry once / fail)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.models import ResearchTask


class RecoveryAction(StrEnum):
    RETRY = "retry"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str = ""


def decide_recovery(
    task: ResearchTask,
    evidence: TaskEvidence,
    *,
    attempt: int,
) -> RecoveryDecision:
    """Stub policy: retry once if ``max_retries`` allows, else fail."""
    max_retries = task.retry_policy.max_retries
    if attempt < max_retries:
        return RecoveryDecision(
            action=RecoveryAction.RETRY,
            reason=f"attempt {attempt + 1}/{max_retries}",
        )
    if task.retry_policy.abort_on_failure:
        return RecoveryDecision(
            action=RecoveryAction.FAIL,
            reason=evidence.error or evidence.summary or "verification failed",
        )
    return RecoveryDecision(action=RecoveryAction.SKIP, reason="abort_on_failure=False")

"""Operator approval gates with durable comments."""

from __future__ import annotations

from collections.abc import Callable

from labpilot.research_engine.conductor.models import ApprovalResult, OperatorFeedback
from labpilot.research_engine.conductor.store import ConductorStore

# Tools that pause for human approval by default.
GATED_TOOLS = frozenset({"generate_plan", "submit", "submit_learn"})

ApprovalPrompt = Callable[[str], ApprovalResult]


def auto_approve(tool_name: str) -> ApprovalResult:
    """Non-interactive approve (tests / ``--yes``)."""
    return ApprovalResult(decision="approve", comment="", gated_tool=tool_name)


def prompt_approval(tool_name: str) -> ApprovalResult:
    """Interactive CLI approve/reject + optional comment."""
    print(f"\nApproval required for tool: {tool_name}")
    raw = input("Approve? [y/N]: ").strip().lower()
    approved = raw in {"y", "yes"}
    comment = input("Comment (optional, feeds future Conductor decisions): ").strip()
    return ApprovalResult(
        decision="approve" if approved else "reject",
        comment=comment,
        gated_tool=tool_name,
    )


def maybe_approve(
    store: ConductorStore,
    *,
    session_id: str,
    tool_name: str,
    decision_id: str | None = None,
    task_id: str | None = None,
    auto: bool = False,
    prompt: ApprovalPrompt | None = None,
) -> ApprovalResult | None:
    """Return ApprovalResult when ``tool_name`` is gated; else ``None`` (no gate)."""
    if tool_name not in GATED_TOOLS:
        return None
    fn = prompt or (auto_approve if auto else prompt_approval)
    result = fn(tool_name)
    result.decision_id = decision_id
    result.task_id = task_id
    store.append_feedback(
        OperatorFeedback(
            id=store.new_feedback_id(),
            session_id=session_id,
            gated_tool=tool_name,
            decision=result.decision,
            comment=result.comment,
            decision_id=decision_id,
            task_id=task_id,
        )
    )
    return result

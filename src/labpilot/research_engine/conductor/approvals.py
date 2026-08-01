"""Operator approval gates with durable comments and autonomy levels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from labpilot.research_engine.conductor.models import ApprovalResult, OperatorFeedback
from labpilot.research_engine.conductor.store import ConductorStore

SUBMIT_TOOLS = frozenset({"submit", "submit_learn"})
PLAN_TOOLS = frozenset({"generate_plan"})

ApprovalPrompt = Callable[[str], ApprovalResult]
OfflineFallbackDecision = Literal["allow", "deny", "retry"]
OfflineFallbackPrompt = Callable[[str], OfflineFallbackDecision]


def gated_tools_for_autonomy(level: int) -> frozenset[str]:
    """Return tools that require approval for the given autonomy level.

    Level 0: plan + submit. Level 1: submit only. Submit always included (S2).
    """
    level = 0 if level < 0 else (1 if level > 1 else level)
    if level == 0:
        return PLAN_TOOLS | SUBMIT_TOOLS
    return SUBMIT_TOOLS


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
    autonomy: int = 0,
) -> ApprovalResult | None:
    """Return ApprovalResult when ``tool_name`` is gated; else ``None``."""
    gated = gated_tools_for_autonomy(autonomy)
    # S2: submit family always gated regardless of misconfigured autonomy.
    if tool_name in SUBMIT_TOOLS:
        gated = gated | SUBMIT_TOOLS
    if tool_name not in gated:
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
    store.increment_metric(session_id, "human_interventions")
    return result


def auto_allow_offline_fallback(_reason: str) -> OfflineFallbackDecision:
    """Non-interactive allow (tests / ``--yes``)."""
    return "allow"


def prompt_offline_fallback(reason: str) -> OfflineFallbackDecision:
    """Ask whether to use deterministic offline policy after an LLM failure."""
    print(f"\nLLM policy unavailable or failed: {reason}")
    print("Fall back to offline deterministic policy?")
    while True:
        raw = input("[a]llow / [d]eny / [r]etry: ").strip().lower()
        if raw in {"a", "allow", "y", "yes"}:
            return "allow"
        if raw in {"d", "deny", "n", "no"}:
            return "deny"
        if raw in {"r", "retry"}:
            return "retry"
        print("Enter allow, deny, or retry.")


def resolve_offline_fallback(
    reason: str,
    *,
    auto: bool = False,
    prompt: OfflineFallbackPrompt | None = None,
) -> OfflineFallbackDecision:
    """Return operator decision for offline policy fallback."""
    fn = prompt or (auto_allow_offline_fallback if auto else prompt_offline_fallback)
    return fn(reason)

"""Post-AI artifact verification — accept / reject / spot-check.

Standalone helper (no Conductor import) so tools and reflection can gate durable
writes without violating control-plane import direction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

VerifyDecision = Literal["approve", "reject", "spot_check"]
VerifyPrompt = Callable[[str, dict[str, Any]], "VerifyResult"]


class VerifyResult(BaseModel):
    """Outcome of an explicit second-pass check on an AI-produced artifact."""

    decision: VerifyDecision
    kind: str
    comment: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def auto_approve_artifact(kind: str, payload: dict[str, Any] | None = None) -> VerifyResult:
    """Non-interactive approve (tests / ``--yes`` / default analyze path)."""
    del payload
    return VerifyResult(decision="approve", kind=kind)


def prompt_verify_artifact(kind: str, payload: dict[str, Any] | None = None) -> VerifyResult:
    """Interactive approve / reject / spot-check for AI artifacts."""
    del payload
    print(f"\nVerify AI artifact: {kind}")
    print("  [a]pprove  [r]eject  [s]pot-check (write with needs_review)")
    while True:
        raw = input("Decision [a/r/s] (default a): ").strip().lower()
        if raw in {"", "a", "approve", "y", "yes"}:
            decision: VerifyDecision = "approve"
            break
        if raw in {"r", "reject", "n", "no"}:
            decision = "reject"
            break
        if raw in {"s", "spot", "spot_check", "spot-check"}:
            decision = "spot_check"
            break
        print("Enter a, r, or s.")
    comment = input("Comment (optional): ").strip()
    return VerifyResult(decision=decision, kind=kind, comment=comment)


def verify_ai_artifact(
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    auto: bool = True,
    prompt: VerifyPrompt | None = None,
) -> VerifyResult:
    """Run an explicit verification step for an AI-produced artifact.

    Defaults to auto-approve so existing non-interactive flows stay unchanged.
    Pass a custom ``prompt`` for interactive / test gates. ``auto=False`` without
    ``prompt`` raises — it does not block on ``input()``.
    """
    data = dict(payload or {})
    if prompt is not None:
        fn = prompt
    elif auto:
        fn = auto_approve_artifact
    else:
        raise ValueError(
            "verify_ai_artifact(auto=False) requires an explicit prompt= callback; "
            "refusing to block on interactive input()"
        )
    result = fn(kind, data)
    if result.kind != kind:
        result = result.model_copy(update={"kind": kind})
    return result

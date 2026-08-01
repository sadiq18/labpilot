"""ResearchAction → ActionPlan mapper (existing tools only)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ResearchAction(BaseModel):
    """High-level research intent from policy."""

    intent: str
    rationale: str = ""
    suggested_tools: list[str] = Field(default_factory=list)
    stop: bool = False


class ActionPlan(BaseModel):
    """Concrete tool chain mapped from a ResearchAction."""

    steps: list[ToolStep] = Field(default_factory=list)
    unmapped: bool = False
    suggestion: str | None = None


# Offline / template intents → tool chains
_TEMPLATES: list[tuple[tuple[str, ...], list[ToolStep]]] = [
    (
        ("paper", "literature", "search", "read"),
        [ToolStep(tool="search_papers", args={"offline": True})],
    ),
    (
        ("memory", "recall", "retrieve", "what do we know"),
        [ToolStep(tool="query_memory", args={})],
    ),
    (
        ("analyze", "competition", "understand"),
        [ToolStep(tool="analyze_competition", args={})],
    ),
    (
        ("plan", "baseline", "hypothesis"),
        [ToolStep(tool="generate_plan", args={"baseline": True})],
    ),
    (
        ("experiment", "run", "train", "try"),
        [
            ToolStep(tool="generate_plan", args={"baseline": True}),
            ToolStep(tool="run_plan", args={"plan_id": "P-001", "dry_run": True}),
            ToolStep(tool="reflect", args={"persist": False}),
        ],
    ),
    (
        ("augment", "minority", "investigate"),
        [
            ToolStep(tool="search_papers", args={"offline": True}),
            ToolStep(tool="generate_plan", args={"baseline": True}),
            ToolStep(tool="run_plan", args={"plan_id": "P-001", "dry_run": True}),
            ToolStep(tool="reflect", args={"persist": False}),
        ],
    ),
    (
        ("submit", "leaderboard", "upload"),
        [ToolStep(tool="submit", args={"execution_id": "E-001"})],
    ),
]


def map_research_action(
    action: ResearchAction,
    allowlist: set[str],
) -> ActionPlan:
    """Map an action to tools in ``allowlist``; never invent tool names."""
    if action.stop:
        return ActionPlan(steps=[], unmapped=False)

    # Prefer explicit suggested_tools if all are registered.
    if action.suggested_tools:
        steps: list[ToolStep] = []
        for name in action.suggested_tools:
            if name not in allowlist:
                return ActionPlan(
                    steps=[],
                    unmapped=True,
                    suggestion=(
                        f"Need capability/tool {name!r} for intent: {action.intent}"
                    ),
                )
            steps.append(ToolStep(tool=name, args=_default_args(name)))
        return ActionPlan(steps=steps)

    intent_l = action.intent.lower()
    for keywords, template_steps in _TEMPLATES:
        if any(k in intent_l for k in keywords):
            steps = []
            for step in template_steps:
                if step.tool not in allowlist:
                    return ActionPlan(
                        steps=[],
                        unmapped=True,
                        suggestion=(
                            f"Need capability/tool {step.tool!r} for intent: {action.intent}"
                        ),
                    )
                steps.append(step)
            return ActionPlan(steps=steps)

    return ActionPlan(
        steps=[],
        unmapped=True,
        suggestion=f"No available tool/capability for intent: {action.intent}",
    )


def _default_args(tool: str) -> dict[str, Any]:
    if tool == "generate_plan":
        return {"baseline": True}
    if tool == "search_papers":
        return {"offline": True}
    if tool == "run_plan":
        return {"plan_id": "P-001", "dry_run": True}
    if tool == "submit":
        return {"execution_id": "E-001"}
    return {}


def offline_next_research_action(
    completed_tools: list[str],
    allowlist: set[str],
) -> ResearchAction:
    """Deterministic research actions for offline campaign tests."""
    done = set(completed_tools)
    sequence = [
        ("Analyze the competition landscape", ["analyze_competition"]),
        ("Search related papers", ["search_papers"]),
        ("Query memory for techniques", ["query_memory"]),
        (
            "Investigate whether augmentation helps minority classes",
            ["search_papers", "generate_plan", "run_plan", "reflect"],
        ),
    ]
    for intent, tools in sequence:
        if not all(t in allowlist for t in tools):
            continue
        if all(t in done for t in tools):
            continue
        missing = [t for t in tools if t not in done]
        return ResearchAction(
            intent=intent,
            rationale="offline campaign template",
            suggested_tools=missing,
        )
    return ResearchAction(intent="done", rationale="catalog exhausted", stop=True)

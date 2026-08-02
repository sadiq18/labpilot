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
    missing_tools: list[str] = Field(default_factory=list)


# Resolved against real workspace state just before execution. Hardcoding
# "P-001"/"E-001" meant a campaign either re-ran the first plan forever or
# failed outright as soon as ids advanced.
LATEST = "@latest"
FIRST_PLAN_ID = "P-001"
FIRST_EXECUTION_ID = "E-001"


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
        [
            ToolStep(
                tool="analyze_competition",
                args={"fetch_kaggle": True},
            )
        ],
    ),
    (
        ("plan", "baseline", "hypothesis"),
        [ToolStep(tool="generate_plan", args={"baseline": True})],
    ),
    (
        ("implement", "write code", "code fix", "eda", "feature"),
        [ToolStep(tool="implement", args={"description": "update workspace code"})],
    ),
    (
        ("experiment", "run", "train", "try"),
        [
            ToolStep(tool="generate_plan", args={"baseline": True}),
            ToolStep(tool="implement", args={"description": "prepare train/infer code"}),
            ToolStep(tool="run_experiment", args={"plan_id": LATEST}),
            ToolStep(tool="reflect", args={"persist": False}),
        ],
    ),
    (
        ("augment", "minority", "investigate"),
        [
            ToolStep(tool="search_papers", args={"offline": True}),
            ToolStep(tool="generate_plan", args={"baseline": True}),
            ToolStep(tool="run_experiment", args={"plan_id": LATEST}),
            ToolStep(tool="reflect", args={"persist": False}),
        ],
    ),
    (
        ("submit", "leaderboard", "upload"),
        [ToolStep(tool="submit", args={"execution_id": LATEST})],
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
                    missing_tools=[name],
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
                        missing_tools=[step.tool],
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
    if tool == "analyze_competition":
        # Evidence breadth is the input to everything downstream: artifacts ->
        # concepts -> techniques -> beliefs/claims -> hypotheses. Running the
        # default analyzer set (competition, dataset, experiments, papers,
        # repositories) AND pulling Kaggle kernels/discussions is what gives a
        # campaign something to iterate on. Previously this ran with no
        # arguments at all, so `fetch_kaggle` defaulted to False and no kernel
        # or discussion evidence was ever gathered.
        return {"fetch_kaggle": True}
    if tool == "generate_plan":
        return {"baseline": True}
    if tool == "search_papers":
        return {"offline": True}
    if tool in {"run_plan", "run_experiment"}:
        return {"plan_id": LATEST}
    if tool == "implement":
        return {"description": "update workspace code"}
    if tool == "submit":
        return {"execution_id": LATEST}
    return {}


def resolve_step_args(
    tool: str,
    args: dict[str, Any],
    *,
    latest_plan_id: str | None,
    latest_execution_id: str | None,
    next_hypothesis_id: str | None = None,
    baseline_plan_exists: bool = False,
) -> dict[str, Any]:
    """Replace ``@latest`` placeholders with ids that actually exist.

    Falls back to the conventional first id when nothing has been created yet:
    a step earlier in the same batch is usually about to mint exactly that id,
    so failing here would stall the batch on its first run.

    Also switches ``generate_plan`` off ``baseline`` once a baseline plan
    exists. Baseline compilation is idempotent, so a campaign that only ever
    asked for a baseline could never mint a second plan — and therefore could
    never run a second experiment. Iterating means planning against a proposed
    hypothesis instead.
    """
    resolved = dict(args)
    if resolved.get("plan_id") == LATEST:
        resolved["plan_id"] = latest_plan_id or FIRST_PLAN_ID
    if resolved.get("execution_id") == LATEST:
        resolved["execution_id"] = latest_execution_id or FIRST_EXECUTION_ID
    if tool == "generate_plan" and resolved.get("baseline") and baseline_plan_exists:
        if next_hypothesis_id:
            resolved.pop("baseline", None)
            resolved["hypothesis_id"] = next_hypothesis_id
    return resolved


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

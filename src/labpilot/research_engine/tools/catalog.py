"""Catalog of default Research OS tools."""

from __future__ import annotations

from labpilot.research_engine.tools.descriptors import ToolDescriptor
from labpilot.research_engine.tools.handlers import (
    analyze_competition,
    generate_plan,
    implement,
    query_memory,
    reflect,
    run_experiment,
    run_plan,
    search_papers,
    submit,
    submit_learn,
)
from labpilot.research_engine.tools.registry import ToolRegistry


def build_default_tool_registry() -> ToolRegistry:
    """Return a registry with the default Research OS tools registered."""
    registry = ToolRegistry()
    for descriptor in default_tool_descriptors():
        registry.register(descriptor)
    return registry


def default_tool_descriptors() -> list[ToolDescriptor]:
    """Descriptors for the initial tool catalog."""
    return [
        ToolDescriptor(
            name="analyze_competition",
            description="Analyze a competition and write CompetitionAnalysis.",
            input_schema={
                "type": "object",
                "properties": {
                    "only": {"type": ["string", "null"]},
                    "refresh": {"type": "boolean"},
                },
            },
            output_artifacts=["competition_analysis"],
            handler=analyze_competition,
            capability_status="real",
            varies_by=["only"],
        ),
        ToolDescriptor(
            name="search_papers",
            description="Search papers for the competition / query and write a projection.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                    "offline": {"type": "boolean"},
                },
            },
            output_artifacts=["paper_search"],
            handler=search_papers,
            # Real when authenticated; degrades to an empty hit list
            # (source="offline" / source="error:<Type>") under offline=True
            # or any network failure — honestly, not disguised as success.
            capability_status="partial",
        ),
        ToolDescriptor(
            name="generate_plan",
            description="Compile a baseline or hypothesis research plan.",
            input_schema={
                "type": "object",
                "properties": {
                    "baseline": {"type": "boolean"},
                    "hypothesis_id": {"type": ["string", "null"]},
                    "priority": {"type": "integer"},
                },
            },
            output_artifacts=["research_plan"],
            handler=generate_plan,
            capability_status="real",
            varies_by=["hypothesis_id"],
        ),
        ToolDescriptor(
            name="implement",
            description=(
                "Implementation specialist: write/update code via CodingTool "
                "(EDA/features as code tasks)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "capability": {"type": "string"},
                    "force_rewrite": {"type": "boolean"},
                },
            },
            output_artifacts=["code"],
            handler=implement,
            # partial, not real, for two independent reasons — see
            # docs/research-os/autonomy-roadmap/10-capability-audit.md
            # §"implement: a second hollow path":
            #
            # 1. ImplementationSpecialist's prefer_patch shortcut skips the
            #    M19-fixed codegen path by default whenever the workspace
            #    already has code, and reports success without touching
            #    train.py. Only reaches the real path on a fresh workspace or
            #    with force_rewrite=True explicitly passed.
            # 2. `varies_by` is `description`, NOT `technique`. The
            #    `technique` kwarg never reaches the codegen prompt on this
            #    path: `build_v1_task_context` puts the agent task's metadata
            #    on the synthetic *ResearchTask*, while
            #    `CodeEngineeringCapability._write` reads `plan.metadata` —
            #    written to one object, read from another, so the prompt
            #    renders `Technique: —` however the caller sets it. Only
            #    `description` (via `goal`/`task_description`) actually
            #    conditions the output. Declaring `technique` here would be
            #    exactly the unverified capability claim M15 exists to catch.
            capability_status="partial",
            varies_by=["description"],
        ),
        ToolDescriptor(
            name="run_plan",
            description="Execute an approved ResearchPlan via the Research Engineer.",
            input_schema={
                "type": "object",
                "required": ["plan_id"],
                "properties": {
                    "plan_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                    "submit": {"type": "boolean"},
                },
            },
            output_artifacts=["execution"],
            handler=run_plan,
            capability_status="real",
            varies_by=["plan_id"],
        ),
        ToolDescriptor(
            name="run_experiment",
            description=(
                "Experiment specialist: run a plan, collect metrics, write experiment "
                "record (never live-submits)."
            ),
            input_schema={
                "type": "object",
                "required": ["plan_id"],
                "properties": {
                    "plan_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                    "description": {"type": "string"},
                },
            },
            output_artifacts=["execution", "experiment", "metrics"],
            handler=run_experiment,
            # Independent handler from run_plan (routes to CodeEngineeringCapability
            # directly, not through ImplementationSpecialist) — the prefer_patch
            # finding on `implement` does not apply here.
            capability_status="real",
            varies_by=["plan_id"],
        ),
        ToolDescriptor(
            name="reflect",
            description="Reflect on an execution and update beliefs / claims.",
            input_schema={
                "type": "object",
                "properties": {
                    "execution_id": {"type": ["string", "null"]},
                    "persist": {"type": "boolean"},
                },
            },
            output_artifacts=["reflection"],
            handler=reflect,
            # Real but inert: produces genuinely different beliefs/evidence per
            # execution; nothing downstream reads them yet — see M8.
            capability_status="real",
            varies_by=["execution_id"],
        ),
        ToolDescriptor(
            name="submit",
            description="Package an execution submission CSV under workspace artifacts.",
            input_schema={
                "type": "object",
                "required": ["execution_id"],
                "properties": {"execution_id": {"type": "string"}},
            },
            output_artifacts=["submission"],
            handler=submit,
            # package_execution_submission copies submission.csv verbatim;
            # execution_id only relabels the destination filename. Content
            # never depends on input — an honest fixed step, not a hollow one.
            capability_status="fixed",
        ),
        ToolDescriptor(
            name="submit_learn",
            description="Upload a submission and apply leaderboard learning.",
            input_schema={
                "type": "object",
                "required": ["execution_id"],
                "properties": {
                    "execution_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
            },
            output_artifacts=["submission"],
            handler=submit_learn,
            # Real even under dry_run=True: build_execution_outcome /
            # load_execution_outcome still return real per-execution metrics,
            # not a canned dry-run stub.
            capability_status="real",
            varies_by=["execution_id"],
        ),
        ToolDescriptor(
            name="query_memory",
            description="Retrieve research context and technique memory for a competition.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "include_techniques": {"type": "boolean"},
                },
            },
            output_artifacts=[],
            handler=query_memory,
            capability_status="real",
            varies_by=["query"],
        ),
    ]

"""Catalog of default Research OS tools."""

from __future__ import annotations

from labpilot.research_engine.tools.descriptors import ToolDescriptor
from labpilot.research_engine.tools.handlers import (
    analyze_competition,
    generate_plan,
    query_memory,
    reflect,
    run_plan,
    submit,
    submit_learn,
)
from labpilot.research_engine.tools.registry import ToolRegistry


def build_default_tool_registry() -> ToolRegistry:
    """Return a registry with the M1 stage-capability tools registered."""
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
        ),
    ]

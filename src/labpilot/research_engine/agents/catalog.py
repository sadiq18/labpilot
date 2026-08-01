"""Default specialist catalog — Implementation + Experiment."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.agents.coding import V1CodeEngineeringCodingTool
from labpilot.research_engine.agents.events import EventEmitter
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.implementation import ImplementationSpecialist
from labpilot.research_engine.agents.models import SpecialistDescriptor
from labpilot.research_engine.agents.registry import SpecialistRegistry


def build_default_specialist_registry(
    *,
    llm_client: Any | None = None,
    on_event: EventEmitter | None = None,
    dry_run_default: bool = True,
) -> SpecialistRegistry:
    """Register Implementation + Experiment specialists for Conductor routing."""
    coding = V1CodeEngineeringCodingTool(llm_client=llm_client)
    impl = ImplementationSpecialist(coding, on_event=on_event)
    exp = ExperimentSpecialist(
        dry_run_default=dry_run_default,
        on_event=on_event,
        llm_client=llm_client,
    )
    registry = SpecialistRegistry()
    registry.register(
        SpecialistDescriptor(
            name=impl.name,
            capabilities=list(impl.capabilities),
            required_tools=[],
            input_artifacts=["workspace", "context"],
            output_artifacts=["code"],
            cost_hint=5.0,
            duration_hint=120.0,
            agent=impl,
        )
    )
    registry.register(
        SpecialistDescriptor(
            name=exp.name,
            capabilities=list(exp.capabilities),
            required_tools=["run_plan"],
            input_artifacts=["research_plan", "code"],
            output_artifacts=["execution", "metrics", "experiment"],
            cost_hint=10.0,
            duration_hint=600.0,
            agent=exp,
        )
    )
    return registry

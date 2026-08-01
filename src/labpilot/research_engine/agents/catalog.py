"""Default specialist catalog — Implementation + Experiment."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.agents.coding import V1CodeEngineeringCodingTool
from labpilot.research_engine.agents.events import EventBus, EventEmitter, default_event_bus
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.implementation import ImplementationSpecialist
from labpilot.research_engine.agents.models import SpecialistDescriptor
from labpilot.research_engine.agents.registry import SpecialistRegistry
from labpilot.research_engine.agents.subscribers import install_default_subscribers


def build_default_specialist_registry(
    *,
    llm_client: Any | None = None,
    on_event: EventEmitter | None = None,
    dry_run_default: bool = True,
    bus: EventBus | None = None,
    install_subscribers: bool = True,
) -> SpecialistRegistry:
    """Register Implementation + Experiment specialists for Conductor routing.

    When ``on_event`` is omitted, specialists publish on ``bus`` (default Blinker
    bus). Default subscribers are installed unless ``install_subscribers`` is false.
    """
    if on_event is None:
        active_bus = bus or default_event_bus()
        if install_subscribers:
            install_default_subscribers(active_bus)
        emitter: EventEmitter = active_bus.publish
    else:
        emitter = on_event

    coding = V1CodeEngineeringCodingTool(llm_client=llm_client)
    impl = ImplementationSpecialist(coding, on_event=emitter)
    exp = ExperimentSpecialist(
        dry_run_default=dry_run_default,
        on_event=emitter,
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

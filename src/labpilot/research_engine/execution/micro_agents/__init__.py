"""Execution Platform Micro Agents (design §2.4).

Optional reasoning slice for the run/reflect side (M2). Shares the
``MicroAgent`` contract via :mod:`labpilot.accessor.common.micro_agents`; agents fall
back to a deterministic ``rule_engine`` with no API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from labpilot.accessor.common.micro_agents import BaseMicroAgent, MicroAgent, StructuredContext
from labpilot.research_engine.reflection.critic.micro_agent import (
    ReflectionDraft,
    ReflectionGeneratorAgent,
    RootCauseAgent,
)

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

_AGENT_TYPES: tuple[type[BaseMicroAgent], ...] = (RootCauseAgent,)

_AGENTS_BY_NAME: dict[str, type[BaseMicroAgent]] = {
    RootCauseAgent.name: RootCauseAgent,
    "ReflectionGeneratorAgent": RootCauseAgent,
}


class UnknownMicroAgentError(KeyError):
    """Raised when a Micro Agent name is not registered."""


def available_agents() -> list[str]:
    return sorted(_AGENTS_BY_NAME)


def get_agent(name: str, *, llm_client: LLMClient | None = None) -> MicroAgent:
    try:
        cls = _AGENTS_BY_NAME[name]
    except KeyError as exc:
        raise UnknownMicroAgentError(
            f"Unknown Micro Agent '{name}'. Available: {', '.join(available_agents())}"
        ) from exc
    return cls(llm_client=llm_client)


__all__ = [
    "BaseMicroAgent",
    "MicroAgent",
    "ReflectionDraft",
    "ReflectionGeneratorAgent",
    "RootCauseAgent",
    "StructuredContext",
    "UnknownMicroAgentError",
    "available_agents",
    "get_agent",
]

"""Research Intelligence Micro Agents (design §2.4).

Optional reasoning slice of the platform: each agent maps a typed
``StructuredContext`` to a typed Pydantic artifact. All agents work with **no
API key** via a deterministic ``rule_engine`` fallback, so the pipeline never
hard-depends on an LLM.

Use :func:`get_agent` / :func:`build_agents` to construct agents; pass
``llm_client=None`` (the default) to force the deterministic path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from labpilot.accessor.common.micro_agents import BaseMicroAgent, MicroAgent, StructuredContext
from labpilot.research_engine.intelligence.micro_agents.competition_page_analyzer import (
    CompetitionPageAnalyzerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.concept_normalizer import (
    ConceptNormalizerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.experiment_reviewer import (
    ExperimentReviewerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.forum_analyzer import ForumAnalyzerAgent
from labpilot.research_engine.intelligence.micro_agents.hypothesis_generator import (
    HypothesisGeneratorAgent,
)
from labpilot.research_engine.intelligence.micro_agents.intent_classifier import (
    IntentClassifierAgent,
)
from labpilot.research_engine.intelligence.micro_agents.paper_analyzer import PaperAnalyzerAgent
from labpilot.research_engine.intelligence.micro_agents.repository_analyzer import (
    RepositoryAnalyzerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.repo_query_planner import (
    RepoQueryPlannerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.research_brief import (
    ResearchBriefAgent,
)

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

_AGENT_TYPES: tuple[type[BaseMicroAgent], ...] = (
    PaperAnalyzerAgent,
    RepoQueryPlannerAgent,
    RepositoryAnalyzerAgent,
    ForumAnalyzerAgent,
    HypothesisGeneratorAgent,
    ConceptNormalizerAgent,
    ExperimentReviewerAgent,
    CompetitionPageAnalyzerAgent,
    IntentClassifierAgent,
    ResearchBriefAgent,
)

_AGENTS_BY_NAME: dict[str, type[BaseMicroAgent]] = {cls.name: cls for cls in _AGENT_TYPES}


class UnknownMicroAgentError(KeyError):
    """Raised when a Micro Agent name is not registered."""


def available_agents() -> list[str]:
    """Sorted names of every registered Research Intelligence Micro Agent."""
    return sorted(_AGENTS_BY_NAME)


def get_agent(name: str, *, llm_client: LLMClient | None = None) -> MicroAgent:
    """Construct one Micro Agent by name.

    ``llm_client=None`` (default) selects the deterministic ``rule_engine``
    path — no network, no API key.
    """
    try:
        cls = _AGENTS_BY_NAME[name]
    except KeyError as exc:
        raise UnknownMicroAgentError(
            f"Unknown Micro Agent '{name}'. Available: {', '.join(available_agents())}"
        ) from exc
    return cls(llm_client=llm_client)


def build_agents(*, llm_client: LLMClient | None = None) -> dict[str, MicroAgent]:
    """Instantiate every registered agent sharing one (optional) LLM client."""
    return {name: cls(llm_client=llm_client) for name, cls in _AGENTS_BY_NAME.items()}


__all__ = [
    "BaseMicroAgent",
    "CompetitionPageAnalyzerAgent",
    "ConceptNormalizerAgent",
    "ExperimentReviewerAgent",
    "ForumAnalyzerAgent",
    "HypothesisGeneratorAgent",
    "IntentClassifierAgent",
    "MicroAgent",
    "PaperAnalyzerAgent",
    "RepoQueryPlannerAgent",
    "RepositoryAnalyzerAgent",
    "ResearchBriefAgent",
    "StructuredContext",
    "UnknownMicroAgentError",
    "available_agents",
    "build_agents",
    "get_agent",
]

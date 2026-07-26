"""Assemble a compact, structured planning context from retrieved material.

Deterministic normalization only — turns a :class:`RetrievedContext` into the
fields the templates (and, later, the Planning Engine) actually read. No LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from labpilot.experiments.models import Hypothesis
from labpilot.research_engine.planner.retrieval import RetrievedContext


@dataclass
class PlanningContext:
    hypothesis: Hypothesis
    goal: str
    current_state: str
    expected_outcome: str
    tags: list[str] = field(default_factory=list)
    technique_names: list[str] = field(default_factory=list)
    belief_summaries: list[str] = field(default_factory=list)
    brief_excerpt: str = ""

    @property
    def keywords(self) -> set[str]:
        """Lowercased tokens for template selection (tags + technique names)."""
        tokens: set[str] = set()
        for tag in self.tags:
            tokens.add(tag.strip().lower())
        for name in self.technique_names:
            tokens.add(name.strip().lower())
        return {token for token in tokens if token}


def build_context(retrieved: RetrievedContext) -> PlanningContext:
    hypothesis = retrieved.hypothesis
    goal = hypothesis.prediction or hypothesis.observation
    technique_names = [
        str(row.get("name", "")).strip()
        for row in retrieved.techniques
        if str(row.get("name", "")).strip()
    ]
    belief_summaries = [
        str(row.get("technique", "")).strip()
        for row in retrieved.beliefs
        if str(row.get("technique", "")).strip()
    ]
    return PlanningContext(
        hypothesis=hypothesis,
        goal=goal,
        current_state=hypothesis.observation,
        expected_outcome=hypothesis.prediction,
        tags=list(hypothesis.tags),
        technique_names=technique_names,
        belief_summaries=belief_summaries,
        brief_excerpt=retrieved.brief_excerpt,
    )

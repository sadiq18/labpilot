"""Assemble a compact, structured planning context from retrieved material.

Deterministic normalization only — turns a :class:`RetrievedContext` into the
fields the templates (and, later, the Planning Engine) actually read. No LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.models import Hypothesis
from labpilot.research_engine.planner.parent import resolve_parent_context
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
    parent_hypothesis_id: str | None = None
    parent_execution_id: str | None = None
    parent_metrics: dict[str, Any] = field(default_factory=dict)
    parent_actual_outcome: str | None = None
    technique: str | None = None
    technique_stack: list[str] = field(default_factory=list)
    combo_techniques: list[str] = field(default_factory=list)
    change_category: str = "other"
    parent_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def keywords(self) -> set[str]:
        """Lowercased tokens for template selection (tags + technique names)."""
        tokens: set[str] = set()
        for tag in self.tags:
            tokens.add(tag.strip().lower())
        for name in self.technique_names:
            tokens.add(name.strip().lower())
        if self.technique:
            tokens.add(self.technique.strip().lower())
        if self.change_category:
            tokens.add(self.change_category.strip().lower())
        return {token for token in tokens if token}


def build_context(
    retrieved: RetrievedContext,
    *,
    knowledge_dir: Path | None = None,
    competition: str | None = None,
) -> PlanningContext:
    hypothesis = retrieved.hypothesis
    technique = hypothesis.technique
    stack = list(hypothesis.technique_stack)
    combo = list(hypothesis.combo_techniques)
    parent_meta: dict[str, Any] = {}
    if knowledge_dir is not None:
        parent_meta = resolve_parent_context(
            hypothesis,
            knowledge_dir=knowledge_dir,
            competition=competition or hypothesis.competition,
        )
        technique = technique or parent_meta.get("technique")
        if not stack:
            stack = list(parent_meta.get("technique_stack") or [])
        if not combo:
            combo = list(parent_meta.get("combo_techniques") or [])

    parent_id = parent_meta.get("parent_hypothesis_id") or hypothesis.parent_hypothesis_id
    if combo:
        tech_label = " + ".join(combo)
    else:
        tech_label = technique or (hypothesis.tags[0] if hypothesis.tags else "the change")
    if parent_id:
        goal = (
            f"Improve on {parent_id} by applying {tech_label}; "
            f"beat prior metrics {parent_meta.get('parent_metrics') or 'baseline'}."
        )
        current_state = (
            f"Prior hypothesis {parent_id} "
            f"(outcome={parent_meta.get('parent_actual_outcome') or 'known'}); "
            f"stack=[{', '.join(stack) or 'baseline'}]. "
            f"Now test: {hypothesis.observation}"
        )
        expected_outcome = (
            f"{hypothesis.prediction} Compare against parent {parent_id} metrics, "
            "not only an abstract baseline."
        )
    else:
        goal = hypothesis.prediction or hypothesis.observation
        current_state = hypothesis.observation
        expected_outcome = hypothesis.prediction

    technique_names = [
        str(row.get("name", "")).strip()
        for row in retrieved.techniques
        if str(row.get("name", "")).strip()
    ]
    if technique and technique not in technique_names:
        technique_names.insert(0, technique)
    for member in combo:
        if member and member not in technique_names:
            technique_names.append(member)
    belief_summaries = [
        str(row.get("technique", "")).strip()
        for row in retrieved.beliefs
        if str(row.get("technique", "")).strip()
    ]
    return PlanningContext(
        hypothesis=hypothesis,
        goal=goal,
        current_state=current_state,
        expected_outcome=expected_outcome,
        tags=list(hypothesis.tags),
        technique_names=technique_names,
        belief_summaries=belief_summaries,
        brief_excerpt=retrieved.brief_excerpt,
        parent_hypothesis_id=parent_id,
        parent_execution_id=parent_meta.get("parent_execution_id"),
        parent_metrics=dict(parent_meta.get("parent_metrics") or {}),
        parent_actual_outcome=parent_meta.get("parent_actual_outcome"),
        technique=technique,
        technique_stack=stack,
        combo_techniques=combo,
        change_category=str(parent_meta.get("change_category") or "other"),
        parent_metadata=parent_meta,
    )

"""Retrieval models — Intent, QueryPlan, hits, compressed cards, ResearchContext.

Plan 9: multi-stage retrieval feeds typed context to reasoning (Plan 10). The LLM
never sees SQLite; it only sees serialized ``ResearchContext`` after compression.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryType(StrEnum):
    """Fixed Phase-1 query kinds. Rich planner stays Future."""

    HYPOTHESIS_GENERATION = "hypothesis_generation"
    STRUCTURED_QUERY = "structured_query"
    EXPLAIN = "explain"
    COMPARE = "compare"


class RetrievalIntent(BaseModel):
    """Stage 1 output — structured intent for Symbolic Retrieval."""

    task: str | None = None
    dataset: str | None = None
    domain: str | None = None
    goal: str | None = None
    metric: str | None = None
    query_type: QueryType = QueryType.HYPOTHESIS_GENERATION
    need_experiments: bool = True
    need_papers: bool = True
    need_repositories: bool = True
    need_forums: bool = False
    current_pipeline: list[str] = Field(default_factory=list)
    question: str = ""
    classified_by: str = "rules"  # rules | llm | mixed


class QueryPlan(BaseModel):
    """Fixed Phase-1 plan per ``query_type`` (stub planner; no adaptive optimizer)."""

    query_type: QueryType
    tables: list[str] = Field(default_factory=list)
    traversals: list[str] = Field(default_factory=list)
    use_embeddings: bool = False
    limits: dict[str, int] = Field(default_factory=dict)
    compress: bool = True
    reasoning_agent: str | None = None
    rounds: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    """One symbolic/expanded evidence item."""

    kind: Literal["paper", "experiment", "repository", "discussion", "failure", "technique"]
    document_id: str | None = None
    label: str
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    axes_matched: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    why: str = ""
    summary: str = ""


class TechniqueCard(BaseModel):
    """Compressed technique card (~80 tokens target) — Stage 5 output unit."""

    id: str
    name: str
    confidence: float = 0.5
    category: str = ""
    domain: str = ""
    evidence_labels: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    repository_ids: list[str] = Field(default_factory=list)
    benefits: str = ""
    known_issues: str = ""
    belief_status: str = ""
    failure_notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        evidence = " · ".join(self.evidence_labels) or "none"
        lines = [
            f"Technique: {self.name}",
            f"Evidence: {evidence}",
        ]
        if self.benefits:
            lines.append(f"Benefits: {self.benefits}")
        if self.known_issues:
            lines.append(f"Known Issues: {self.known_issues}")
        if self.belief_status:
            lines.append(f"Belief: {self.belief_status}")
        if self.failure_notes:
            lines.append(f"Relevant Failures: {'; '.join(self.failure_notes)}")
        lines.append(f"Confidence: {self.confidence:.2f}")
        return "\n".join(lines)


class SymbolicBundle(BaseModel):
    """Stage 2+4 intermediate: techniques plus expanded evidence (pre-compress)."""

    techniques: list[dict[str, Any]] = Field(default_factory=list)
    papers: list[RetrievalHit] = Field(default_factory=list)
    experiments: list[RetrievalHit] = Field(default_factory=list)
    repositories: list[RetrievalHit] = Field(default_factory=list)
    discussions: list[RetrievalHit] = Field(default_factory=list)
    failures: list[RetrievalHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchContext(BaseModel):
    """Typed context for Reasoning — serialize to prompt; never dump L4."""

    competition: dict[str, Any] = Field(default_factory=dict)
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    techniques: list[dict[str, Any]] = Field(default_factory=list)
    papers: list[dict[str, Any]] = Field(default_factory=list)
    repositories: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    question: str = ""
    step: str | None = None
    intent: RetrievalIntent | None = None
    brief: str = ""
    budget: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


#: Hierarchical memory token targets (design §5d) and char safety caps (C = T × 3.5).
CHARS_PER_TOKEN = 3.5
L1_TOKEN_BUDGET = 200
L2_TOKEN_BUDGET = 1000
L3_TOKEN_BUDGET = 3000
L1_CHAR_BUDGET = int(L1_TOKEN_BUDGET * CHARS_PER_TOKEN)
L2_CHAR_BUDGET = int(L2_TOKEN_BUDGET * CHARS_PER_TOKEN)
L3_CHAR_BUDGET = int(L3_TOKEN_BUDGET * CHARS_PER_TOKEN)
TOTAL_CHAR_BUDGET = L1_CHAR_BUDGET + L2_CHAR_BUDGET + L3_CHAR_BUDGET

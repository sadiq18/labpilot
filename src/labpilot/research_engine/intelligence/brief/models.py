"""Research Brief models — researcher's briefing document before experiments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchBriefNarrative(BaseModel):
    """LLM / rule_engine prose slices over structured brief inputs."""

    problem_summary: str = ""
    key_risks: list[str] = Field(default_factory=list)
    recommended_focus: str = ""


class ResearchBrief(BaseModel):
    """Concise AI-generated summary of everything known before experimentation."""

    problem_summary: str = ""
    dataset_overview: str = ""
    rules_and_metric: str = ""
    related_papers: list[str] = Field(default_factory=list)
    similar_competitions: list[str] = Field(default_factory=list)
    repositories: list[str] = Field(default_factory=list)
    winning_techniques: list[str] = Field(default_factory=list)
    beliefs: list[str] = Field(default_factory=list)
    top_hypotheses: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    suggested_experiments: list[str] = Field(default_factory=list)
    generated_by: str = "rule_engine"  # llm | rule_engine
    notes: list[str] = Field(default_factory=list)

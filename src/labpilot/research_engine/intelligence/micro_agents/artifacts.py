"""Typed artifacts Micro Agents emit (design §2.4 "LLM as a structured
reasoning engine").

These are the *scaffold* shapes: minimal, valid, and stable enough for later
plans to wire real prompts against. Plans 6–10 refine / extend them (e.g. a
richer ``PaperKnowledge``); extract modules there may thin-wrap these agents.
The contract is only that every agent's ``run`` returns one of these models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Technique(BaseModel):
    """A named method with supporting evidence — the core knowledge unit."""

    name: str
    category: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ResearchFinding(BaseModel):
    source: str = ""
    finding: str = ""
    applicability: list[str] = Field(default_factory=list)


class PaperExtract(BaseModel):
    """Structured extraction from a single paper (never a free-form summary)."""

    techniques: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)


class RepoExtract(BaseModel):
    """Structured card for a GitHub repository."""

    architecture: str = ""
    components: list[str] = Field(default_factory=list)
    files_worth_reading: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    # Easy | Medium | Hard | unknown
    integration_difficulty: str = "unknown"


class ForumExtract(BaseModel):
    """Structured signals mined from a discussion thread."""

    mistakes: list[str] = Field(default_factory=list)
    discoveries: list[str] = Field(default_factory=list)
    dataset_bugs: list[str] = Field(default_factory=list)
    lb_shakeups: list[str] = Field(default_factory=list)
    ood_notes: list[str] = Field(default_factory=list)


class HypothesisDraft(BaseModel):
    """M3 draft shape — maps to the M2 Hypothesis store when persisted."""

    observation: str = ""
    prediction: str = ""
    rationale: str = ""
    expected_impact: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ConceptNormalization(BaseModel):
    """One canonical concept plus the aliases that collapse into it."""

    canonical: str = ""
    aliases: list[str] = Field(default_factory=list)
    category: str = ""


class ExperimentReview(BaseModel):
    """LLM diagnosis over deterministic comparator inputs."""

    diagnosis: str = ""
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

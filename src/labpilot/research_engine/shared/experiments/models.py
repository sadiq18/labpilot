from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class HypothesisOrigin(StrEnum):
    PAPER = "paper"
    EXPERIMENT = "experiment"
    FORUM = "forum"
    REPOSITORY = "repository"
    COMPETITION = "competition"
    USER = "user"
    MIXED = "mixed"


class HypothesisGenerator(StrEnum):
    LLM = "llm"
    RULE_ENGINE = "rule_engine"
    HUMAN = "human"
    IMPORTED = "imported"


class HypothesisCreatedBy(StrEnum):
    ANALYZE = "analyze"
    REFLECTION = "reflection"
    MANUAL = "manual"
    IMPORT = "import"
    HYPOTHESIZE = "hypothesize"


class HypothesisEvidenceRef(BaseModel):
    kind: HypothesisOrigin | str
    ref: str
    note: str = ""


class Hypothesis(BaseModel):
    """Durable, reusable unit of research intent — independent of any one run.

    Linked experiments are derived by filtering the ExperimentGraph on
    `hypothesis_id` (see `experiments.hypothesis.linked_experiments`), not
    stored on this object.
    """

    id: str
    competition: str
    observation: str
    reason: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: Estimated metric delta if the prediction holds (e.g. 0.015); 0.0 = unknown.
    expected_impact: float = 0.0
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    tags: list[str] = Field(default_factory=list)
    # Deprecated M2 alias — prefer created_by / generator / origin (§12.3).
    source: Literal["manual", "reflection", "llm", "analyze"] = "manual"
    created_by: HypothesisCreatedBy | None = None
    generator: HypothesisGenerator | None = None
    origin: HypothesisOrigin | None = None
    origins: list[HypothesisOrigin] = Field(default_factory=list)
    evidence: list[HypothesisEvidenceRef] = Field(default_factory=list)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    #: Primary technique under test (durable; also usually present in tags).
    technique: str | None = None
    #: Hypothesis this one improves on (stacked / fork lineage).
    parent_hypothesis_id: str | None = None
    #: Techniques already assumed in the pipeline (parent stack + this change).
    technique_stack: list[str] = Field(default_factory=list)
    #: Free-text local / LB outcome narrative after an execution or submit.
    actual_outcome: str | None = None
    #: Public leaderboard score when a submission for this hypothesis scored.
    public_score: float | None = None
    created_at: datetime
    updated_at: datetime


class HypothesisUpdate(BaseModel):
    hypothesis_id: str
    new_status: HypothesisStatus
    note: str = ""


class HypothesisDraft(BaseModel):
    observation: str
    reason: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class StructuredReflection(BaseModel):
    run_id: str
    observation: str
    evidence: list[str] = Field(default_factory=list)
    likely_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_next: list[str] = Field(default_factory=list)
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    new_hypotheses: list[HypothesisDraft] = Field(default_factory=list)
    generated_by: Literal["llm", "template_fallback"]


class Experiment(BaseModel):
    """Read-side view of one run, assembled from its existing artifacts.

    Not a new file written to `runs/<id>/` — there is exactly one writer per
    field elsewhere in the pipeline (manifest, baseline_choice.json, ...);
    this model just aggregates them for display/comparison. See
    docs/milestones/experiment-scientist/plan-1-experiment-graph.md for the full
    design rationale.
    """

    id: str
    competition: str
    status: str
    progress: str
    description: str
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    iteration: int = 0
    hypothesis_id: str | None = None
    git_commit: str | None = None
    template_name: str | None = None
    problem_type: str | None = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    feature_recipes: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    public_score: float | None = None
    runtime_seconds: float | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    reflection_path: str | None = None
    reflection: StructuredReflection | None = None
    report_path: str | None = None
    created_at: datetime


class ChangeCategory(StrEnum):
    MODEL = "model"
    AUGMENTATION = "augmentation"
    TRAINING_STRATEGY = "training_strategy"
    SCHEDULER = "scheduler"
    FEATURE_ENGINEERING = "feature_engineering"
    OTHER = "other"


class ConfigChange(BaseModel):
    category: ChangeCategory
    field: str
    base_value: Any
    compare_value: Any
    label: str


class Verdict(StrEnum):
    WORTH_KEEPING = "worth_keeping"
    NOT_WORTH_KEEPING = "not_worth_keeping"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


class ExperimentComparison(BaseModel):
    base_id: str
    compare_id: str
    primary_metric_key: str | None
    metric_deltas: dict[str, float]
    changes: list[ConfigChange]
    runtime_delta_seconds: float | None
    runtime_delta_pct: float | None
    verdict: Verdict
    verdict_reason: str


class KnowledgeEffect(StrEnum):
    IMPROVES = "improves"
    HURTS = "hurts"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class KnowledgeEntry(BaseModel):
    """Cross-experiment observation about one technique on one metric."""

    technique: str
    metric_key: str
    effect: KnowledgeEffect
    delta_estimate: float
    confidence: float = Field(ge=0.0, le=1.0)
    sample_size: int = 0
    evidence_run_ids: list[str] = Field(default_factory=list)
    updated_at: datetime


class RankedCandidate(BaseModel):
    """Scored backlog item over a proposed Hypothesis (Plan 6)."""

    hypothesis: Hypothesis
    expected_gain: float
    implementation_cost: float
    gpu_cost_seconds: float
    risk: float
    novelty: float
    score: float


class ExperimentReport(BaseModel):
    """Competition-level rollup for terminal report + HTML dashboard (Plan 8)."""

    competition: str
    experiment_count: int
    primary_metric_key: str | None = None
    best_experiment_id: str | None = None
    best_score: float | None = None
    top_discoveries: list[KnowledgeEntry] = Field(default_factory=list)
    known_failures: list[KnowledgeEntry] = Field(default_factory=list)
    best_pipeline: list[Experiment] = Field(default_factory=list)
    recommended_next: RankedCandidate | None = None
    experiments: list[Experiment] = Field(default_factory=list)

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
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    tags: list[str] = Field(default_factory=list)
    source: Literal["manual", "reflection", "llm"] = "manual"
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class Experiment(BaseModel):
    """Read-side view of one run, assembled from its existing artifacts.

    Not a new file written to `runs/<id>/` — there is exactly one writer per
    field elsewhere in the pipeline (manifest, baseline_choice.json, ...);
    this model just aggregates them for display/comparison. See
    docs/milestones/milestone-2/plan-1-experiment-graph.md for the full
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

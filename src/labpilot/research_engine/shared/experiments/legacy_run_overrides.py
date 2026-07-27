"""Legacy Pipeline improvement_plan / training_overrides readers for runs/.

Moved from ``labpilot.improvement.models`` (Plan 9). Graph still loads
historical ``runs/*/improvement_plan.json`` and ``training_overrides.json``.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ImprovementAction(StrEnum):
    RETRAIN = "retrain"
    TUNE_HYPERPARAMS = "tune_hyperparams"
    APPLY_FEATURE_RECIPE = "apply_feature_recipe"


DEFAULT_IMPROVE_STAGES = [
    "generate_code",
    "train_model",
    "evaluate_cv",
    "generate_submission",
    "export_kernel",
    "upload_submission",
    "log_experiment",
    "write_reflection",
    "write_report",
]


class ImprovementPlan(BaseModel):
    parent_run_id: str
    strategy: str  # auto | tune | features
    actions: list[ImprovementAction] = Field(default_factory=list)
    model_params: dict[str, Any] = Field(default_factory=dict)
    feature_recipes: list[str] = Field(default_factory=list)
    stages_to_run: list[str] = Field(default_factory=list)
    rationale: str = ""


class TrainingOverrides(BaseModel):
    model_params: dict[str, Any] = Field(default_factory=dict)
    feature_recipes: list[str] = Field(default_factory=list)
    target_encoding_columns: list[str] = Field(default_factory=list)
    log_numeric_columns: list[str] = Field(default_factory=list)


DEFAULT_TABULAR_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}


def improvement_plan_path(run_dir: Path) -> Path:
    return run_dir / "improvement_plan.json"


def training_overrides_path(run_dir: Path) -> Path:
    return run_dir / "training_overrides.json"


def save_improvement_plan(run_dir: Path, plan: ImprovementPlan) -> Path:
    path = improvement_plan_path(run_dir)
    path.write_text(plan.model_dump_json(indent=2))
    return path


def load_improvement_plan(run_dir: Path) -> ImprovementPlan | None:
    path = improvement_plan_path(run_dir)
    if not path.is_file():
        return None
    return ImprovementPlan.model_validate_json(path.read_text())


def save_training_overrides(run_dir: Path, overrides: TrainingOverrides) -> Path:
    path = training_overrides_path(run_dir)
    path.write_text(overrides.model_dump_json(indent=2))
    return path


def load_training_overrides(run_dir: Path) -> TrainingOverrides:
    path = training_overrides_path(run_dir)
    if not path.is_file():
        return TrainingOverrides()
    return TrainingOverrides.model_validate_json(path.read_text())

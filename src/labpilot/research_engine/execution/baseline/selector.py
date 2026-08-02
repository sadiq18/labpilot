import logging
from pathlib import Path

from pydantic import BaseModel, Field

from labpilot.research_engine.execution.baseline.registry import get_template
from labpilot.research_engine.intelligence.competition.infer_problem_type import (
    infer_problem_type_from_metadata,
)
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec, ProblemType
from labpilot.accessor.profiler.tabular import DatasetProfile

logger = logging.getLogger(__name__)

# P0 has exactly one template per problem type, and each template's generated
# training script always writes a single hardcoded `cv_<metric>` key to
# metrics.json (see templates/*/train.py.j2). The metric a competition
# actually uses on Kaggle (e.g. RMSLE, F1, AUC) is informational only in P0 —
# `_evaluate_cv` must look for the key the template really produces, not
# whatever a competition's metadata happens to say, or evaluation would fail
# for otherwise-correct runs.
DEFAULT_METRIC_BY_PROBLEM_TYPE: dict[str, str] = {
    ProblemType.TABULAR_CLASSIFICATION.value: "accuracy",
    ProblemType.TABULAR_REGRESSION.value: "rmse",
    ProblemType.TEXT_CLASSIFICATION.value: "accuracy",
    ProblemType.IMAGE_CLASSIFICATION.value: "accuracy",
}

SUPPORTED_METRICS_BY_PROBLEM_TYPE: dict[str, set[str]] = {
    ProblemType.TABULAR_CLASSIFICATION.value: {"accuracy", "auc", "logloss", "f1"},
    ProblemType.TEXT_CLASSIFICATION.value: {"accuracy", "auc", "logloss", "f1"},
    ProblemType.IMAGE_CLASSIFICATION.value: {"accuracy", "auc", "logloss", "f1"},
    ProblemType.TABULAR_REGRESSION.value: {"rmse", "mse", "mae", "rmsle"},
}

# A target is treated as classification if it's non-numeric, OR numeric with
# few enough distinct values that they read as class labels rather than a
# continuous quantity (e.g. Titanic's 0/1 `Survived`, stored as int64).
# Requiring at least one repeated value (`unique_count < row_count`) keeps
# small regression datasets where every row happens to have a unique target
# (common with only a handful of rows) from being misread as classification.
MAX_CLASSIFICATION_CARDINALITY = 20


class ValidationPlan(BaseModel):
    """How to split data so local CV *means* what the leaderboard measures.

    Derived from the dataset profile rather than hardcoded per template. A
    shuffled row-level KFold on a partitioned dataset scores a near-duplicate
    of every training row (adjacent rows in a partition are almost identical),
    producing a CV number that is both wildly optimistic and uncorrelated with
    the leaderboard.
    """

    scheme: str = "kfold"  # kfold | group_kfold | partition_suffix_holdout
    group_key: str | None = None
    n_splits: int = 5
    holdout_fraction: float = 0.0
    # Columns present in train but not at inference time. Using them as
    # features trains a model that cannot be served (and usually leaks the
    # target outright).
    exclude_features: list[str] = Field(default_factory=list)
    rationale: str = ""


def derive_validation_plan(profile: DatasetProfile, n_splits: int = 5) -> ValidationPlan:
    """Choose a validation scheme that mirrors the test-time information split."""
    exclude = [c for c in profile.train_only_columns if c != profile.target_column]

    if profile.scored_is_partition_suffix:
        return ValidationPlan(
            scheme="partition_suffix_holdout",
            group_key=profile.partition_key,
            n_splits=n_splits,
            holdout_fraction=profile.scored_fraction or 0.5,
            exclude_features=exclude,
            rationale=(
                "scored rows form a contiguous suffix of each test partition, so "
                "validation holds out each training partition's tail to reproduce "
                "the same predict-forward gap"
            ),
        )
    if profile.partitioned:
        return ValidationPlan(
            scheme="group_kfold",
            group_key=profile.partition_key,
            n_splits=n_splits,
            exclude_features=exclude,
            rationale=(
                "rows are not iid across partitions; grouping prevents "
                "near-duplicate rows from spanning the train/validation boundary"
            ),
        )
    return ValidationPlan(
        scheme="kfold",
        n_splits=n_splits,
        exclude_features=exclude,
        rationale="iid rows — plain KFold is appropriate",
    )


class BaselineChoice(BaseModel):
    problem_type: str
    template_name: str
    rationale: str
    target_column: str | None = None
    id_column: str | None = None
    train_file: str | None = None
    test_file: str | None = None
    sample_submission_file: str | None = None
    submission_columns: list[str] = Field(default_factory=list)
    metric_name: str = "accuracy"
    text_column: str | None = None
    image_dir: str | None = None
    image_column: str | None = None
    baseline_strategy: str = "lightweight"
    validation: ValidationPlan = Field(default_factory=ValidationPlan)
    partitioned: bool = False
    partition_kinds: dict[str, int] = Field(default_factory=dict)


class BaselineSelector:
    """Rule-based baseline template selection for P0."""

    def select(self, competition: CompetitionSpec, profile: DatasetProfile) -> BaselineChoice:
        problem_type = self._infer_problem_type(competition, profile)
        template_name = self._resolve_template_name(problem_type, competition)
        template = get_template(problem_type, template_name=template_name)

        if template is None:
            raise ValueError(f"No baseline template for problem type: {problem_type}")

        metric_name = self._resolve_metric_name(competition, problem_type)
        logger.info(
            "Selected baseline template '%s' for problem type '%s' (metric key: cv_%s).",
            template.name,
            problem_type,
            metric_name,
        )
        return BaselineChoice(
            problem_type=problem_type,
            template_name=template.name,
            rationale=self._rationale(problem_type, profile),
            target_column=profile.target_column,
            id_column=profile.id_column,
            train_file=profile.train_file,
            test_file=profile.test_file,
            sample_submission_file=profile.sample_submission_file,
            submission_columns=profile.submission_columns,
            metric_name=metric_name,
            text_column=profile.text_column,
            image_dir=profile.image_dir,
            image_column=profile.image_column,
            baseline_strategy=competition.baseline_strategy,
            validation=derive_validation_plan(profile),
            partitioned=profile.partitioned,
            partition_kinds=profile.partition_kinds,
        )

    def save(self, run_dir: Path, choice: BaselineChoice) -> Path:
        output = run_dir / "baseline_choice.json"
        output.write_text(choice.model_dump_json(indent=2))
        return output

    def _infer_problem_type(self, competition: CompetitionSpec, profile: DatasetProfile) -> str:
        if competition.problem_type not in (ProblemType.UNKNOWN,):
            return competition.problem_type.value

        metric = competition.evaluation_metric
        from_meta = infer_problem_type_from_metadata(
            title=competition.title,
            description=competition.description,
            tags=list(competition.tags),
            metric_name=(metric.name if metric else ""),
            metric_description=(metric.description if metric else ""),
        )

        # Regression/classification from tags/metric beats incidental images in
        # the inventory (e.g. ROGII: well-log CSVs + PNG previews, MSE metric).
        if from_meta in (
            ProblemType.TABULAR_REGRESSION,
            ProblemType.TABULAR_CLASSIFICATION,
        ):
            return from_meta.value

        # Profile modality is authoritative for clear vision/text layouts.
        if profile.modality == "image":
            return ProblemType.IMAGE_CLASSIFICATION.value
        if profile.modality == "text":
            return ProblemType.TEXT_CLASSIFICATION.value

        if profile.target_column and profile.row_count > 0:
            target = next((c for c in profile.columns if c.name == profile.target_column), None)
            if target:
                looks_like_discrete_labels = (
                    target.unique_count <= MAX_CLASSIFICATION_CARDINALITY
                    and target.unique_count < profile.row_count
                )
                if not target.is_numeric or looks_like_discrete_labels:
                    return ProblemType.TABULAR_CLASSIFICATION.value
                return ProblemType.TABULAR_REGRESSION.value

        if from_meta is not ProblemType.UNKNOWN:
            return from_meta.value

        if profile.row_count > 0 and profile.column_count > 0:
            # Tabular-looking profile without a clear target — classification default.
            return ProblemType.TABULAR_CLASSIFICATION.value

        raise ValueError(
            "Cannot infer problem type: competition.problem_type is unknown, "
            "dataset profile is empty/unusable, and metadata tags/description "
            "do not indicate tabular/text/image. Run prepare_workspace (download + "
            "profile) or set problem_type in configs/competitions/<slug>.yaml."
        )

    def _resolve_metric_name(self, competition: CompetitionSpec, problem_type: str) -> str:
        default = DEFAULT_METRIC_BY_PROBLEM_TYPE.get(problem_type, "accuracy")
        supported = SUPPORTED_METRICS_BY_PROBLEM_TYPE.get(problem_type, {default})
        metric = competition.evaluation_metric
        if metric is None or metric.key is None:
            return default
        if metric.key in supported:
            return metric.key
        logger.info(
            "Competition metric key '%s' is not supported for %s; using default '%s'.",
            metric.key,
            problem_type,
            default,
        )
        return default

    def _resolve_template_name(self, problem_type: str, competition: CompetitionSpec) -> str | None:
        if (
            competition.baseline_strategy == "deep"
            and problem_type
            in {
                ProblemType.TEXT_CLASSIFICATION.value,
                ProblemType.IMAGE_CLASSIFICATION.value,
            }
        ):
            return f"{problem_type}_deep"
        return None

    def _rationale(self, problem_type: str, profile: DatasetProfile) -> str:
        return (
            f"Selected {problem_type} based on competition metadata and "
            f"dataset profile ({profile.row_count} rows, {profile.column_count} columns)."
        )

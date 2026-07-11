import logging
from pathlib import Path

from pydantic import BaseModel, Field

from labpilot.baseline.registry import get_template
from labpilot.competition.models import CompetitionSpec, ProblemType
from labpilot.profiler.tabular import DatasetProfile

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
}

# A target is treated as classification if it's non-numeric, OR numeric with
# few enough distinct values that they read as class labels rather than a
# continuous quantity (e.g. Titanic's 0/1 `Survived`, stored as int64).
# Requiring at least one repeated value (`unique_count < row_count`) keeps
# small regression datasets where every row happens to have a unique target
# (common with only a handful of rows) from being misread as classification.
MAX_CLASSIFICATION_CARDINALITY = 20


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


class BaselineSelector:
    """Rule-based baseline template selection for P0."""

    def select(self, competition: CompetitionSpec, profile: DatasetProfile) -> BaselineChoice:
        problem_type = self._infer_problem_type(competition, profile)
        template = get_template(problem_type)

        if template is None:
            raise ValueError(f"No baseline template for problem type: {problem_type}")

        metric_name = DEFAULT_METRIC_BY_PROBLEM_TYPE.get(problem_type, "accuracy")
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
        )

    def save(self, run_dir: Path, choice: BaselineChoice) -> Path:
        output = run_dir / "baseline_choice.json"
        output.write_text(choice.model_dump_json(indent=2))
        return output

    def _infer_problem_type(self, competition: CompetitionSpec, profile: DatasetProfile) -> str:
        if competition.problem_type not in (ProblemType.UNKNOWN,):
            return competition.problem_type.value

        if profile.target_column:
            target = next((c for c in profile.columns if c.name == profile.target_column), None)
            if target:
                looks_categorical = target.dtype in ("object", "category", "bool")
                looks_like_discrete_labels = (
                    target.unique_count <= MAX_CLASSIFICATION_CARDINALITY
                    and target.unique_count < profile.row_count
                )
                if looks_categorical or looks_like_discrete_labels:
                    return ProblemType.TABULAR_CLASSIFICATION.value
                return ProblemType.TABULAR_REGRESSION.value

        # Default P0 assumption: tabular classification
        return ProblemType.TABULAR_CLASSIFICATION.value

    def _rationale(self, problem_type: str, profile: DatasetProfile) -> str:
        return (
            f"Selected {problem_type} based on competition metadata and "
            f"dataset profile ({profile.row_count} rows, {profile.column_count} columns)."
        )

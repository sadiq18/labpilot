import logging
from pathlib import Path

from pydantic import BaseModel, Field

from labpilot.baseline.registry import get_template
from labpilot.competition.models import CompetitionSpec, ProblemType
from labpilot.profiler.tabular import DatasetProfile

logger = logging.getLogger(__name__)


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
    """Rule-based baseline template selection for P0.

    # TODO: control the verbosity of this class's logging via a future CLI
    # --verbose/--quiet flag (see docs/MILESTONES.md).
    """

    def select(self, competition: CompetitionSpec, profile: DatasetProfile) -> BaselineChoice:
        problem_type = self._infer_problem_type(competition, profile)
        template = get_template(problem_type)

        if template is None:
            raise ValueError(f"No baseline template for problem type: {problem_type}")

        logger.info(
            "Selected baseline template '%s' for problem type '%s'.",
            template.name,
            problem_type,
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
            metric_name=(
                competition.evaluation_metric.name if competition.evaluation_metric else "accuracy"
            ),
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
            if target and target.dtype in ("object", "category", "bool"):
                return ProblemType.TABULAR_CLASSIFICATION.value
            if target:
                return ProblemType.TABULAR_REGRESSION.value

        # Default P0 assumption: tabular classification
        return ProblemType.TABULAR_CLASSIFICATION.value

    def _rationale(self, problem_type: str, profile: DatasetProfile) -> str:
        return (
            f"Selected {problem_type} based on competition metadata and "
            f"dataset profile ({profile.row_count} rows, {profile.column_count} columns)."
        )

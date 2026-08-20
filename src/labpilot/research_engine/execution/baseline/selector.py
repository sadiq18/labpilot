import logging
from pathlib import Path

from pydantic import BaseModel, Field

from labpilot.accessor.profiler.tabular import DatasetProfile
from labpilot.research_engine.execution.baseline.registry import get_template
from labpilot.research_engine.intelligence.competition.infer_problem_type import (
    infer_problem_type_from_metadata,
)
from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    metrics_for_problem_type,
)
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec, ProblemType
from labpilot.research_engine.intelligence.competition.objective import ObjectiveSpec

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

#: Derived from the registry rather than restated. Kept as a module constant
#: because it is imported elsewhere, but every entry now comes from the metric
#: definitions, so a metric added there is supported here without a second edit —
#: and a metric named but not scorable (`balanced_accuracy`) stays out of both.
SUPPORTED_METRICS_BY_PROBLEM_TYPE: dict[str, set[str]] = {
    problem_type.value: set(metrics_for_problem_type(problem_type))
    for problem_type in (
        ProblemType.TABULAR_CLASSIFICATION,
        ProblemType.TEXT_CLASSIFICATION,
        ProblemType.IMAGE_CLASSIFICATION,
        ProblemType.TABULAR_REGRESSION,
    )
}

# A target is treated as classification if it's non-numeric, OR numeric with
# few enough distinct values that they read as class labels rather than a
# continuous quantity (e.g. Titanic's 0/1 `Survived`, stored as int64).
# Requiring at least one repeated value (`unique_count < row_count`) keeps
# small regression datasets where every row happens to have a unique target
# (common with only a handful of rows) from being misread as classification.
#: Retired in favour of `DatasetProfile.target_type`, which draws the same line
#: at 30. Kept as a name for readers of older `baseline_choice.json` files and
#: read by nothing here — two thresholds for one question is how a 25-label
#: target came to be classification or regression depending on which path ran.
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
    # The target's known prefix, when the profiler found one: equal to the
    # target wherever present, absent exactly on the scored rows. Neither a
    # plain feature nor an excluded one — as a feature it learns "copy" and
    # then meets NaN on every row it has to predict, and dropping it discards
    # the strongest signal in the dataset. It is a *baseline to subtract*:
    # carry its last known value forward and model the residual.
    #
    # Named on the plan rather than left in `profile.warnings`, which is where
    # it was: nothing in `src/` read `anchor_column`, so the profiler's finding
    # reached the pipeline only as one sentence in a warnings array that the
    # model was free to skip. A field the validation plan carries is one the
    # baseline choice records and the delta checks can see.
    anchor_column: str | None = None
    rationale: str = ""


def derive_validation_plan(profile: DatasetProfile, n_splits: int = 5) -> ValidationPlan:
    """Choose a validation scheme that mirrors the test-time information split."""
    exclude = [c for c in profile.train_only_columns if c != profile.target_column]
    anchor = profile.anchor_column
    anchor_note = (
        f" {anchor!r} is the target's known prefix: carry its last known value forward "
        f"and model the residual, rather than fitting {profile.target_column!r} from the "
        "other columns. Do not pass it as a plain feature — it equals the target in "
        "training, so the model learns to copy it and then meets NaN on every scored row."
        if anchor
        else ""
    )

    if profile.scored_is_partition_suffix:
        return ValidationPlan(
            scheme="partition_suffix_holdout",
            group_key=profile.partition_key,
            n_splits=n_splits,
            holdout_fraction=profile.scored_fraction or 0.5,
            exclude_features=exclude,
            anchor_column=anchor,
            rationale=(
                "scored rows form a contiguous suffix of each test partition, so "
                "validation holds out each training partition's tail to reproduce "
                "the same predict-forward gap." + anchor_note
            ),
        )
    if profile.partitioned:
        return ValidationPlan(
            scheme="group_kfold",
            group_key=profile.partition_key,
            n_splits=n_splits,
            exclude_features=exclude,
            anchor_column=anchor,
            rationale=(
                "rows are not iid across partitions; grouping prevents "
                "near-duplicate rows from spanning the train/validation boundary." + anchor_note
            ),
        )
    return ValidationPlan(
        scheme="kfold",
        n_splits=n_splits,
        exclude_features=exclude,
        anchor_column=anchor,
        rationale="iid rows — plain KFold is appropriate." + anchor_note,
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
    #: Where `problem_type` and `metric_name` came from: ``objective`` when
    #: `objective.json` answered, ``derived`` when this file worked it out for
    #: itself. Recorded because the two disagreeing is a finding, and until now
    #: there was nothing to compare — the objective was resolved, printed, and
    #: never reached this decision at all.
    objective_source: str = "derived"
    #: The metric the competition is actually scored by, as the objective
    #: resolved it. Kept beside `metric_name` rather than replacing it, because
    #: when they differ that difference is the whole finding.
    objective_metric: str | None = None
    #: Set when `metric_name` is **not** what the competition is scored by. CV
    #: then optimises a proxy, and it used to do so behind a `logger.info` — the
    #: metric-mismatch class this layer exists to remove, arriving one level up.
    #: playground-series-s6e7 states balanced accuracy and every campaign scored
    #: plain accuracy.
    metric_substituted_from: str | None = None
    partitioned: bool = False
    partition_kinds: dict[str, int] = Field(default_factory=dict)


class BaselineSelector:
    """Rule-based baseline template selection for P0."""

    def select(
        self,
        competition: CompetitionSpec,
        profile: DatasetProfile,
        objective: ObjectiveSpec | None = None,
    ) -> BaselineChoice:
        """Choose a baseline. `objective` is the stage before this one.

        Optional so every existing caller keeps working, and passed on the
        production path. Without it this file re-derives the task from the
        target's shape and the metric from the contract — two more
        implementations of questions `objective.json` has already answered with
        evidence, and two more chances to answer them differently.
        """
        problem_type = self._infer_problem_type(competition, profile, objective)
        task_from_objective = bool(
            objective is not None and objective.task and objective.task == problem_type
        )
        template_name = self._resolve_template_name(problem_type, competition)
        # A partitioned predict-forward dataset cannot be served by the plain
        # single-train-file template: it would read one partition and validate
        # on shuffled rows.
        if (
            profile.partitioned
            and problem_type == ProblemType.TABULAR_REGRESSION.value
            and template_name is None
        ):
            template_name = "tabular_regression_partitioned"
        template = get_template(problem_type, template_name=template_name)

        if template is None:
            raise ValueError(f"No baseline template for problem type: {problem_type}")

        metric_name, substituted_from = self._resolve_metric_name(
            competition, problem_type, objective
        )
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
            # What the objective actually supplied, not merely that one was
            # passed. An unresolved objective — no task, no metric — reported
            # `objective` while both values came from the older derivation,
            # which is the wrong answer to the one question this field exists
            # to answer.
            objective_source=(
                "objective"
                if task_from_objective or (objective is not None and objective.metric_name)
                else "derived"
            ),
            objective_metric=objective.metric_name if objective is not None else None,
            metric_substituted_from=substituted_from,
        )

    def save(self, run_dir: Path, choice: BaselineChoice) -> Path:
        output = run_dir / "baseline_choice.json"
        output.write_text(choice.model_dump_json(indent=2))
        return output

    def _infer_problem_type(
        self,
        competition: CompetitionSpec,
        profile: DatasetProfile,
        objective: ObjectiveSpec | None = None,
    ) -> str:
        # The objective's task is measured over the resolved target (M23 step 1).
        # Everything below is the older answer to the same question: a keyword
        # match on the description, then a cardinality rule this file keeps its
        # own copy of. They agree on the easy cases, and where they do not, the
        # one that looked at the column wins.
        #
        # Only when it names a type with a template. `image_regression` is a
        # perfectly honest task string with nothing to build for it, and
        # returning it here would raise past a caller that reads the exception as
        # "defer to the LLM" — turning an improvement into a silent loss of
        # rule-based selection.
        if objective is not None and objective.task:
            known = {member.value for member in ProblemType} - {ProblemType.UNKNOWN.value}
            if objective.task in known:
                return objective.task

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

        # `profile.target_type`, not a second cardinality rule. This file kept
        # its own — `MAX_CLASSIFICATION_CARDINALITY = 20` against the profiler's
        # 30 — so a target with 25 labels was classification if `objective.json`
        # existed and regression if it did not. Two implementations of one
        # question is what step 2 exists to remove, and leaving the copy here
        # with a different number was the worst of both.
        #
        # `target_type` is a derived field, so it is available on every profile
        # including ones written before it existed; there is nothing to fall
        # back to a local rule *for*.
        by_shape = {
            "binary": ProblemType.TABULAR_CLASSIFICATION.value,
            "multiclass": ProblemType.TABULAR_CLASSIFICATION.value,
            "multilabel": ProblemType.TABULAR_CLASSIFICATION.value,
            "continuous": ProblemType.TABULAR_REGRESSION.value,
            "count": ProblemType.TABULAR_REGRESSION.value,
        }.get(profile.target_type)
        if by_shape is not None:
            return by_shape

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

    def _resolve_metric_name(
        self,
        competition: CompetitionSpec,
        problem_type: str,
        objective: ObjectiveSpec | None = None,
    ) -> tuple[str, str | None]:
        """`(metric_to_optimise, the_one_it_replaced)`.

        The second element is the point. A metric the pipeline cannot compute
        used to become a default behind a `logger.info`, so CV optimised a proxy
        and nothing downstream could tell. Returning what was displaced makes the
        substitution a field on the artifact, which the evidence layer and the
        gate can both read.
        """
        default = DEFAULT_METRIC_BY_PROBLEM_TYPE.get(problem_type, "accuracy")
        supported = SUPPORTED_METRICS_BY_PROBLEM_TYPE.get(problem_type, {default})
        # The objective resolved this from six ranked sources and a probe, and
        # the contract is one of those sources. Reading the contract again here
        # is how the two came to disagree.
        key = objective.metric_name if objective is not None else None
        if key is None:
            metric = competition.evaluation_metric
            key = metric.key if metric is not None else None
        if key is None:
            return default, None
        if key in supported:
            return key, None
        logger.info(
            "Competition metric key '%s' is not supported for %s; using default '%s'.",
            key,
            problem_type,
            default,
        )
        return default, key

    def _resolve_template_name(self, problem_type: str, competition: CompetitionSpec) -> str | None:
        if competition.baseline_strategy == "deep" and problem_type in {
            ProblemType.TEXT_CLASSIFICATION.value,
            ProblemType.IMAGE_CLASSIFICATION.value,
        }:
            return f"{problem_type}_deep"
        return None

    def _rationale(self, problem_type: str, profile: DatasetProfile) -> str:
        return (
            f"Selected {problem_type} based on competition metadata and "
            f"dataset profile ({profile.row_count} rows, {profile.column_count} columns)."
        )

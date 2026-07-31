"""Deterministic rule_engine templates that turn context into a task blueprint.

Templates are the offline / CI path: they emit a fully-formed task graph without
any LLM, mirroring the soft-fail posture of the Research Brief. Each template
returns a :class:`PlanBlueprint` of :class:`TaskSpec` nodes keyed by local ids;
the driver lowers those into real ``ResearchTask`` rows with allocated ids.

The classic worked example (README §3) — "add SpecAugment" — maps to the
augmentation template: read → modify code → config → unit test → smoke train →
1-epoch train → evaluate → compare → (gated) full train → report → belief →
reflect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from labpilot.research_engine.planner.context_builder import PlanningContext
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: Tokens that route a hypothesis to the augmentation template.
_AUGMENTATION_KEYWORDS = {
    "augmentation",
    "augment",
    "specaugment",
    "spec augment",
    "mixup",
    "cutout",
    "cutmix",
    "time mask",
    "freq mask",
}

_FE_KEYWORDS = {
    "feature_engineering",
    "feature engineering",
    "target_encoding",
    "target encoding",
    "one_hot",
    "one-hot",
    "tfidf",
    "tf-idf",
    "binning",
    "aggregation",
    "feature_interactions",
    "lag_features",
}

_MODEL_KEYWORDS = {
    "model",
    "xgboost",
    "lightgbm",
    "catboost",
    "resnet",
    "efficientnet",
    "transformer",
    "architecture",
    "backbone",
}


@dataclass
class TaskSpec:
    """A template node keyed by a local id; deps reference other local ids."""

    key: str
    type: TaskType
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    check: str = ""
    expected_output: str = ""
    failure_recovery: str = ""
    max_retries: int = 0
    abort_on_failure: bool = True


@dataclass
class PlanBlueprint:
    goal: str
    current_state: str
    expected_outcome: str
    tasks: list[TaskSpec]
    risk: str = ""
    success_criteria: list[str] = field(default_factory=list)
    rollback: str = ""
    artifacts: list[str] = field(default_factory=list)
    template_name: str = "generic"


def _baseline_template(competition: str, *, brief_excerpt: str = "") -> PlanBlueprint:
    """Full baseline spine (P-001) — workspace through report/reflect."""
    tasks = [
        TaskSpec(
            key="workspace",
            type=TaskType.PREPARE_WORKSPACE,
            description="Create isolated workspace dirs and verify layout.",
            outputs=["workspace_layout"],
            check="Workspace directories exist and are writable.",
        ),
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description="Inspect registry / problem-type baseline candidates.",
            inputs=["workspace_layout"],
            outputs=["baseline_notes"],
            depends_on=["workspace"],
            check="Baseline selection inputs understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description="Scaffold train/infer pipeline for the baseline.",
            inputs=["baseline_notes"],
            outputs=["train_script"],
            depends_on=["read"],
            check="Baseline scaffold implemented.",
            failure_recovery="Revert scaffold via git.",
        ),
        TaskSpec(
            key="config",
            type=TaskType.MODIFY_CONFIG,
            description="Write baseline training/inference config.",
            inputs=["config.yaml"],
            outputs=["config.yaml"],
            depends_on=["write"],
            check="Config loads successfully.",
        ),
        TaskSpec(
            key="review",
            type=TaskType.RESEARCH_REVIEW,
            description="Research-correctness gate on the baseline scaffold.",
            inputs=["train_script", "config.yaml"],
            outputs=["review_findings"],
            depends_on=["config"],
            check="No critical research-correctness findings.",
        ),
        TaskSpec(
            key="deps",
            type=TaskType.INSTALL_PACKAGE,
            description="Install required packages from requirements/lockfile.",
            inputs=["requirements.txt"],
            outputs=["env_pins"],
            depends_on=["review"],
            check="Packages install and import.",
            max_retries=1,
        ),
        TaskSpec(
            key="unit",
            type=TaskType.RUN_UNIT_TEST,
            description="Run unit tests for the baseline scaffold.",
            inputs=["tests/"],
            outputs=["unit_test_report"],
            depends_on=["deps"],
            check="Exit 0; required tests pass.",
        ),
        TaskSpec(
            key="smoke",
            type=TaskType.RUN_SMOKE_TEST,
            description="Production-shaped smoke gate before full training.",
            inputs=["config.yaml"],
            outputs=["smoke_log"],
            depends_on=["unit"],
            check="Tiny-batch / 1-epoch smoke completes without crash.",
            failure_recovery="Fix via WRITE_CODE/MODIFY_CONFIG, or abort.",
        ),
        TaskSpec(
            key="runtime",
            type=TaskType.SELECT_RUNTIME,
            description="Select and provision the runtime for full training.",
            inputs=["smoke_log"],
            outputs=["runtime_target"],
            depends_on=["smoke"],
            check="Runtime target selected and recorded.",
        ),
        TaskSpec(
            key="train",
            type=TaskType.RUN_TRAINING,
            description="Budgeted full baseline training.",
            inputs=["config.yaml", "runtime_target"],
            outputs=["run_dir"],
            depends_on=["runtime"],
            check="Loss decreases (or metrics are finite).",
            max_retries=2,
        ),
        TaskSpec(
            key="infer",
            type=TaskType.RUN_INFERENCE,
            description="Run inference for evaluation / submission inputs.",
            inputs=["run_dir"],
            outputs=["predictions"],
            depends_on=["train"],
            check="Predictions produced.",
        ),
        TaskSpec(
            key="evaluate",
            type=TaskType.EVALUATE,
            description="Evaluate baseline predictions; record metrics.",
            inputs=["predictions", "run_dir"],
            outputs=["metrics"],
            depends_on=["infer"],
            check="Validation metrics recorded.",
        ),
        TaskSpec(
            key="submit",
            type=TaskType.BUILD_SUBMISSION,
            description="Package submission artifact (upload gated by config).",
            inputs=["predictions"],
            outputs=["submission"],
            depends_on=["evaluate"],
            check="Submission artifact matches expected format.",
        ),
        TaskSpec(
            key="report",
            type=TaskType.GENERATE_REPORT,
            description="Write the baseline experiment report.",
            inputs=["metrics", "submission"],
            outputs=["report.md"],
            depends_on=["submit"],
            check="Report written.",
        ),
        TaskSpec(
            key="reflect",
            type=TaskType.REFLECT,
            description="Structured reflection on the baseline outcome.",
            inputs=["report.md"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured.",
        ),
        TaskSpec(
            key="belief",
            type=TaskType.UPDATE_BELIEF,
            description="Record baseline beliefs for later hypothesis plans.",
            inputs=["metrics", "reflection"],
            outputs=["belief_update"],
            depends_on=["reflect"],
            check="Belief store updated.",
        ),
    ]
    excerpt_note = ""
    if brief_excerpt:
        # Prefer a short complete clause — never trail off with an ellipsis.
        snippet = " ".join(brief_excerpt.split())
        if len(snippet) > 120:
            cut = snippet[:120].rsplit(" ", 1)[0].rstrip(".,;:")
            snippet = cut if cut else snippet[:120].rstrip(".,;:")
        if snippet and snippet[-1] not in ".!?":
            snippet = f"{snippet}."
        excerpt_note = f" Brief: {snippet}"
    return PlanBlueprint(
        goal=f"Establish a verified baseline experiment for {competition}.{excerpt_note}",
        current_state="Analyze complete; no baseline experiment yet.",
        expected_outcome="Smoke-gated train/eval/submit with durable metrics and report.",
        tasks=tasks,
        risk="Baseline may be weak for the problem type; still establishes the floor.",
        success_criteria=[
            "Smoke gate passes before full training.",
            "Metrics and submission artifact recorded.",
            "Report and reflection written.",
        ],
        rollback="Abandon execution; keep Analyze artifacts intact.",
        artifacts=["report.md", "metrics", "submission"],
        template_name="baseline",
    )


def select_template(context: PlanningContext) -> PlanBlueprint:
    """Pick the most specific matching template for the context."""
    keywords = context.keywords
    text = f"{context.goal} {context.current_state} {context.expected_outcome}".lower()
    category = (context.change_category or "").lower()
    if (
        category == "feature_engineering"
        or keywords & _FE_KEYWORDS
        or any(kw in text for kw in _FE_KEYWORDS)
    ):
        return _feature_engineering_template(context)
    if keywords & _AUGMENTATION_KEYWORDS or any(
        kw in text for kw in _AUGMENTATION_KEYWORDS
    ):
        return _augmentation_template(context)
    if (
        category == "model"
        or keywords & _MODEL_KEYWORDS
        or any(kw in text for kw in _MODEL_KEYWORDS)
    ):
        return _model_template(context)
    return _generic_template(context)


def _technique_label(context: PlanningContext) -> str:
    if context.combo_techniques:
        return " + ".join(context.combo_techniques)
    return context.technique or (
        context.technique_names[0] if context.technique_names else "the proposed change"
    )


def _parent_compare_desc(context: PlanningContext) -> str:
    parent = context.parent_hypothesis_id or "baseline"
    metrics = context.parent_metrics
    if metrics:
        return f"Compare metrics against parent {parent} ({metrics}), not only abstract baseline."
    return f"Compare metrics against parent {parent} / prior best run, not only abstract baseline."


def _improve_read_desc(context: PlanningContext) -> str:
    tech = _technique_label(context)
    parent = context.parent_hypothesis_id
    if parent:
        return (
            f"Read the prior pipeline for {parent} (stack="
            f"[{', '.join(context.technique_stack) or 'baseline'}]); "
            f"identify where to apply technique `{tech}` as a delta."
        )
    return f"Inspect code paths relevant to applying technique `{tech}`."


def _improve_write_desc(context: PlanningContext) -> str:
    tech = _technique_label(context)
    parent = context.parent_hypothesis_id
    if context.combo_techniques:
        members = ", ".join(f"`{t}`" for t in context.combo_techniques)
        if parent:
            return (
                f"Override train.py: keep what worked from {parent} and apply ALL "
                f"combination techniques in one delta: {members} "
                "(single improve-on-prior experiment, not sequential singles)."
            )
        return (
            f"Implement combination techniques in one WRITE_CODE delta: {members}."
        )
    if parent:
        return (
            f"Override train.py: keep what worked from {parent} and add technique "
            f"`{tech}` as an incremental improvement (not an independent baseline)."
        )
    return f"Implement technique `{tech}` in the training pipeline."


def _feature_engineering_template(context: PlanningContext) -> PlanBlueprint:
    tech = _technique_label(context)
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description=_improve_read_desc(context),
            outputs=["fe_notes"],
            check="Prior pipeline and feature entry points understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description=(
                f"{_improve_write_desc(context)} Focus on feature recipe for `{tech}` "
                "(name, inputs, outputs, transform)."
            ),
            inputs=["fe_notes"],
            outputs=["train_script"],
            depends_on=["read"],
            check=f"Feature engineering `{tech}` applied on prior pipeline.",
            failure_recovery="Restore prior train.py from artifacts/code_backups.",
        ),
        TaskSpec(
            key="unit",
            type=TaskType.RUN_UNIT_TEST,
            description=f"Run unit tests covering feature changes for `{tech}`.",
            inputs=["tests/"],
            outputs=["unit_test_report"],
            depends_on=["write"],
            check="Exit 0; required tests pass.",
        ),
        TaskSpec(
            key="smoke",
            type=TaskType.RUN_SMOKE_TEST,
            description="Short smoke run after FE change.",
            outputs=["smoke_log"],
            depends_on=["unit"],
            check="Pipeline runs on a tiny sample without error.",
        ),
        TaskSpec(
            key="train",
            type=TaskType.RUN_TRAINING,
            description=f"Train to test FE technique `{tech}` on the improved pipeline.",
            inputs=["config.yaml"],
            outputs=["run_dir"],
            depends_on=["smoke"],
            check="Loss decreases (or metrics are finite).",
            max_retries=2,
        ),
        TaskSpec(
            key="evaluate",
            type=TaskType.EVALUATE,
            description="Evaluate the run on the validation split.",
            inputs=["run_dir"],
            outputs=["metrics"],
            depends_on=["train"],
            check="Validation metrics recorded.",
        ),
        TaskSpec(
            key="compare",
            type=TaskType.COMPARE,
            description=_parent_compare_desc(context),
            inputs=["metrics", "parent_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs parent / prior best.",
        ),
        TaskSpec(
            key="report",
            type=TaskType.GENERATE_REPORT,
            description=f"Report outcome of FE `{tech}` vs parent.",
            inputs=["comparison"],
            outputs=["report.md"],
            depends_on=["compare"],
            check="Report written.",
        ),
        TaskSpec(
            key="reflect",
            type=TaskType.REFLECT,
            description="Reflect: what helped, what to stack next.",
            inputs=["report.md", "comparison"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or f"Improve prior pipeline with FE `{tech}`.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="New features may overfit or leak; compare carefully vs parent.",
        success_criteria=[
            "Smoke run completes without error.",
            "Validation metric improves over parent / prior best.",
        ],
        rollback="Restore train.py from artifacts/code_backups.",
        artifacts=["report.md", "comparison"],
        template_name="feature_engineering",
    )


def _model_template(context: PlanningContext) -> PlanBlueprint:
    tech = _technique_label(context)
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description=_improve_read_desc(context),
            outputs=["model_notes"],
            check="Prior model/training entry points understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description=f"{_improve_write_desc(context)} Swap/adjust model toward `{tech}`.",
            inputs=["model_notes"],
            outputs=["train_script"],
            depends_on=["read"],
            check=f"Model change `{tech}` applied on prior pipeline.",
            failure_recovery="Restore prior train.py from artifacts/code_backups.",
        ),
        TaskSpec(
            key="config",
            type=TaskType.MODIFY_CONFIG,
            description=f"Update config for model technique `{tech}`.",
            inputs=["config.yaml"],
            outputs=["config.yaml"],
            depends_on=["write"],
            check="Config loads successfully.",
        ),
        TaskSpec(
            key="unit",
            type=TaskType.RUN_UNIT_TEST,
            description="Run unit tests covering the model change.",
            inputs=["tests/"],
            outputs=["unit_test_report"],
            depends_on=["config"],
            check="Exit 0; required tests pass.",
        ),
        TaskSpec(
            key="smoke",
            type=TaskType.RUN_SMOKE_TEST,
            description="Short smoke run after model change.",
            outputs=["smoke_log"],
            depends_on=["unit"],
            check="Pipeline runs on a tiny sample without error.",
        ),
        TaskSpec(
            key="train",
            type=TaskType.RUN_TRAINING,
            description=f"Train to test model technique `{tech}`.",
            inputs=["config.yaml"],
            outputs=["run_dir"],
            depends_on=["smoke"],
            check="Loss decreases (or metrics are finite).",
            max_retries=2,
        ),
        TaskSpec(
            key="evaluate",
            type=TaskType.EVALUATE,
            description="Evaluate the run on the validation split.",
            inputs=["run_dir"],
            outputs=["metrics"],
            depends_on=["train"],
            check="Validation metrics recorded.",
        ),
        TaskSpec(
            key="compare",
            type=TaskType.COMPARE,
            description=_parent_compare_desc(context),
            inputs=["metrics", "parent_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs parent / prior best.",
        ),
        TaskSpec(
            key="report",
            type=TaskType.GENERATE_REPORT,
            description=f"Report outcome of model `{tech}` vs parent.",
            inputs=["comparison"],
            outputs=["report.md"],
            depends_on=["compare"],
            check="Report written.",
        ),
        TaskSpec(
            key="reflect",
            type=TaskType.REFLECT,
            description="Reflect: keep model change or stack further techniques.",
            inputs=["report.md", "comparison"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or f"Improve prior pipeline with model `{tech}`.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="Model change may raise cost without gain; gate on parent compare.",
        success_criteria=[
            "Smoke run completes without error.",
            "Validation metric improves over parent / prior best.",
        ],
        rollback="Restore train.py from artifacts/code_backups.",
        artifacts=["report.md", "comparison"],
        template_name="model",
    )


def _augmentation_template(context: PlanningContext) -> PlanBlueprint:
    tech = _technique_label(context)
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description=_improve_read_desc(context),
            inputs=["augmentation.py"],
            outputs=["augmentation_notes"],
            check="Relevant augmentation code located and understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description=f"{_improve_write_desc(context)} Add augmentation `{tech}`.",
            inputs=["augmentation.py", "augmentation_notes"],
            outputs=["augmentation.py"],
            depends_on=["read"],
            check="Change requested; new augmentation implemented in a WRITE_CODE task.",
            failure_recovery="Revert to previous augmentation.py via git.",
        ),
        TaskSpec(
            key="config",
            type=TaskType.MODIFY_CONFIG,
            description="Enable/parameterize the new augmentation in config.",
            inputs=["config.yaml"],
            outputs=["config.yaml"],
            depends_on=["write"],
            check="Config loads successfully.",
            failure_recovery="Restore previous config version.",
        ),
        TaskSpec(
            key="unit",
            type=TaskType.RUN_UNIT_TEST,
            description="Run unit tests covering the augmentation.",
            inputs=["tests/"],
            outputs=["unit_test_report"],
            depends_on=["write"],
            check="Exit 0; required tests pass.",
            failure_recovery="Open a follow-up WRITE_CODE fix task, or abort.",
        ),
        TaskSpec(
            key="smoke",
            type=TaskType.RUN_SMOKE_TEST,
            description="Short smoke run to catch integration errors early.",
            inputs=["config.yaml"],
            outputs=["smoke_log"],
            depends_on=["config", "unit"],
            check="Pipeline runs end-to-end on a tiny sample without error.",
            failure_recovery="Fix via WRITE_CODE/MODIFY_CONFIG, or abort.",
        ),
        TaskSpec(
            key="train_smoke",
            type=TaskType.RUN_TRAINING,
            description="Train 1 epoch to sanity-check learning dynamics.",
            inputs=["config.yaml"],
            outputs=["run_dir"],
            depends_on=["smoke"],
            check="Loss decreases (or metrics are finite).",
            failure_recovery="Abort after N failures.",
            max_retries=2,
        ),
        TaskSpec(
            key="evaluate",
            type=TaskType.EVALUATE,
            description="Evaluate the 1-epoch run on the validation split.",
            inputs=["run_dir"],
            outputs=["metrics"],
            depends_on=["train_smoke"],
            check="Validation metrics recorded.",
        ),
        TaskSpec(
            key="compare",
            type=TaskType.COMPARE,
            description=_parent_compare_desc(context),
            inputs=["metrics", "parent_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs parent / prior best.",
            failure_recovery="Mark inconclusive; skip the gated full-training task.",
        ),
        TaskSpec(
            key="train_full",
            type=TaskType.RUN_TRAINING,
            description=(
                "Gated: continue to full training only if the comparison shows "
                "improvement (see plan success_criteria)."
            ),
            inputs=["config.yaml", "comparison"],
            outputs=["run_dir_full"],
            depends_on=["compare"],
            check="Only runs when success criteria are met; loss/metrics improve.",
            failure_recovery="Abort after N failures.",
            max_retries=2,
        ),
        TaskSpec(
            key="report",
            type=TaskType.GENERATE_REPORT,
            description=f"Report augmentation `{tech}` outcome vs parent.",
            inputs=["comparison", "run_dir_full"],
            outputs=["report.md"],
            depends_on=["compare"],
            check="Report written summarizing outcome vs hypothesis.",
        ),
        TaskSpec(
            key="belief",
            type=TaskType.UPDATE_BELIEF,
            description="Update beliefs about this augmentation from the evidence.",
            inputs=["comparison"],
            outputs=["belief_update"],
            depends_on=["compare"],
            check="Belief store reflects the observed effect.",
        ),
        TaskSpec(
            key="reflect",
            type=TaskType.REFLECT,
            description="Structured reflection on results and next stack steps.",
            inputs=["report.md", "comparison"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured with suggested next hypotheses.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or f"Improve prior pipeline with augmentation `{tech}`.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="Augmentation may not transfer; training cost if kept without gain.",
        success_criteria=[
            "Smoke training completes without error.",
            "1-epoch validation metric improves over parent before full training.",
        ],
        rollback="Revert augmentation.py and config changes via git.",
        artifacts=["report.md", "comparison"],
        template_name="augmentation",
    )


def _generic_template(context: PlanningContext) -> PlanBlueprint:
    tech = _technique_label(context)
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description=_improve_read_desc(context),
            outputs=["notes"],
            check="Relevant code located and understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description=_improve_write_desc(context),
            inputs=["notes"],
            outputs=["code_change"],
            depends_on=["read"],
            check=f"Technique `{tech}` implemented as improve-on-prior WRITE_CODE.",
            failure_recovery="Restore prior train.py from artifacts/code_backups.",
        ),
        TaskSpec(
            key="unit",
            type=TaskType.RUN_UNIT_TEST,
            description="Run unit tests covering the change.",
            inputs=["tests/"],
            outputs=["unit_test_report"],
            depends_on=["write"],
            check="Exit 0; required tests pass.",
            failure_recovery="Open a follow-up WRITE_CODE fix task, or abort.",
        ),
        TaskSpec(
            key="smoke",
            type=TaskType.RUN_SMOKE_TEST,
            description="Short smoke run to catch integration errors early.",
            outputs=["smoke_log"],
            depends_on=["unit"],
            check="Pipeline runs end-to-end on a tiny sample without error.",
            failure_recovery="Fix via WRITE_CODE, or abort.",
        ),
        TaskSpec(
            key="train",
            type=TaskType.RUN_TRAINING,
            description=f"Train to test technique `{tech}` on the improved pipeline.",
            inputs=["config.yaml"],
            outputs=["run_dir"],
            depends_on=["smoke"],
            check="Loss decreases (or metrics are finite).",
            failure_recovery="Abort after N failures.",
            max_retries=2,
        ),
        TaskSpec(
            key="evaluate",
            type=TaskType.EVALUATE,
            description="Evaluate the run on the validation split.",
            inputs=["run_dir"],
            outputs=["metrics"],
            depends_on=["train"],
            check="Validation metrics recorded.",
        ),
        TaskSpec(
            key="compare",
            type=TaskType.COMPARE,
            description=_parent_compare_desc(context),
            inputs=["metrics", "parent_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs parent / prior best.",
            failure_recovery="Mark inconclusive.",
        ),
        TaskSpec(
            key="report",
            type=TaskType.GENERATE_REPORT,
            description="Generate a human-readable report of the experiment.",
            inputs=["comparison"],
            outputs=["report.md"],
            depends_on=["compare"],
            check="Report written summarizing outcome vs hypothesis.",
        ),
        TaskSpec(
            key="reflect",
            type=TaskType.REFLECT,
            description="Structured reflection: what helped and what to stack next.",
            inputs=["report.md"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured with suggested next hypotheses.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or f"Improve prior pipeline with `{tech}`.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="Change may not improve the metric; costs training time.",
        success_criteria=[
            "Smoke run completes without error.",
            "Validation metric improves over parent / prior best.",
        ],
        rollback="Restore train.py from artifacts/code_backups.",
        artifacts=["report.md", "comparison"],
        template_name="generic",
    )

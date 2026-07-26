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


def select_template(context: PlanningContext) -> PlanBlueprint:
    """Pick the most specific matching template for the context."""
    keywords = context.keywords
    text = f"{context.goal} {context.current_state} {context.expected_outcome}".lower()
    if keywords & _AUGMENTATION_KEYWORDS or any(
        kw in text for kw in _AUGMENTATION_KEYWORDS
    ):
        return _augmentation_template(context)
    return _generic_template(context)


def _augmentation_template(context: PlanningContext) -> PlanBlueprint:
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description="Inspect the augmentation pipeline and current transforms.",
            inputs=["augmentation.py"],
            outputs=["augmentation_notes"],
            check="Relevant augmentation code located and understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description="Add the proposed augmentation to the pipeline.",
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
            description="Compare metrics against the baseline run.",
            inputs=["metrics", "baseline_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs baseline.",
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
            description="Generate a human-readable report of the experiment.",
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
            description="Structured reflection on results and next steps.",
            inputs=["report.md", "comparison"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured with suggested next hypotheses.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or "Apply the proposed augmentation change.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="Augmentation may not transfer; training cost if kept without gain.",
        success_criteria=[
            "Smoke training completes without error.",
            "1-epoch validation metric improves over baseline before full training.",
        ],
        rollback="Revert augmentation.py and config changes via git.",
        artifacts=["report.md", "comparison"],
        template_name="augmentation",
    )


def _generic_template(context: PlanningContext) -> PlanBlueprint:
    tasks = [
        TaskSpec(
            key="read",
            type=TaskType.READ_CODE,
            description="Inspect the code paths relevant to the hypothesis.",
            outputs=["notes"],
            check="Relevant code located and understood.",
        ),
        TaskSpec(
            key="write",
            type=TaskType.WRITE_CODE,
            description="Implement the change implied by the hypothesis.",
            inputs=["notes"],
            outputs=["code_change"],
            depends_on=["read"],
            check="Change implemented in a WRITE_CODE task.",
            failure_recovery="Revert changes via git.",
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
            description="Train to test the hypothesis.",
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
            description="Compare metrics against the baseline run.",
            inputs=["metrics", "baseline_metrics"],
            outputs=["comparison"],
            depends_on=["evaluate"],
            check="Metric delta recorded vs baseline.",
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
            description="Structured reflection on results and next steps.",
            inputs=["report.md"],
            outputs=["reflection"],
            depends_on=["report"],
            check="Reflection captured with suggested next hypotheses.",
        ),
    ]
    return PlanBlueprint(
        goal=context.goal or "Test the hypothesis with a controlled experiment.",
        current_state=context.current_state,
        expected_outcome=context.expected_outcome,
        tasks=tasks,
        risk="Change may not improve the metric; costs training time.",
        success_criteria=[
            "Smoke run completes without error.",
            "Validation metric improves over baseline.",
        ],
        rollback="Revert code and config changes via git.",
        artifacts=["report.md", "comparison"],
        template_name="generic",
    )

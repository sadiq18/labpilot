"""The planning compiler driver.

Turns a :class:`Hypothesis` into a validated, persisted :class:`ResearchPlan`
DAG through a fail-safe pipeline (Option B)::

    hypothesis → retrieval → context → template baseline (rule_engine)
      → [optional] Planning Engine LLM revises slim draft (ONE call)
      → lower (ids, defaults) → validate → optimize → schedule
      → on LLM/DAG failure: keep baseline
      → PlanStore.upsert_plan → serializer JSON/MD

``generated_by`` reflects the origin of the **final validated** plan
(``llm`` | ``rule_engine``), not the attempted path. With ``llm_client=None``
behavior is identical to the Plan 3 offline path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from labpilot.accessor.common.ids import task_id as make_task_id
from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.shared.experiments.models import Hypothesis
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner import optimizer, scheduler, serializer
from labpilot.research_engine.planner.context_builder import PlanningContext, build_context
from labpilot.research_engine.planner.micro_agents.planning_engine import ResearchPlannerAgent
from labpilot.research_engine.planner.retrieval import retrieve
from labpilot.research_engine.planner.schemas.draft import DraftTask, ResearchPlanDraft
from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
    TaskVerification,
)
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskStatus
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.templates import (
    PlanBlueprint,
    _baseline_template,
    select_template,
)
from labpilot.research_engine.planner.validator import PlanValidationError, validate_plan

logger = logging.getLogger(__name__)

class BaselinePlanError(ValueError):
    """Baseline compile refused (missing Analyze, plan already exists, …)."""


def compile_research_plan(
    hypothesis: Hypothesis,
    *,
    knowledge_dir: Path,
    competition: str | None = None,
    llm_client: Any | None = None,
    plan_store: PlanStore | None = None,
    knowledge_store: Any | None = None,
    write_projections: bool = True,
    priority: int = 0,
) -> ResearchPlan:
    """Compile and persist a research plan for ``hypothesis``."""
    competition = competition or hypothesis.competition

    store = plan_store or PlanStore(knowledge_dir, competition)
    owns_store = plan_store is None
    try:
        retrieved = retrieve(
            hypothesis,
            knowledge_dir=knowledge_dir,
            competition=competition,
            knowledge_store=knowledge_store,
        )
        context = build_context(retrieved)
        blueprint = select_template(context)
        plan_id = store.new_plan_id()

        baseline_draft = blueprint_to_draft(blueprint)
        baseline = _finalize_plan(
            lower_draft(
                baseline_draft,
                plan_id=plan_id,
                hypothesis=hypothesis,
                competition=competition,
                generated_by="rule_engine",
                metadata={"template": blueprint.template_name},
                priority=priority,
            )
        )

        plan = baseline
        if llm_client is not None:
            plan = _try_llm_revision(
                baseline=baseline,
                baseline_draft=baseline_draft,
                context=context,
                hypothesis=hypothesis,
                competition=competition,
                plan_id=plan_id,
                template_name=blueprint.template_name,
                llm_client=llm_client,
                priority=priority,
            )

        store.upsert_plan(plan)
        if write_projections:
            serializer.write_projections(
                plan, knowledge_dir=knowledge_dir, competition=competition
            )
        return plan
    finally:
        if owns_store:
            store.close()


def compile_baseline_plan(
    competition: str,
    *,
    knowledge_dir: Path,
    llm_client: Any | None = None,
    plan_store: PlanStore | None = None,
    write_projections: bool = True,
    priority: int = 0,
) -> ResearchPlan:
    """Compile P-001 baseline plan from Analyze context (no hypothesis)."""
    paths = ResearchPaths(knowledge_dir, competition)
    if not paths.report_path.is_file():
        raise BaselinePlanError(
            f"Analyze context missing: expected {paths.report_path}. "
            f"Run `research analyze {competition}` first."
        )

    store = plan_store or PlanStore(knowledge_dir, competition)
    owns_store = plan_store is None
    try:
        existing = store.list_plans()
        if existing:
            raise BaselinePlanError(
                f"Competition {competition} already has {len(existing)} plan(s); "
                "baseline must be the first plan (P-001)."
            )

        brief = ""
        if paths.brief_path.is_file():
            brief = paths.brief_path.read_text(encoding="utf-8")[:2000]

        blueprint = _baseline_template(competition, brief_excerpt=brief)
        plan_id = store.new_plan_id()
        if plan_id != "P-001":
            raise BaselinePlanError(
                f"Expected first plan id P-001, got {plan_id}."
            )

        baseline_draft = blueprint_to_draft(blueprint)
        plan = _finalize_plan(
            lower_draft(
                baseline_draft,
                plan_id=plan_id,
                competition=competition,
                hypothesis_id="",
                generated_by="rule_engine",
                metadata={
                    "template": blueprint.template_name,
                    "plan_kind": "baseline",
                },
                priority=priority,
            )
        )

        # Optional LLM revision: soft-fail keeps rule_engine baseline.
        if llm_client is not None:
            plan = _try_baseline_llm_revision(
                baseline=plan,
                baseline_draft=baseline_draft,
                competition=competition,
                plan_id=plan_id,
                brief=brief,
                llm_client=llm_client,
                priority=priority,
            )
            # Preserve plan_kind even if LLM revised metadata incompletely.
            plan.metadata = {
                **dict(plan.metadata),
                "plan_kind": "baseline",
                "template": blueprint.template_name,
            }

        store.upsert_plan(plan)
        if write_projections:
            serializer.write_projections(
                plan, knowledge_dir=knowledge_dir, competition=competition
            )
        return plan
    finally:
        if owns_store:
            store.close()


def _try_baseline_llm_revision(
    *,
    baseline: ResearchPlan,
    baseline_draft: ResearchPlanDraft,
    competition: str,
    plan_id: str,
    brief: str,
    llm_client: Any,
    priority: int = 0,
) -> ResearchPlan:
    agent = ResearchPlannerAgent(llm_client=llm_client)
    structured = StructuredContext(
        competition=competition,
        question=baseline.goal,
        text=brief,
        data={
            "baseline_draft": baseline_draft.model_dump(mode="json"),
            "plan_kind": "baseline",
            "hypothesis_id": "",
            "goal": baseline.goal,
            "current_state": baseline.current_state,
            "expected_outcome": baseline.expected_outcome,
            "brief_excerpt": brief[:1000],
        },
    )
    draft = agent.run(structured)
    if not agent.last_used_llm:
        return baseline
    try:
        return _finalize_plan(
            lower_draft(
                draft,
                plan_id=plan_id,
                competition=competition,
                hypothesis_id="",
                generated_by="llm",
                metadata={
                    "template": "baseline",
                    "plan_kind": "baseline",
                    "revised_by": "llm",
                },
                priority=priority,
            )
        )
    except (PlanValidationError, ValueError, TypeError) as exc:
        logger.warning(
            "Baseline LLM revision for %s failed (%s); keeping rule_engine.",
            plan_id,
            exc,
        )
        baseline.notes = list(baseline.notes) + [
            f"LLM revision rejected; kept rule_engine baseline ({exc})."
        ]
        return baseline


def blueprint_to_draft(blueprint: PlanBlueprint) -> ResearchPlanDraft:
    """Convert a template blueprint into the slim draft shape."""
    return ResearchPlanDraft(
        goal=blueprint.goal,
        current_state=blueprint.current_state,
        expected_outcome=blueprint.expected_outcome,
        risk=blueprint.risk,
        success_criteria=list(blueprint.success_criteria),
        rollback=blueprint.rollback,
        artifacts=list(blueprint.artifacts),
        tasks=[
            DraftTask(
                key=spec.key,
                type=spec.type,
                description=spec.description,
                inputs=list(spec.inputs),
                outputs=list(spec.outputs),
                depends_on=list(spec.depends_on),
            )
            for spec in blueprint.tasks
        ],
    )


def lower_draft(
    draft: ResearchPlanDraft,
    *,
    plan_id: str,
    competition: str,
    hypothesis: Hypothesis | None = None,
    hypothesis_id: str = "",
    estimated_gain: float = 0.0,
    generated_by: Literal["llm", "rule_engine"] = "rule_engine",
    metadata: dict[str, Any] | None = None,
    priority: int = 0,
) -> ResearchPlan:
    """Allocate ids/timestamps and build a ``ResearchPlan`` from a slim draft."""
    now = datetime.now(UTC)
    if not draft.tasks:
        raise PlanValidationError("draft has no tasks")
    id_by_key: dict[str, str] = {
        task.key: make_task_id(plan_id, index + 1)
        for index, task in enumerate(draft.tasks)
    }
    missing = [
        dep
        for task in draft.tasks
        for dep in task.depends_on
        if dep not in id_by_key
    ]
    if missing:
        raise PlanValidationError(f"draft depends_on unknown keys: {sorted(set(missing))}")

    hyp_id = hypothesis.id if hypothesis is not None else hypothesis_id
    gain = hypothesis.expected_impact if hypothesis is not None else estimated_gain

    tasks: list[ResearchTask] = []
    for index, task in enumerate(draft.tasks):
        tasks.append(
            ResearchTask(
                id=id_by_key[task.key],
                plan_id=plan_id,
                type=task.type,
                description=task.description,
                inputs=list(task.inputs),
                outputs=list(task.outputs),
                dependencies=[id_by_key[dep] for dep in task.depends_on],
                status=TaskStatus.PENDING,
                order=index,
                verification=TaskVerification(),
                retry_policy=RetryPolicy(),
            )
        )
    return ResearchPlan(
        id=plan_id,
        competition=competition,
        hypothesis_id=hyp_id,
        goal=draft.goal,
        current_state=draft.current_state,
        expected_outcome=draft.expected_outcome,
        status=PlanStatus.READY,
        priority=priority,
        estimated_gain=gain,
        risk=draft.risk,
        success_criteria=list(draft.success_criteria),
        artifacts=list(draft.artifacts),
        rollback=draft.rollback,
        generated_by=generated_by,
        metadata=dict(metadata or {}),
        tasks=tasks,
        created_at=now,
        updated_at=now,
    )


def _finalize_plan(plan: ResearchPlan) -> ResearchPlan:
    optimizer.apply_defaults(plan)
    validate_plan(plan)
    scheduler.schedule(plan)
    return plan


def _try_llm_revision(
    *,
    baseline: ResearchPlan,
    baseline_draft: ResearchPlanDraft,
    context: PlanningContext,
    hypothesis: Hypothesis,
    competition: str,
    plan_id: str,
    template_name: str,
    llm_client: Any,
    priority: int = 0,
) -> ResearchPlan:
    """One Planning Engine call; keep baseline unless a revised draft validates."""
    agent = ResearchPlannerAgent(llm_client=llm_client)
    structured = _planning_structured_context(context, hypothesis, baseline_draft)
    draft = agent.run(structured)

    if not agent.last_used_llm:
        logger.info(
            "Planning Engine soft-fell back to rule_engine for %s; keeping baseline.",
            plan_id,
        )
        return baseline

    try:
        candidate = _finalize_plan(
            lower_draft(
                draft,
                plan_id=plan_id,
                hypothesis=hypothesis,
                competition=competition,
                generated_by="llm",
                metadata={"template": template_name, "revised_by": "llm"},
                priority=priority,
            )
        )
    except (PlanValidationError, ValueError, TypeError) as exc:
        logger.warning(
            "Planning Engine draft for %s failed validation (%s); keeping baseline.",
            plan_id,
            exc,
        )
        baseline.notes = list(baseline.notes) + [
            f"LLM revision rejected; kept rule_engine baseline ({exc})."
        ]
        return baseline

    return candidate


def _planning_structured_context(
    context: PlanningContext,
    hypothesis: Hypothesis,
    baseline_draft: ResearchPlanDraft,
) -> StructuredContext:
    text_parts = []
    if context.brief_excerpt:
        text_parts.append(context.brief_excerpt)
    if context.technique_names:
        text_parts.append("Techniques: " + ", ".join(context.technique_names))
    if context.belief_summaries:
        text_parts.append("Beliefs: " + ", ".join(context.belief_summaries))
    return StructuredContext(
        competition=hypothesis.competition,
        question=context.goal or hypothesis.prediction,
        text="\n".join(text_parts),
        data={
            "baseline_draft": baseline_draft.model_dump(mode="json"),
            "hypothesis_id": hypothesis.id,
            "observation": hypothesis.observation,
            "reason": hypothesis.reason,
            "prediction": hypothesis.prediction,
            "expected_impact": hypothesis.expected_impact,
            "confidence": hypothesis.confidence,
            "tags": list(hypothesis.tags),
            "goal": context.goal,
            "current_state": context.current_state,
            "expected_outcome": context.expected_outcome,
            "technique_names": list(context.technique_names),
            "belief_summaries": list(context.belief_summaries),
            "brief_excerpt": context.brief_excerpt,
        },
    )

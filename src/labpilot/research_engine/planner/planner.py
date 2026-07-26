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

from labpilot.accessor.commons.ids import task_id as make_task_id
from labpilot.common.micro_agents import StructuredContext
from labpilot.experiments.models import Hypothesis
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
from labpilot.research_engine.planner.templates import PlanBlueprint, select_template
from labpilot.research_engine.planner.validator import PlanValidationError, validate_plan

logger = logging.getLogger(__name__)


def compile_research_plan(
    hypothesis: Hypothesis,
    *,
    knowledge_dir: Path,
    competition: str | None = None,
    llm_client: Any | None = None,
    plan_store: PlanStore | None = None,
    knowledge_store: Any | None = None,
    write_projections: bool = True,
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
    hypothesis: Hypothesis,
    competition: str,
    generated_by: Literal["llm", "rule_engine"] = "rule_engine",
    metadata: dict[str, Any] | None = None,
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
        hypothesis_id=hypothesis.id,
        goal=draft.goal,
        current_state=draft.current_state,
        expected_outcome=draft.expected_outcome,
        status=PlanStatus.READY,
        estimated_gain=hypothesis.expected_impact,
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

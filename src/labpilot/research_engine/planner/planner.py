"""The planning compiler driver.

Turns a :class:`Hypothesis` into a validated, persisted :class:`ResearchPlan`
DAG through a deterministic pipeline:

    hypothesis -> retrieval -> context -> template (rule_engine)
      -> lower (ids, defaults) -> validate -> optimize -> schedule
      -> PlanStore.upsert_plan -> serializer JSON/MD

The LLM is an *optional* upgrade layered on top of this path (Plan 4); with
``llm_client=None`` the compiler still produces a valid plan offline / in CI.
The planner never writes source, configs, or ``runs/`` — it only emits nodes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.accessor.commons.ids import task_id as make_task_id
from labpilot.experiments.models import Hypothesis
from labpilot.research_engine.planner import optimizer, scheduler, serializer
from labpilot.research_engine.planner.context_builder import build_context
from labpilot.research_engine.planner.retrieval import retrieve
from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
    TaskVerification,
)
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskStatus
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.templates import PlanBlueprint, select_template
from labpilot.research_engine.planner.validator import validate_plan


def compile_research_plan(
    hypothesis: Hypothesis,
    *,
    knowledge_dir: Path,
    competition: str | None = None,
    llm_client: Any | None = None,  # noqa: ARG001 - Plan 4 wires the Planning Engine
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
        plan = _lower(blueprint, plan_id, hypothesis, competition)

        optimizer.apply_defaults(plan)
        validate_plan(plan)
        scheduler.schedule(plan)

        store.upsert_plan(plan)
        if write_projections:
            serializer.write_projections(
                plan, knowledge_dir=knowledge_dir, competition=competition
            )
        return plan
    finally:
        if owns_store:
            store.close()


def _lower(
    blueprint: PlanBlueprint,
    plan_id: str,
    hypothesis: Hypothesis,
    competition: str,
) -> ResearchPlan:
    now = datetime.now(UTC)
    id_by_key: dict[str, str] = {
        spec.key: make_task_id(plan_id, index + 1)
        for index, spec in enumerate(blueprint.tasks)
    }
    tasks: list[ResearchTask] = []
    for index, spec in enumerate(blueprint.tasks):
        tasks.append(
            ResearchTask(
                id=id_by_key[spec.key],
                plan_id=plan_id,
                type=spec.type,
                description=spec.description,
                inputs=list(spec.inputs),
                outputs=list(spec.outputs),
                dependencies=[id_by_key[dep] for dep in spec.depends_on],
                status=TaskStatus.PENDING,
                order=index,
                verification=TaskVerification(
                    expected_output=spec.expected_output,
                    check=spec.check,
                    failure_recovery=spec.failure_recovery,
                ),
                retry_policy=RetryPolicy(
                    max_retries=spec.max_retries,
                    abort_on_failure=spec.abort_on_failure,
                ),
            )
        )
    return ResearchPlan(
        id=plan_id,
        competition=competition,
        hypothesis_id=hypothesis.id,
        goal=blueprint.goal,
        current_state=blueprint.current_state,
        expected_outcome=blueprint.expected_outcome,
        status=PlanStatus.READY,
        estimated_gain=hypothesis.expected_impact,
        risk=blueprint.risk,
        success_criteria=list(blueprint.success_criteria),
        artifacts=list(blueprint.artifacts),
        rollback=blueprint.rollback,
        generated_by="rule_engine",
        metadata={"template": blueprint.template_name},
        tasks=tasks,
        created_at=now,
        updated_at=now,
    )

"""DAG validation and topological ordering for a ResearchPlan.

Deterministic, LLM-free: the compiler runs this after lowering to reject plans
with unknown ids, dangling dependencies, or cycles, and to compute the parallel
execution levels used by the scheduler.
"""

from __future__ import annotations

from labpilot.research_engine.planner.schemas.models import ResearchPlan


class PlanValidationError(ValueError):
    """Raised when a plan is not a valid DAG."""


def validate_plan(plan: ResearchPlan) -> None:
    """Raise :class:`PlanValidationError` unless ``plan`` is a valid DAG."""
    ids: list[str] = [task.id for task in plan.tasks]
    seen: set[str] = set()
    duplicates = {task_id for task_id in ids if task_id in seen or seen.add(task_id)}
    if duplicates:
        raise PlanValidationError(
            f"duplicate task ids: {sorted(duplicates)}"
        )

    id_set = set(ids)
    for task in plan.tasks:
        for dep in task.dependencies:
            if dep not in id_set:
                raise PlanValidationError(
                    f"task {task.id!r} depends on unknown task {dep!r}"
                )
            if dep == task.id:
                raise PlanValidationError(f"task {task.id!r} depends on itself")
        if task.parent_task_id is not None and task.parent_task_id not in id_set:
            raise PlanValidationError(
                f"task {task.id!r} has unknown parent {task.parent_task_id!r}"
            )

    # Kahn's algorithm — if any node remains, there is a cycle.
    _topological_order(plan)


def topological_levels(plan: ResearchPlan) -> list[list[str]]:
    """Return task ids grouped into parallel levels (level 0 has no deps).

    Tasks within a level are independent and could run concurrently. Raises
    :class:`PlanValidationError` on a cycle.
    """
    remaining = {task.id: set(task.dependencies) for task in plan.tasks}
    # Preserve declaration order for deterministic output within a level.
    order = [task.id for task in plan.tasks]
    levels: list[list[str]] = []
    while remaining:
        ready = [tid for tid in order if tid in remaining and not remaining[tid]]
        if not ready:
            raise PlanValidationError(
                f"dependency cycle among tasks: {sorted(remaining)}"
            )
        levels.append(ready)
        for tid in ready:
            del remaining[tid]
        for deps in remaining.values():
            deps.difference_update(ready)
    return levels


def _topological_order(plan: ResearchPlan) -> list[str]:
    ordered: list[str] = []
    for level in topological_levels(plan):
        ordered.extend(level)
    return ordered

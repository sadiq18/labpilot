"""Assign execution order from the DAG's topological levels.

Deterministic: ``order`` follows the topological levels (dependencies first);
tasks in the same level are independent and could run in parallel. This does not
execute anything — it only records ordering the future executor can consume.
"""

from __future__ import annotations

from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.validator import topological_levels


def schedule(plan: ResearchPlan) -> ResearchPlan:
    """Set each task's ``order`` from its topological level; returns ``plan``."""
    order_by_id: dict[str, int] = {}
    index = 0
    for level in topological_levels(plan):
        for task_id in level:
            order_by_id[task_id] = index
            index += 1
    for task in plan.tasks:
        task.order = order_by_id.get(task.id, task.order)
    plan.tasks.sort(key=lambda task: task.order)
    return plan


def parallel_levels(plan: ResearchPlan) -> list[list[str]]:
    """Task ids grouped into levels that could run concurrently."""
    return topological_levels(plan)

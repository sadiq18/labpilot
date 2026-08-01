"""generate_plan tool handler."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.artifacts.plan import PlanArtifacts
from labpilot.research_engine.planner import compile_baseline_plan, compile_research_plan
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def generate_plan(
    workspace: Workspace,
    *,
    baseline: bool = False,
    hypothesis_id: str | None = None,
    llm_client: Any | None = None,
    priority: int = 0,
) -> ToolResult:
    """Compile a research plan and persist it through :class:`PlanArtifacts`."""
    if baseline and hypothesis_id:
        raise ValueError("pass baseline=True or hypothesis_id, not both")
    if not baseline and not hypothesis_id:
        raise ValueError("pass baseline=True or hypothesis_id")

    plan_arts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        if baseline:
            plan = compile_baseline_plan(
                workspace.competition,
                knowledge_dir=workspace.knowledge_dir,
                llm_client=llm_client,
                plan_store=plan_arts.store,
                write_projections=False,
                priority=priority,
            )
        else:
            assert hypothesis_id is not None
            hyp = HypothesisStore(workspace.knowledge_dir, workspace.competition).get(
                hypothesis_id
            )
            if hyp is None:
                raise ValueError(
                    f"hypothesis not found: {hypothesis_id} "
                    f"(competition={workspace.competition})"
                )
            plan = compile_research_plan(
                hyp,
                knowledge_dir=workspace.knowledge_dir,
                competition=workspace.competition,
                llm_client=llm_client,
                plan_store=plan_arts.store,
                write_projections=False,
                priority=priority,
            )
        ref = plan_arts.upsert(plan, write_projection_files=True)
    finally:
        plan_arts.close()

    return ToolResult(
        refs=[ref],
        data={
            "plan_id": plan.id,
            "status": str(plan.status),
            "task_count": len(plan.tasks),
            "generated_by": plan.generated_by,
            "plan": plan,
        },
    )

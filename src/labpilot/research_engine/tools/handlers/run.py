"""run_plan tool handler."""

from __future__ import annotations

import logging
from typing import Any

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.artifacts.execution import ExecutionArtifacts
from labpilot.research_engine.artifacts.plan import PlanArtifacts
from labpilot.research_engine.execution import (
    EngineerError,
    ResearchEngineer,
    default_capability_registry,
)
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


def _codegen_strategy(workspace: Workspace) -> str:
    """`codegen.strategy` from the workspace config, or the safe default.

    Never raises: an unreadable or absent config means `whole_file`, which is
    the path that has always worked. A campaign should not fail to produce code
    because a config file has a typo in an unrelated section.
    """
    try:
        from labpilot.config import load_config

        return str(load_config(workspace.root / "configs" / "default.yaml").codegen.strategy)
    except Exception as exc:  # noqa: BLE001 — config trouble must not stop a run
        logger.debug("codegen strategy unreadable, using whole_file: %s", exc)
        return "whole_file"


def run_plan(
    workspace: Workspace,
    *,
    plan_id: str,
    dry_run: bool = False,
    submit: bool = False,
    install_packages: bool = False,
    llm_client: Any | None = None,
    constraints: dict[str, Any] | None = None,
) -> ToolResult:
    """Create an execution via adapters and run the Engineer for ``plan_id``."""
    plan_arts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        plan = plan_arts.get(plan_id)
    finally:
        plan_arts.close()
    if plan is None:
        raise EngineerError(
            f"Plan not found: {plan_id} (competition={workspace.competition}). "
            "Create one with `research plan create --baseline` "
            "or `--hypothesis H-xxx`."
        )

    exec_arts = ExecutionArtifacts(workspace.knowledge_dir, workspace.competition)
    # `codegen.strategy` is the config surface M19 §10 specifies. Read here
    # because this is the last place that has both the workspace and the
    # constraint dict; the capability sees only constraints. An explicit
    # caller-supplied `codegen_strategy` still wins, since it is spread after.
    merged: dict[str, Any] = {
        "codegen_strategy": _codegen_strategy(workspace),
        "dry_run": dry_run,
        "allow_upload": submit,
        "smoke_syntax_only": dry_run,
        "train_stub": dry_run,
        "remote_dry_run": True,
        "skip_download": dry_run,
        "llm_client": llm_client,
        **(constraints or {}),
    }
    configs_dir = workspace.root / "configs"
    if configs_dir.is_dir():
        merged.setdefault("competitions_dir", configs_dir)

    registry = default_capability_registry(
        install_packages=install_packages and not dry_run,
        llm_client=llm_client,
    )
    engineer = ResearchEngineer(
        knowledge_dir=workspace.knowledge_dir,
        competition=workspace.competition,
        registry=registry,
        execution_store=exec_arts.store,
        constraints=merged,
    )
    try:
        execution, create_ref = exec_arts.create(plan_id)
        execution = engineer.run_plan(plan_id, execution=execution)
    finally:
        engineer.close()
        exec_arts.close()

    ref = ArtifactRef(
        kind="execution",
        id=execution.id,
        schema_id=create_ref.schema_id,
        path=execution.workspace_path,
        competition=workspace.competition,
    )
    return ToolResult(
        refs=[ref],
        data={
            "execution_id": execution.id,
            "plan_id": execution.plan_id,
            "status": execution.status,
            "error": execution.error,
            "workspace_path": execution.workspace_path,
        },
    )

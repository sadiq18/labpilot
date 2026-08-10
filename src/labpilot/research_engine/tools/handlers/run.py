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
    """`codegen.strategy` for this workspace — see `codegen_strategy`."""
    from labpilot.research_engine.execution.codegen_strategy import (
        resolve_codegen_strategy,
        workspace_config_path,
    )

    return resolve_codegen_strategy(workspace_config_path(workspace))


def _kaggle_config(workspace: Workspace):
    """The workspace's Kaggle credentials, or None when it has no config.

    Read the same way `_codegen_strategy` reads its setting, and never raising:
    a campaign should not fail to start because a config file has a typo in an
    unrelated section — it should fail at the step that needs the thing.
    """
    from labpilot.research_engine.execution.codegen_strategy import workspace_config_path

    path = workspace_config_path(workspace)
    try:
        from labpilot.config import load_config

        # `load_config(None)` still applies the env layer, and that is where
        # credentials actually are: `KaggleConfig.api_token/username/key` are
        # `Field(exclude=True)` and `_apply_settings` overwrites all three from
        # `Settings` unconditionally. Returning `None` when `configs/default.yaml`
        # is missing therefore reported "no credentials" for the **documented**
        # setup — a `.env` beside `labpilot.yaml`, which is exactly what
        # `kaggle_credentials_setup_hint()` tells users to create. Reported four
        # times on PR #120.
        return load_config(path if path and path.is_file() else None).kaggle
    except Exception:
        # Loudly, and with the traceback. This is the PR's own subject: a
        # swallowed config error here returns `None`, which `prepare_workspace`
        # then reports as "no credentials" — a false diagnosis, and at `debug`
        # nobody would ever see the real one.
        logger.exception("kaggle config unreadable for %s; leaving it unset", workspace)
        return None


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
        # The conductor's path never forwarded these, so `prepare_workspace`
        # reached its download step with no credentials on *every* campaign and
        # skipped it — silently, because a skip and a success were the same
        # answer. M20 made the skip visible, which turned an old silent gap into
        # a loud failure and exposed the gap itself. Reported on PR #120.
        "kaggle": _kaggle_config(workspace),
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

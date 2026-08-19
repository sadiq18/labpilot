"""The single-tool path (`campaign_mode=False`), which had no coverage at all.

A review of PR #111 reported that `budget_cfg` is unbound outside campaign mode
and that the goal-override logic at `loop.py:376` therefore raises once
`step >= 1`. It is not — `load_budget_pair` runs at the top of every iteration,
before the `if campaign_mode:` branch — but the finding pointed at a real gap:
nothing exercised `campaign_mode=False` past a single step.

These tests exist so the claim has an answer that survives a refactor, rather
than one that has to be re-derived by reading the loop.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str) -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    def echo(workspace: Workspace, **kwargs: object) -> ToolResult:
        path = workspace.artifacts_dir / "echo.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(kwargs), encoding="utf-8")
        return ToolResult(
            refs=[
                ArtifactRef(
                    kind="echo",
                    id="echo:1",
                    schema_id="labpilot.artifact.echo/v1",
                    path=str(path),
                    competition=workspace.competition,
                )
            ]
        )

    reg.register(
        ToolDescriptor(
            name="echo",
            summary="echo inputs to an artifact",
            handler=echo,
            gated=False,
            capability_status="fixed",
        )
    )
    return reg


def test_single_tool_mode_runs_past_the_first_step(tmp_path: Path) -> None:
    """The reported failure needed `step >= 1`, so one step would not show it.

    `budget_cfg` is read by the goal-override logic on every iteration; if it
    were bound only under `campaign_mode`, this would raise `UnboundLocalError`
    on the second pass.
    """
    ws = _ws(tmp_path, "singletool")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("run one tool")
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            _registry(),
            max_steps=4,
            auto_approve=True,
            prefer_offline=True,
            campaign_mode=False,
        )
        assert isinstance(decisions, list)
    finally:
        store.close()


def test_budget_is_loaded_outside_campaign_mode(tmp_path: Path) -> None:
    """The specific claim, pinned: the budget pair is available to the
    override logic regardless of `campaign_mode`."""
    import inspect

    from labpilot.research_engine.conductor import loop as loop_mod

    # The loop lives in the inner function; `run_until_stop` is a wrapper.
    src = inspect.getsource(loop_mod._run_until_stop_inner)
    load_at = src.index("budget_cfg, budget_state = load_budget_pair(session)")
    guard_at = src.index("if campaign_mode:")
    assert load_at < guard_at, (
        "load_budget_pair must stay ahead of the campaign_mode branch; "
        "moving it inside would unbind budget_cfg for the single-tool path"
    )

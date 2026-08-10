"""``research conduct`` — product entry for the Campaign / Conductor engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from labpilot.cli.config_helpers import (
    default_tools,
    load_cli_config,
    resolve_competition,
    resolve_os_workspace,
)
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.conductor import ConductorStore, run_until_stop
from labpilot.research_engine.conductor.approvals import ApprovalResult
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    budgets_to_metadata,
    recompute_metric_history,
)
from labpilot.research_engine.conductor.checkpoint import (
    latest_active_session,
    load_budget_pair,
    save_checkpoint,
)
from labpilot.research_engine.conductor.metrics import ensure_metrics
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.workspace_facade import Workspace

conduct_app = typer.Typer(
    help="Run the Research Conductor / Campaign Engine (product entry).",
    no_args_is_help=True,
)
console = Console()


def _apply_deterministic_env(offline: bool) -> None:
    """Historical hook: offline used to opt micro agents into rule engines.

    Rule engines are gone (issue #39). Offline campaigns must avoid agents that
    need an LLM (policy), not silently substitute deterministic output.
    """
    _ = offline


def _test_registry_subset() -> object:
    """Build a registry with cheap offline-safe tools for dry/offline loops."""
    from labpilot.research_engine.tools.handlers.memory import query_memory
    from labpilot.research_engine.tools.handlers.papers import search_papers
    from labpilot.research_engine.tools.registry import ToolRegistry

    reg = ToolRegistry()

    def echo_analyze(workspace: Workspace, **_kwargs: object) -> ToolResult:
        path = workspace.research_paths.reports_dir
        path.mkdir(parents=True, exist_ok=True)
        out = path / "analyze.json"
        if not out.is_file():
            out.write_text(
                '{"schema_version":1,"competition":{"slug":"%s"},"analyzers":[],"notes":["conduct stub"]}\n'
                % workspace.competition,
                encoding="utf-8",
            )
        return ToolResult(
            refs=[],
            data={"path": str(out), "stub": True},
        )

    reg.register(
        ToolDescriptor(
            name="analyze_competition",
            description="stub",
            handler=echo_analyze,
        )
    )
    reg.register(
        ToolDescriptor(
            name="search_papers",
            description="search papers",
            handler=lambda ws, **kw: search_papers(ws, offline=True, **kw),
        )
    )
    reg.register(
        ToolDescriptor(
            name="query_memory",
            description="query memory",
            handler=query_memory,
        )
    )
    return reg


def _offline_fallback_prompt(yes: bool):
    def _prompt(reason: str):
        if yes:
            return "allow"
        console.print(f"\n[yellow]LLM policy unavailable[/yellow]: {reason}")
        console.print("Fall back to offline deterministic policy?")
        while True:
            raw = typer.prompt("[a]llow / [d]eny / [r]etry", default="a")
            choice = str(raw).strip().lower()
            if choice in {"a", "allow", "y", "yes"}:
                return "allow"
            if choice in {"d", "deny", "n", "no"}:
                return "deny"
            if choice in {"r", "retry"}:
                return "retry"
            console.print("Enter allow, deny, or retry.")

    return _prompt


def _approval_prompt(yes: bool):
    def _approve(tool_name: str) -> ApprovalResult:
        if yes:
            return ApprovalResult(decision="approve", comment="", gated_tool=tool_name)
        console.print(f"\n[yellow]Approval required[/yellow] for [cyan]{tool_name}[/cyan]")
        raw = typer.prompt("Approve? [y/N]", default="n")
        approved = str(raw).strip().lower() in {"y", "yes"}
        comment = typer.prompt(
            "Comment (optional, feeds future decisions)", default="", show_default=False
        )
        return ApprovalResult(
            decision="approve" if approved else "reject",
            comment=str(comment or ""),
            gated_tool=tool_name,
        )

    return _approve


def _open_workspace(
    *,
    competition: str | None,
    config_path: Path,
    knowledge_dir: Path | None,
    workspace_path: Path | None,
    goal: str | None = None,
) -> tuple[Any, Workspace, str]:
    config, marker_ws = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, marker_ws)
    ws = resolve_os_workspace(
        competition=competition,
        config=config,
        client=marker_ws,
        goal=goal or "",
    )
    ws.ensure_roots()
    return config, ws, competition


def _resolve_session(
    store: ConductorStore,
    session_id: str | None,
) -> Any:
    if session_id:
        session = store.get_session(session_id)
        if session is None:
            console.print(f"[red]Unknown session[/red] {session_id}")
            raise typer.Exit(1)
        return session
    session = latest_active_session(store)
    if session is None:
        console.print("[red]No active session[/red] — run `research conduct run` first")
        raise typer.Exit(1)
    return session


def _resolve_campaign_direction(ws: Any, competition: str) -> bool | None:
    """Whether this competition maximises its metric, or None if unknown.

    None leaves `BudgetConfig`'s own default in place rather than guessing —
    the objective checks only fire when a metric target is set, so an unknown
    direction is better left visible than papered over.
    """
    from labpilot.research_engine.intelligence.competition.direction import resolve_maximize
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    try:
        paths = ResearchPaths(ws.knowledge_dir, competition)
        return resolve_maximize(
            competition=competition,
            workspace_root=getattr(ws, "root", None),
            knowledge_root=paths.root,
            extracted_dir=paths.extracted_dir,
        )
    except Exception:  # noqa: BLE001 — an unknown direction must not block a run
        return None


def _budget_metadata(
    *,
    max_submissions: int | None,
    max_wall_s: float | None,
    max_cost_usd: float | None,
    target_metric: str | None,
    target_value: float | None,
    plateau_window: int,
    maximize: bool | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(existing or {})
    cfg = BudgetConfig(
        max_submissions=max_submissions,
        max_wall_s=max_wall_s,
        max_cost_usd=max_cost_usd,
        target_metric=target_metric,
        target_value=target_value,
        plateau_window=plateau_window,
        # Resolved from the competition, not defaulted. `BudgetConfig.maximize`
        # used to default True with nothing overriding it, so on rogii (MSE)
        # every session stored `"maximize": true` and a metric target would have
        # stopped the campaign on the wrong side. Latent only because rogii ran
        # with target_metric unset. Same defect as `build_evidence_card`.
        **({} if maximize is None else {"maximize": maximize}),
    )
    if maximize is None and target_metric and target_value is not None:
        # A target is the only thing that reads `maximize`, so an unknown
        # direction is harmless until one is set — and unacceptable after.
        # "stop when the metric reaches X" would fire on the wrong side.
        # `build_evidence_card` already refuses on this; the campaign should
        # not be more permissive about the same unknown.
        raise typer.BadParameter(
            f"cannot determine whether {target_metric!r} should be maximised or "
            "minimised, so --target-value would stop the campaign on the wrong "
            "side. Set metric.direction in the workspace competition.json, or "
            "run `research analyze` to produce the competition profile."
        )
    from labpilot.research_engine.conductor.budgets import BudgetState

    state = BudgetState.model_validate(meta.get("budget_state") or {})
    # `persist_budgets` is not the only writer — `existing` can carry a prior
    # session's `budget_state`, `score_events` included, and this must recompute
    # the same as that path does or a resumed session's `metric_history` goes
    # stale the moment this function is the one that writes it back out.
    recompute_metric_history(state)
    return budgets_to_metadata(meta, cfg, state)


@conduct_app.command("run")
def conduct_run(
    goal: str = typer.Argument(..., help='Research goal, e.g. "Win Rogii"'),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int = typer.Option(8, "--max-steps", help="Stop after N policy steps"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-approve gated tools (no interactive prompts)",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help=(
            "No LLM policy; deterministic catalog order + safe stubs where needed. "
            "Also permits micro agents to use their rule engines, which otherwise "
            "refuse to run without an LLM."
        ),
    ),
    autonomy: int = typer.Option(
        0,
        "--autonomy",
        help="0=gate plan+submit (default); 1=gate submit only",
        min=0,
        max=1,
    ),
    max_submissions: int | None = typer.Option(None, "--max-submissions"),
    max_wall_s: float | None = typer.Option(None, "--max-wall-s"),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd"),
    target_metric: str | None = typer.Option(None, "--target-metric"),
    target_value: float | None = typer.Option(None, "--target-value"),
    plateau_window: int = typer.Option(3, "--plateau-window"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Observe → research action → approve → execute until stop or max steps."""
    config, ws, competition = _open_workspace(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
        goal=goal,
    )
    store = ConductorStore(ws.knowledge_dir, competition)
    registry = default_tools()
    llm = None if offline else resolve_llm_client(config.llm)
    if offline:
        registry = _test_registry_subset()  # type: ignore[assignment]

    try:
        meta = _budget_metadata(
            max_submissions=max_submissions,
            max_wall_s=max_wall_s,
            max_cost_usd=max_cost_usd,
            target_metric=target_metric,
            target_value=target_value,
            plateau_window=plateau_window,
            maximize=_resolve_campaign_direction(ws, competition),
            existing={
                "max_steps": max_steps,
                "offline": offline,
                "autonomy": autonomy,
            },
        )
        if target_metric:
            meta["target_metric"] = target_metric
        if target_value is not None:
            meta["target_value"] = target_value
        session = store.create_session(goal, metadata=meta)
        ensure_metrics(store, session.id)
        console.print(
            f"[bold]Conductor[/bold] session [cyan]{session.id}[/cyan] "
            f"goal={goal!r} competition={competition} autonomy={autonomy}"
        )
        _apply_deterministic_env(offline)
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            registry,
            llm_client=llm,
            max_steps=max_steps,
            auto_approve=yes,
            approval_prompt=None if yes else _approval_prompt(yes),
            on_progress=lambda msg: console.print(f"  {msg}"),
            autonomy=autonomy,
            prefer_offline=offline,
            offline_fallback_prompt=None if yes else _offline_fallback_prompt(yes),
        )
    finally:
        store.close()

    console.print(f"\n[green]Done[/green] — {len(decisions)} decision(s) recorded")
    for d in decisions:
        stop = " stop" if d.stop else ""
        tool = d.tool_name or "—"
        console.print(f"  {d.id}: {tool}{stop} — {d.rationale[:80]}")


def _continue_session(
    *,
    session_id: str | None,
    competition: str | None,
    max_steps: int,
    yes: bool,
    offline: bool,
    autonomy: int | None,
    config_path: Path,
    knowledge_dir: Path | None,
    workspace_path: Path | None,
) -> None:
    config, ws, competition = _open_workspace(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    store = ConductorStore(ws.knowledge_dir, competition)
    registry = default_tools()
    llm = None if offline else resolve_llm_client(config.llm)
    if offline:
        registry = _test_registry_subset()  # type: ignore[assignment]

    try:
        session = _resolve_session(store, session_id)
        meta = dict(session.metadata)
        level = autonomy if autonomy is not None else int(meta.get("autonomy", 0))
        if session.status == "paused":
            store.update_session_status(session.id, "running")
        console.print(
            f"[bold]Continue[/bold] session [cyan]{session.id}[/cyan] "
            f"status={session.status} autonomy={level}"
        )
        _apply_deterministic_env(offline)
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            registry,
            llm_client=llm,
            max_steps=max_steps,
            auto_approve=yes,
            approval_prompt=None if yes else _approval_prompt(yes),
            on_progress=lambda msg: console.print(f"  {msg}"),
            autonomy=level,
            prefer_offline=offline,
            offline_fallback_prompt=None if yes else _offline_fallback_prompt(yes),
        )
    finally:
        store.close()

    console.print(f"\n[green]Done[/green] — {len(decisions)} decision(s) this run")


@conduct_app.command("continue")
def conduct_continue(
    session: str | None = typer.Option(None, "--session", help="Session id (default: latest active)"),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int = typer.Option(8, "--max-steps"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    offline: bool = typer.Option(False, "--offline"),
    autonomy: int | None = typer.Option(None, "--autonomy", min=0, max=1),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Continue the latest active (or ``--session``) campaign."""
    _continue_session(
        session_id=session,
        competition=competition,
        max_steps=max_steps,
        yes=yes,
        offline=offline,
        autonomy=autonomy,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )


@conduct_app.command("resume")
def conduct_resume(
    session: str | None = typer.Option(None, "--session"),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int = typer.Option(8, "--max-steps"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    offline: bool = typer.Option(False, "--offline"),
    autonomy: int | None = typer.Option(None, "--autonomy", min=0, max=1),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Alias for ``continue`` — restore paused/running session and run."""
    _continue_session(
        session_id=session,
        competition=competition,
        max_steps=max_steps,
        yes=yes,
        offline=offline,
        autonomy=autonomy,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )


@conduct_app.command("pause")
def conduct_pause(
    session: str | None = typer.Option(None, "--session"),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Mark the active (or ``--session``) campaign as paused."""
    _, ws, competition = _open_workspace(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    store = ConductorStore(ws.knowledge_dir, competition)
    try:
        sess = _resolve_session(store, session)
        store.update_session_status(sess.id, "paused")
        cp = save_checkpoint(store, sess.id, extra={"stop_reason": "operator_pause"})
        console.print(
            f"[yellow]Paused[/yellow] session [cyan]{sess.id}[/cyan] "
            f"(tasks={cp.get('task_count')}, decisions={cp.get('decision_count')})"
        )
    finally:
        store.close()


@conduct_app.command("status")
def conduct_status(
    session: str | None = typer.Option(None, "--session"),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Show campaign session summary, budgets, metrics, and suggestions."""
    _, ws, competition = _open_workspace(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    store = ConductorStore(ws.knowledge_dir, competition)
    try:
        if session:
            sess = store.get_session(session)
            if sess is None:
                console.print(f"[red]Unknown session[/red] {session}")
                raise typer.Exit(1)
        else:
            sess = latest_active_session(store)
            if sess is None:
                sessions = store.list_sessions()
                if not sessions:
                    console.print("[dim]No sessions[/dim]")
                    raise typer.Exit(0)
                sess = sessions[0]
        tasks = store.list_tasks(sess.id)
        decisions = store.list_decisions(sess.id)
        metrics = store.get_metrics(sess.id)
        suggestions = store.list_suggestions(sess.id, limit=10)
        cfg, state = load_budget_pair(sess)
        cp = (sess.metadata or {}).get("checkpoint") or {}

        console.print(f"[bold]Session[/bold] [cyan]{sess.id}[/cyan]")
        console.print(f"  goal: {sess.goal}")
        console.print(f"  status: {sess.status}")
        console.print(f"  competition: {sess.competition}")
        console.print(f"  autonomy: {sess.metadata.get('autonomy', 0)}")
        console.print(
            f"  tasks: {len(tasks)} "
            f"(completed={sum(1 for t in tasks if t.status == 'completed')}, "
            f"pending={sum(1 for t in tasks if t.status == 'pending')})"
        )
        console.print(f"  decisions: {len(decisions)}")

        # The delta failure rate, in the product rather than only in a docstring.
        # M19 §3 flipped the default strategy "when the rate justifies it", and a
        # rate a reviewer cannot recompute is an assertion, not evidence.
        from labpilot.research_engine.telemetry.delta_rate import delta_rate

        # All recorded history, and labelled as such: a rate is meaningless
        # without its window, and this one spans every fix ever made to the
        # adapter. `delta_rate(since=...)` narrows it.
        rate = delta_rate(ws.knowledge_dir, competition)
        if rate.attempts:
            share = "n/a" if rate.failure_rate is None else f"{rate.failure_rate:.1%}"
            console.print(
                f"  delta (all recorded): {rate.attempts} aider attempt(s), "
                f"{rate.succeeded} usable, {rate.failed} failed "
                f"({share}), {rate.excused} excused"
            )
            if rate.by_kind:
                console.print(f"    by kind: {rate.by_kind}")
        if cp:
            console.print(
                f"  checkpoint: tools={cp.get('completed_tools')} "
                f"last={cp.get('last_decision_id')}"
            )
        console.print(
            f"  budgets: max_submissions={cfg.max_submissions} "
            f"max_wall_s={cfg.max_wall_s} max_cost_usd={cfg.max_cost_usd} "
            f"target={cfg.target_metric}:{cfg.target_value}"
        )
        console.print(
            f"  budget_state: submissions={state.submissions} "
            f"cost={state.llm_cost_usd} last_metric={state.last_metric}"
        )
        if metrics:
            console.print(
                f"  metrics: failed={metrics.tasks_failed} blocked={metrics.tasks_blocked} "
                f"unmet={metrics.unmet_goal} interventions={metrics.human_interventions} "
                f"no_capability={metrics.no_capability} submissions={metrics.submissions}"
            )
        if suggestions:
            console.print("  suggestions:")
            for s in suggestions[-5:]:
                console.print(f"    [{s.kind}] {s.message[:100]}")
        gaps = store.list_capability_gaps(status="open", limit=5)
        if gaps:
            console.print("  open_gaps:")
            for g in gaps:
                console.print(f"    {g.gap_key} count={g.count} last={g.last_seen_at}")
    finally:
        store.close()

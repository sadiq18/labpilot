"""``research conduct`` — product entry for the Campaign / Conductor engine."""

from __future__ import annotations

import sys
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
    goal_progress,
)
from labpilot.research_engine.conductor.checkpoint import (
    latest_active_session,
    load_budget_pair,
    save_checkpoint,
)
from labpilot.research_engine.conductor.metrics import ensure_metrics
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.workspace_facade import Workspace

#: Why the default is no bound: a campaign that ends on a step counter has
#: not answered its question, it has run out of turns. M17.
_MAX_STEPS_HELP = (
    "Stop after N policy steps. Unset, the campaign runs until its objective, "
    "a plateau, a budget, or a guidance pause"
)

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
            capability_status="fixed",
        )
    )
    reg.register(
        ToolDescriptor(
            name="search_papers",
            description="search papers",
            handler=lambda ws, **kw: search_papers(ws, offline=True, **kw),
            capability_status="fixed",
        )
    )
    reg.register(
        ToolDescriptor(
            name="query_memory",
            description="query memory",
            handler=query_memory,
            # The real catalog handler, unwrapped — unlike the two above
            # (a local stub, and search_papers pinned to offline=True), this
            # one genuinely varies by query, so `fixed` would be a false
            # declaration in the field that exists to prevent them.
            capability_status="real",
            varies_by=["query"],
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


def _schema_prompt(question):
    """Ask an operator which column is right, showing what the profiler saw.

    Only ever installed for an interactive run. There is no `--yes` variant on
    purpose: `--yes` means "do not ask me to approve your plan", and it must not
    come to mean "decide what my label is".
    """
    print(f"\nSchema question: {question.field}")
    print(f"  {question.context}")
    if question.provisional:
        print(f"  provisional (not acted on): {question.provisional}")
    for candidate in question.candidates:
        fired = ", ".join(signal.id for signal in candidate.signals) or "nothing fired"
        print(f"  · {candidate.candidate} ({candidate.confidence:.2f}) — {fired}")
    answer = input(f"Which column is {question.field}? [blank to stop]: ").strip()
    return answer or None


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


def _stated_objective(
    ws: Any, competition: str
) -> tuple[str | None, str | None, str | None, str | None]:
    """(metric_raw, declared_direction, problem_type, target) from the workspace.

    A view over `read_inputs`, which is now the one reader of what a workspace
    states. It used to parse `competition.json` and `profile.json` itself, a
    second parser of the same two files free to drift from the stage's — and the
    drift would show up as the CLI and `objective.json` disagreeing about the
    metric, which is precisely the thing a persisted objective exists to stop.
    """
    from labpilot.research_engine.intelligence.competition.objective_stage import read_inputs

    root = getattr(ws, "root", None)
    if root is None:
        return None, None, None, None
    facts = read_inputs(Path(root))
    return facts.metric_raw, facts.declared_direction, facts.task, facts.target


def _preflight_objective(ws: Any, competition: str, *, assume_yes: bool) -> dict[str, Any]:
    """Refuse to start a campaign whose objective cannot be justified.

    Checked here rather than mid-run, and the placement is the point: refusing to
    *start* an unattended job costs nothing and reaches the operator while they
    are still at the keyboard. Halting at 2am reaches nobody until morning and
    throws away the night.

    rogii ran campaigns for two weeks with `evaluation_metric: None`, and all
    fifteen of its evidence cards were built as though MSE were maximised. Both
    were free to catch at second zero.

    Returns metadata to stamp on the session. An operator who overrides the gate
    leaves a record there, because a campaign built on an unknown direction must
    stay distinguishable from a resolved one long after the console line is gone.
    """
    from labpilot.research_engine.intelligence.competition.objective import resolve_objective
    from labpilot.research_engine.intelligence.competition.objective_stage import (
        ensure_objective,
    )

    root = getattr(ws, "root", None)
    if root is None:
        # No workspace, nothing to persist to. The preflight still has to answer,
        # and an objective resolved from nothing says so rather than defaulting.
        objective = resolve_objective(metric_raw=None)
        objective_path = None
    else:
        stored, _how = ensure_objective(Path(root), competition)
        objective = stored.spec
        objective_path = Path(root) / "objective.json"

    if not objective.blocks_launch:
        console.print(
            f"[dim]objective:[/dim] {objective.metric_name} "
            f"[dim]({objective.direction} from {objective.direction_source}; "
            f"confidence {objective.confidence:.2f}, capped by "
            f"{objective.source})[/dim]"
        )
        return {
            "objective_metric": objective.metric_name,
            "objective_target": objective.target,
            "objective_direction": objective.direction,
            "objective_source": objective.source,
            "objective_confidence": objective.confidence,
            # The file is the record; these five strings are the index into it.
            # They stay because sessions are queried by them, and a session that
            # named its metric only by a path would have to open the workspace to
            # answer "what was this campaign optimising?".
            **({} if objective_path is None else {"objective_path": str(objective_path)}),
        }

    from labpilot.research_engine.validation import harness

    console.print(f"[red]Objective not resolved[/red] - {objective.why_blocked()}")
    for line in objective.evidence:
        console.print(f"  [dim]-[/dim] {line}")
    if objective.alternatives:
        console.print(
            f"  [dim]candidates:[/dim] {', '.join(objective.alternatives)}"
        )
    # Advice from the operator's own domain. Telling a benchmark operator to
    # "set evaluation_metric in competition.json" names a file their workspace
    # will never have, which is the same leak as the gate refusing them outright
    # — a refusal nobody can act on is a wall, and this gate exists to ask rather
    # than to wall.
    if bool(root) and harness.handles(Path(root)):
        console.print(
            f"\n  Set it in {harness.OBJECTIVE_FILE} and re-run:\n"
            '    [cyan]{"metric": "pass_rate", "direction": "maximize"}[/cyan]'
        )
    else:
        console.print(
            "\n  Set it in the workspace contract and re-run, e.g. competition.json:\n"
            '    [cyan]"evaluation_metric": {"name": "rmse", "direction": "minimize"}[/cyan]'
        )

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not assume_yes and interactive:
        # The operator is here, so offer the choice rather than only refusing. A
        # `--yes` run must never reach this branch: auto-answering "which fact is
        # true?" is how a wrong objective gets frozen into a workspace, and the
        # reuse discipline everywhere else means it would stay wrong.
        if typer.confirm(
            "\nRun anyway, accepting that every conclusion may carry the wrong sign?",
            default=False,
        ):
            console.print("[yellow]proceeding with an unresolved objective[/yellow]")
            return {
                "objective_override": True,
                "objective_blocked_reason": objective.why_blocked(),
                "objective_metric": objective.metric_name,
                "objective_direction": objective.direction,
                "objective_confidence": objective.confidence,
            }
    raise typer.Exit(2)


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
    return budgets_to_metadata(meta, cfg, state)


@conduct_app.command("run")
def conduct_run(
    goal: str = typer.Argument(..., help='Research goal, e.g. "Win Rogii"'),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int | None = typer.Option(None, "--max-steps", help=_MAX_STEPS_HELP),
    branches: int = typer.Option(
        1,
        "--branches",
        "-k",
        min=1,
        help=(
            "Test the top K untested hypotheses in parallel, each in its own "
            "git worktree with a share of the cores. 1 runs them one at a time. "
            "Keep K within twice your LLM provider's per-minute limit: a branch "
            "waits out one rate-limit window and then fails, so wider fan-outs "
            "lose branches to the limiter and count them as failed experiments."
        ),
    ),
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
    objective_meta = _preflight_objective(ws, competition, assume_yes=yes)
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
                **objective_meta,
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
            schema_prompt=None if yes else _schema_prompt,
            on_progress=lambda msg: console.print(f"  {msg}"),
            autonomy=autonomy,
            prefer_offline=offline,
            offline_fallback_prompt=None if yes else _offline_fallback_prompt(yes),
            branches=branches,
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
    max_steps: int | None,
    branches: int,
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
        # Same gate as `run`: a contract edited between sessions can make an
        # objective contradictory, and resuming would step against it unchecked.
        #
        # Ordered after the session is loaded so a recorded override can be
        # honoured. Gating unconditionally with `assume_yes=True` made a campaign
        # the operator had deliberately started under an unresolved objective
        # impossible to resume at all — the launch accepted the answer and every
        # `continue` afterwards refused it, which is the gate contradicting a
        # decision it had already taken.
        if meta.get("objective_override"):
            console.print(
                "[yellow]objective:[/yellow] resuming under the override recorded at launch — "
                f"{meta.get('objective_blocked_reason') or 'reason not recorded'}"
            )
        else:
            _preflight_objective(ws, competition, assume_yes=True)
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
            schema_prompt=None if yes else _schema_prompt,
            on_progress=lambda msg: console.print(f"  {msg}"),
            autonomy=level,
            prefer_offline=offline,
            offline_fallback_prompt=None if yes else _offline_fallback_prompt(yes),
            branches=branches,
        )
    finally:
        store.close()

    console.print(f"\n[green]Done[/green] — {len(decisions)} decision(s) this run")


@conduct_app.command("continue")
def conduct_continue(
    session: str | None = typer.Option(
        None, "--session", help="Session id (default: latest active)"
    ),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int | None = typer.Option(None, "--max-steps", help=_MAX_STEPS_HELP),
    branches: int = typer.Option(
        1,
        "--branches",
        "-k",
        min=1,
        help=(
            "Test the top K untested hypotheses in parallel (1 = one at a time). "
            "Keep K within twice your provider's per-minute limit."
        ),
    ),
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
        branches=branches,
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
    max_steps: int | None = typer.Option(None, "--max-steps", help=_MAX_STEPS_HELP),
    branches: int = typer.Option(
        1,
        "--branches",
        "-k",
        min=1,
        help=(
            "Test the top K untested hypotheses in parallel (1 = one at a time). "
            "Keep K within twice your provider's per-minute limit."
        ),
    ),
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
        branches=branches,
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
                f"  checkpoint: tools={cp.get('completed_tools')} last={cp.get('last_decision_id')}"
            )
        console.print(
            f"  budgets: max_submissions={cfg.max_submissions} "
            f"max_wall_s={cfg.max_wall_s} max_cost_usd={cfg.max_cost_usd} "
            f"target={cfg.target_metric}:{cfg.target_value}"
        )
        # `last_metric` stays on the raw-state line beside the other persisted
        # counters. It is not a second progress rendering — the goal line below
        # is the interpreted view, this is the field itself, and it is the one
        # `metric_target` compares against. Dropping it also blanked the metric
        # entirely for a session predating `score_events`, which has readings
        # here and an empty series.
        console.print(
            f"  budget_state: submissions={state.submissions} "
            f"cost={state.llm_cost_usd} last_metric={state.last_metric}"
        )
        # The same line the campaign prints each step, so a detached run can be
        # checked without tailing its log — and so there is one rendering of
        # *progress* rather than two that can disagree.
        line = goal_progress(cfg, state)
        if line:
            console.print(f"  {line}")
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

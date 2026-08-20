"""Conductor run loop — campaign-aware observe → action → approve → dispatch.

This file's `store.new_decision_id()` + `store.append_decision(...)` call
sites are correct as written — they only ever run on the sequential (K=1)
path, one decision recorded at a time. They are **not** safe under
concurrent callers: `new_decision_id()` computes `MAX(id)+1` via an unlocked
`SELECT`, so two callers can read the same "next id" before either commits
(M11, verified: 6/20 unlocked concurrent attempts raised `IntegrityError`).
K-way fan-out (M11 task 7) must record each branch's decision through
`ConductorStore.append_new_decision(...)` instead, which holds
`write_lock_for(db_path)` across the whole allocate-then-insert sequence —
not by copying the two-call pattern used below.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from labpilot.accessor.common.micro_agents import LLMDegradedError
from labpilot.accessor.profiler.questions import (
    ANSWERS_FILENAME,
    SchemaQuestion,
    open_questions,
    record_answer,
)
from labpilot.research_engine.conductor.actions import (
    ResearchAction,
    map_research_action,
    offline_next_research_action,
    resolve_step_args,
)
from labpilot.research_engine.conductor.approvals import (
    SUBMIT_TOOLS,
    ApprovalPrompt,
    OfflineFallbackPrompt,
    maybe_approve,
)
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    comparable_tail,
    evaluate_stops,
    goal_progress,
    metric_names_match,
    score_summary,
    submit_tools_allowed,
)
from labpilot.research_engine.conductor.checkpoint import (
    load_budget_pair,
    persist_budgets,
    save_checkpoint,
)
from labpilot.research_engine.conductor.gap_ledger import build_suggestion_context
from labpilot.research_engine.conductor.metrics import ensure_metrics, record_suggestion
from labpilot.research_engine.conductor.models import DecisionRecord
from labpilot.research_engine.conductor.policy import decide_next
from labpilot.research_engine.conductor.scheduler import Scheduler
from labpilot.research_engine.conductor.scoring import score_event_for
from labpilot.research_engine.conductor.stagnation import (
    mint_stagnation_hypothesis,
    stagnation_window,
)
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.telemetry.agent_provenance import recording_provenance
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

# How many times an advisory "stop" is overridden while the objective is unmet.
# Bounded so a policy that genuinely has nothing left can still end the run.
_MAX_STOP_OVERRIDES = 2


#: Tools whose outcome the circuit breaker counts. Only these produce (or fail
#: to produce) an experiment; a failed `analyze_competition` is a bad step, not
#: evidence that the campaign cannot work.
_EXPERIMENT_TOOLS = frozenset({"run_experiment", "run_plan"})

#: What a stop reason means for the session's status. Anything absent is a
#: completion. `failing` is not one — a campaign whose every execution broke
#: must not land where one that met its target does — and `needs_guidance` is
#: not one either: nothing is broken and nothing is finished, so it pauses,
#: which is the status a `conduct continue` picks back up.
#:
#: Immutable for the same reason `_EXPERIMENT_TOOLS` beside it is a frozenset:
#: an importer that could rewrite this would silently change how every
#: campaign's terminal status is recorded.
_STOP_SESSION_STATUS = MappingProxyType({"failing": "failed", "needs_guidance": "paused"})


#: Each experiment tool's own `dry_run` default, so a fan-out of a step that
#: did not set one behaves like the sequential dispatch of that same step.
#: They genuinely differ — `run_plan` trains for real unless told otherwise,
#: `run_experiment` does not — and hardcoding either made the other wrong:
#: defaulting to a dry run turned every unset `run_plan` fan-out into K
#: branches that trained nothing, wrote placeholder metrics, and promoted
#: nobody once the placeholder guard refused them.
_DRY_RUN_DEFAULTS: dict[str, bool] = {"run_plan": False, "run_experiment": True}


#: Execution statuses that mean the experiment actually produced a result.
#: `pending` and `running` are not successes — an execution left mid-flight has
#: produced nothing, and treating it as a win would reset the breaker on a
#: campaign that is stalling rather than progressing.
_SUCCESS_STATUSES = frozenset({"succeeded"})


def _experiment_outcome(result: object) -> tuple[bool, str]:
    """`(succeeded, error)` for an experiment tool that returned without raising.

    A tool call returning is not an experiment succeeding: `run_plan` reports a
    failed execution in `data["status"]` and raises nothing at all. Reading the
    status is what makes this the question the breaker is supposed to ask.

    An absent status is treated as success, deliberately — this runs on the
    path where the call *worked*, and inventing failures from a missing field
    would stop campaigns for a reporting gap rather than a real one.
    """
    data = getattr(result, "data", None)
    if not isinstance(data, dict) or "status" not in data:
        return True, ""
    status = str(data.get("status") or "").strip().lower()
    if status in _SUCCESS_STATUSES:
        return True, ""
    error = str(data.get("error") or "").strip()
    return False, error or f"execution status={status or 'unknown'}"


def _fan_out_experiment(
    store,
    workspace: Workspace,
    session_id: str,
    *,
    step: int,
    branches: int,
    rationale: str,
    llm_client: Any | None,
    dry_run: bool,
    submit: bool,
    agent: Any,
    auto_approve: bool,
    approval_prompt: ApprovalPrompt | None,
    autonomy: int,
    progress: Callable[[str], None],
) -> list[DecisionRecord] | None:
    """Run this step as K parallel branches, or return None to stay sequential.

    None — not an empty list — when the step is not a fan-out after all: too
    few untested hypotheses, or none that could be claimed and set up. The
    caller then dispatches its single task exactly as before, so every path
    that is not a fan-out keeps the behaviour it had before this existed.

    Each branch gets its own `DecisionRecord` and feeds the circuit breaker
    individually, which is the audit parity the sequential path already has:
    a fan-out that recorded one decision for K experiments would let the
    breaker count K failures as one, and a campaign would run past the point
    it was built to stop at. Design §7.
    """
    from labpilot.research_engine.conductor.fanout import (
        PlanRejected,
        prepare_branches,
        resolve_k,
        run_branches,
        select_hypotheses,
        teardown_branches,
    )

    candidates = select_hypotheses(workspace, branches)
    k = resolve_k(branches, available=len(candidates))
    if k < 2:
        return None

    if agent is None:
        logger.warning("no experiment specialist registered; not fanning out")
        return None

    def make_plan(ws: Workspace, hypothesis_id: str) -> str:
        """Compile one branch's plan, through the same gate as a planned step.

        `generate_plan` is in `PLAN_TOOLS`, which `gated_tools_for_autonomy(0)`
        gates — so at autonomy 0 the operator approves every plan compilation
        on the sequential path. Calling the handler directly would fan out K
        billable LLM calls past that gate on one approval given for the step's
        single task. Asked per branch, because that is what is being approved.

        A rejection raises, which `prepare_branches` already handles the right
        way: the claim goes back and the branch is dropped whole.
        """
        from labpilot.research_engine.tools.handlers.plan import generate_plan

        approval = maybe_approve(
            store,
            session_id=session_id,
            tool_name="generate_plan",
            auto=auto_approve,
            prompt=approval_prompt,
            autonomy=autonomy,
        )
        if approval is not None and approval.decision == "reject":
            # `PlanRejected`, not a builtin: `prepare_branches` reports a
            # decline at info and a genuine planning fault at error, and it can
            # only tell them apart by type.
            raise PlanRejected(f"operator declined a plan for {hypothesis_id}")

        result = generate_plan(ws, hypothesis_id=hypothesis_id, llm_client=llm_client)
        plan_id = str((getattr(result, "data", None) or {}).get("plan_id") or "")
        if not plan_id:
            raise ValueError(f"generate_plan returned no plan_id for {hypothesis_id}")
        return plan_id

    prepared = prepare_branches(
        workspace,
        candidates[:k],
        session_id=session_id,
        repo_root=workspace.root,
        make_plan=make_plan,
    )
    if len(prepared) < 2:
        # Whatever was set up is torn down before falling back, so the
        # sequential path does not start with claims held and worktrees
        # checked out by a fan-out that never ran.
        teardown_branches(workspace, prepared, [])
        logger.info("only %d branch(es) could be set up; staying sequential", len(prepared))
        return None

    # The cohort is named by a decision that is actually written, not by the
    # step number and not by a peeked id.
    #
    # `step` restarts with every `conduct resume`, so a resumed session
    # reaching step 3 again reused `S-001-step3` and `promote_within_cohort`
    # re-ranked the new branches against the abandoned attempt's members.
    # `new_decision_id()` is no better on its own: it computes `MAX(id)+1` and
    # reserves nothing, so a fan-out that raised before recording any decision
    # handed the next one the same id and the same merged cohort.
    #
    # Appending the record first allocates the id under the store's write lock
    # and gives the audit log the thing it was missing anyway — one entry
    # saying the campaign chose to fan out, above the per-branch entries.
    outcomes = []
    # Everything after the branches exist is inside the `try`, including
    # naming the cohort: `append_new_decision` writes to the store and can
    # fail, and a failure above the `try` would leave K claims held and K
    # worktrees checked out with nothing to release them.
    try:
        cohort_record = store.append_new_decision(
            session_id,
            "fan_out",
            rationale,
            observe={
                "step": step,
                "branches": [b.hypothesis_id for b in prepared],
            },
        )
        cohort_id = f"{session_id}-{cohort_record.id}"
        progress(f"Fan-out {len(prepared)} branches (cohort {cohort_id})")
        outcomes = run_branches(
            prepared,
            agent=agent,
            cohort_id=cohort_id,
            workspace=workspace,
            context=_branch_context(workspace),
            build_task=lambda branch, cohort: _branch_task(
                branch, cohort, dry_run=dry_run, submit=submit
            ),
        )
    finally:
        # `outcomes` is empty when the block raised, which teardown reads as
        # "nothing succeeded" and releases every claim.
        teardown_branches(workspace, prepared, outcomes)

    decisions: list[DecisionRecord] = [cohort_record]
    for outcome in outcomes:
        record = store.append_new_decision(
            session_id,
            "run_experiment",
            rationale,
            args={"plan_id": outcome.plan_id, "hypothesis_id": outcome.hypothesis_id},
            artifact_refs=[r.model_dump() for r in outcome.refs],
            observe={
                "step": step,
                "cohort_id": cohort_id,
                "branch": outcome.hypothesis_id,
                "ok": outcome.ok,
                "error": outcome.error,
            },
        )
        decisions.append(record)
        _record_experiment_outcome(
            store,
            session_id,
            succeeded=outcome.ok,
            error=outcome.error or "",
            workspace=workspace,
            execution_id=outcome.execution_id,
        )
        progress(f"Branch {outcome.hypothesis_id}: {'ok' if outcome.ok else outcome.error}")
    return decisions


def _reconcile_stale_worktrees(store, workspace: Workspace) -> None:
    """Clear worktrees left by a fan-out that never reached teardown.

    Runs before the first branch of a campaign, because a worktree still
    registered keeps its branch checked out and `worktree add -B` for that
    same experiment key then fails — a crash in one campaign would otherwise
    make the next one unable to re-test those hypotheses at all.

    `live_branches` covers the campaigns that are still running, asked of the
    store here rather than inside the sweep so `git_worktree` keeps no
    conductor dependency. It is not empty even at our own start: a second
    campaign on the same workspace is an ordinary thing to do, and sweeping
    on the assumption that nothing else is live would delete a running
    campaign's worktrees out from under its branches mid-experiment.

    Best-effort throughout — a campaign that cannot tidy up is still a
    campaign that can run, and `create_experiment_worktree` reports the real
    error later for any branch that actually collides.
    """
    from labpilot.research_engine.agents.git_worktree import reconcile_worktrees

    try:
        result = reconcile_worktrees(workspace.root, live_branches=_live_branches(store, workspace))
    except Exception:  # noqa: BLE001 — startup tidying must not stop a campaign
        logger.exception("worktree reconciliation failed; continuing")
        return
    if result.removed:
        logger.info("cleared %d stale experiment worktree(s)", len(result.removed))
    if result.failed:
        logger.warning(
            "%d stale worktree(s) could not be cleared; branches for those "
            "experiment keys will fail to check out: %s",
            len(result.failed),
            ", ".join(str(p) for p in result.failed),
        )
    _untrack_shared_state(workspace)


#: Session statuses whose worktrees a startup sweep must leave alone.
_LIVE_SESSION_STATUSES = frozenset({"running", "paused"})

#: Where a campaign records the process running it, so a later sweep can tell a
#: live campaign from one that died without saying so.
_OWNER_KEY = "owner"


def claim_session_ownership(store, session_id: str) -> None:
    """Record which process is running this campaign.

    Status alone cannot answer "is this campaign alive?". Every transition to a
    terminal state runs inside this loop, so a process killed by SIGKILL, OOM or
    power loss leaves its session `running` for good — and the startup sweep,
    which preserves live sessions' worktrees, then preserves exactly the ones it
    exists to reclaim. Their experiment keys stay checked out and every later
    fan-out over them fails with git's "already checked out".

    A pid and host answer it directly, with no threshold to tune: on this host a
    dead pid means the campaign is gone. Kept in `metadata` rather than a new
    column, so no migration is needed.
    """
    session = store.get_session(session_id)
    if session is None:
        return
    metadata = dict(session.metadata or {})
    metadata[_OWNER_KEY] = {"pid": os.getpid(), "host": socket.gethostname()}
    try:
        store.update_session_metadata(session_id, metadata)
    except Exception:  # noqa: BLE001 — a campaign runs fine without the stamp
        logger.exception("cannot record session ownership for %s", session_id)


def _owner_is_gone(session: Any) -> bool:
    """True only when this host can prove the owning process is dead.

    Conservative in every other case — no stamp (a session from before this
    existed), another host, an unreadable value — because guessing wrong deletes
    a running campaign's checkouts mid-experiment, while over-keeping costs a
    stale worktree that the next provable case clears.
    """
    owner = (getattr(session, "metadata", None) or {}).get(_OWNER_KEY)
    if not isinstance(owner, dict) or owner.get("host") != socket.gethostname():
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        # Signal 0 checks existence without delivering anything.
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # PermissionError and friends mean it exists but is not ours.
        return False
    return False


def _live_branches(store, workspace: Workspace) -> set[str]:
    """Branch names the startup sweep must leave alone.

    Inverted from "which are live" to "which are not provably ours to delete",
    because this store cannot see every campaign that shares the repository.
    `list_sessions()` filters by competition, and each competition keeps its
    own `knowledge.db`, so another competition's running campaign is not merely
    absent from the live set — it is unreadable from here. Treating unknown as
    dead would delete its worktrees mid-experiment, which is the failure the
    empty-set version had, narrowed rather than removed.

    So a worktree is swept only when its session is one this store knows *and*
    reports finished. Live sessions, sessions belonging to another
    competition, and branches whose name does not parse are all kept. Keeping
    a stale worktree costs one `create_experiment_worktree` retry that the
    next sweep clears; deleting a live one costs a running experiment.

    `paused` counts as live. Resuming needs those branches still checked out,
    so sweeping them would turn a pause into a silent loss of work.
    """
    from labpilot.research_engine.agents.git_worktree import list_registered_worktrees

    try:
        known = {session.id: session for session in store.list_sessions()}
    except Exception:  # noqa: BLE001 — an unreadable store means sweep nothing
        logger.exception("cannot list sessions; leaving every worktree in place")
        return _all_registered_branches(workspace)
    try:
        registered = list_registered_worktrees(workspace.root)
    except Exception:  # noqa: BLE001
        logger.exception("cannot list worktrees; leaving every worktree in place")
        return set()

    preserve: set[str] = set()
    for branch in registered.values():
        if not branch:
            continue
        session = known.get(_session_of(branch))
        if session is None:
            preserve.add(branch)
            continue
        if str(session.status).strip().lower() in _LIVE_SESSION_STATUSES and not _owner_is_gone(
            session
        ):
            preserve.add(branch)
    return preserve


def _all_registered_branches(workspace: Workspace) -> set[str]:
    """Every registered branch — the set that makes a sweep a no-op.

    Used when the session table cannot be read: not knowing which campaigns
    are live is a reason to delete nothing, since the cost of over-keeping is
    a stale worktree that the next successful sweep clears, and the cost of
    over-deleting is a running campaign losing its checkout.
    """
    from labpilot.research_engine.agents.git_worktree import list_registered_worktrees

    try:
        return {b for b in list_registered_worktrees(workspace.root).values() if b}
    except Exception:  # noqa: BLE001
        return set()


def _session_of(branch: str) -> str:
    """The session segment of a `research/<session>/<experiment>` branch."""
    parts = branch.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "research" else ""


def _untrack_shared_state(workspace: Workspace) -> None:
    """Stop git copying shared research state into every worktree.

    A `.gitignore` pattern does not untrack a file that is already tracked, so
    a workspace scaffolded before `SHARED_STATE_IGNORES` still has
    `knowledge.db` and `runs/` in the index — and every worktree checkout
    copies them (measured 105 MB per branch). Ignoring fixes new workspaces;
    only the index fixes existing ones.

    Done here rather than in `ensure_required_ignores` because it rewrites a
    user's index, which belongs with the feature that actually needs it and
    not in a helper every command runs. `--cached` keeps the files on disk:
    this unstages them, it does not delete anyone's database.

    Returns early when there is no repository, and checks that *before*
    reaching for a git tool: `open_git_tool` runs `Repo.init()` plus a
    bootstrap commit on a directory that is not one, so calling it first
    created a repository in a workspace the user had deliberately kept out of
    version control — the same side effect `reconcile_worktrees` refuses, from
    the same unattended startup path, one call later.
    """
    from labpilot.research_engine.agents.git_worktree import is_git_repo
    from labpilot.research_engine.git import open_git_tool
    from labpilot.workspace import SHARED_STATE_IGNORES

    if not is_git_repo(workspace.root):
        logger.debug("%s is not a git repository; nothing to untrack", workspace.root)
        return

    pathspecs = _as_pathspecs(SHARED_STATE_IGNORES)
    try:
        tool = open_git_tool(workspace.root)
        tracked = tool.execute("ls-files", "--", *pathspecs).strip()
    except Exception:  # noqa: BLE001 — no repo, or git unavailable
        logger.debug("cannot inspect the index for shared state; skipping untrack")
        return
    if not tracked:
        return
    try:
        # `--ignore-unmatch` because `git rm` aborts on the *whole* invocation
        # when any one pathspec matches nothing — a workspace with `runs/`
        # tracked but no `knowledge.db` would otherwise untrack neither.
        tool.execute("rm", "-r", "--cached", "--quiet", "--ignore-unmatch", "--", *pathspecs)
    except Exception:  # noqa: BLE001
        logger.exception("cannot untrack shared research state; worktrees will copy it")
        return
    logger.info(
        "untracked %d shared research file(s) so branch worktrees stop copying them",
        len(tracked.splitlines()),
    )


def _as_pathspecs(patterns: tuple[str, ...]) -> list[str]:
    """Gitignore patterns as git pathspecs.

    Not the same language, despite looking alike: a leading `/` anchors a
    gitignore pattern to the repository root, but means an absolute path to a
    pathspec — git rejects `/runs/` outright with `fatal: Invalid path
    '/runs'`. Stripping it gives the same meaning, since pathspecs are already
    relative to the repository root.

    Derived from the constant rather than written out again so a pattern
    added there cannot silently go unhandled here.
    """
    return [pattern.lstrip("/") for pattern in patterns]


def _fan_out_with_task_cleanup(store, task_id: str, *args: Any, **kwargs: Any) -> Any:
    """`_fan_out_experiment`, cancelling the superseded task if it raises.

    The task was enqueued for the sequential dispatch this step is about to
    replace. When the fan-out returns records the caller cancels it, and when
    the fan-out *declines* the caller goes on to dispatch it — but an exception
    escaping in between left it `pending` forever, with no worker that would
    ever claim it. A `KeyboardInterrupt` at the approval prompt is the ordinary
    way that happens.

    `BaseException`, for that reason, and re-raised untouched: the operator's
    interrupt still ends the campaign.
    """
    try:
        return _fan_out_experiment(store, *args, **kwargs)
    except BaseException:
        try:
            store.update_task_status(task_id, "cancelled", error="fan-out interrupted")
        except Exception:  # noqa: BLE001 — must not displace the original
            logger.exception("cannot cancel task %s after a failed fan-out", task_id)
        raise


def _experiment_agent(*, llm_client: Any | None) -> Any | None:
    """The specialist every branch runs, or None when none is registered.

    Built once per campaign rather than per fan-out step. Each call
    constructs a coding tool, two specialists, a fresh event bus and a fresh
    subscriber set, and every step's branches would otherwise publish to a
    different bus than the last — so no subscriber could hold state across
    steps even if it wanted to.

    No `dry_run` argument: `_branch_task` always writes `dry_run` into the
    task metadata, and `ExperimentSpecialist` prefers metadata over its own
    default (`agents/experiment.py:78`), so a registry-level default could
    only ever disagree with the step it was built for.
    """
    from labpilot.research_engine.agents.catalog import build_default_specialist_registry

    registry = build_default_specialist_registry(llm_client=llm_client)
    candidates = registry.candidates(capability="run_experiment")
    return candidates[0].agent if candidates else None


def _branch_task(branch: Any, cohort_id: str, *, dry_run: bool, submit: bool) -> Any:
    """The `AgentTask` one branch runs.

    `cohort_id` travels in metadata because that is how it reaches the
    `ExperimentCompleted` event, and the promotion subscriber returns early
    without one — a fan-out whose branches carry no cohort id runs K
    experiments and compares none of them.
    """
    from labpilot.research_engine.agents.models import AgentTask

    return AgentTask(
        id=f"T-{branch.hypothesis_id}",
        capability="run_experiment",
        description=f"branch {branch.hypothesis_id}",
        metadata={
            "plan_id": branch.plan_id,
            "hypothesis_id": branch.hypothesis_id,
            "cohort_id": cohort_id,
            "dry_run": dry_run,
            # Carried from the step rather than pinned off: `dry_run` is read
            # from the step's args, and silently dropping its sibling is how
            # a step that asked to submit stops submitting once fanned out.
            "submit": submit,
        },
    )


def _branch_context(workspace: Workspace) -> Any:
    from labpilot.research_engine.context.models import ContextBundle, ContextRequest

    return ContextBundle(
        request=ContextRequest(competition=workspace.competition, goal=workspace.goal or "")
    )


def _maybe_mint_on_stagnation(
    workspace: Workspace, budget_state: BudgetState, budget_cfg: BudgetConfig
) -> None:
    """Propose a change of direction once per plateau, on the edge into it.

    Edge-triggered, not level-triggered: `steps_since_improvement` only grows
    while a campaign is stuck, so minting whenever it is high would add a
    near-duplicate hypothesis on every remaining step. The latch clears on the
    next improvement, so a later plateau in the same campaign mints again
    rather than staying suppressed for good.

    Runs before `persist_budgets`, so `score_events` and the latch are saved
    by the same write — a crash before that write loses both together rather
    than leaving them disagreeing. The mint itself is a separate write to a
    separate store (the hypothesis lands in `knowledge_dir` the moment
    `HypothesisStore.create` returns, well before `persist_budgets` commits
    the latch), so a crash in that narrow window can still leave a hypothesis
    on disk with the latch unset. On resume this reopens the same plateau
    rather than duplicating it exactly: the technique-exclusion dedup in
    `_untried_technique` sees the orphaned hypothesis and will not name its
    technique again, so the worst case is a second hypothesis for a different
    untried technique, not a byte-for-byte duplicate.

    The latch follows the mint's *result*, not the attempt. A plateau can
    begin with nothing to propose and acquire something mid-way: the M8-5 gate
    reopens `analyze_competition` precisely because the campaign is stuck, so
    the vocabulary grows during exactly the plateau this would otherwise have
    given up on. Suppressing repeats of a mint that happened is the intent;
    suppressing retries of one that did not is a different thing.

    Wrapped in its own guard for the same reason `scoring._techniques_for` is:
    `_record_experiment_outcome` runs inside the dispatch loop's outer
    try/except, so an escape here would not stay local — it would land as a
    dispatch error and count a *successful* experiment as a failure against
    the circuit breaker.
    """
    try:
        # Computed once and threaded through: this runs on every recorded
        # experiment, not just stagnant ones, so re-deriving the same
        # summary/window a second time inside mint_stagnation_hypothesis
        # would pay the comparable_tail scan twice on every single step.
        summary = score_summary(budget_state, budget_cfg)
        window = stagnation_window(budget_state, budget_cfg, summary=summary)
        if not window:
            budget_state.stagnation_mint_fired = False
            return
        if budget_state.stagnation_mint_fired:
            return
        minted = mint_stagnation_hypothesis(
            workspace, budget_state, budget_cfg, window=window, summary=summary
        )
        if minted is not None:
            budget_state.stagnation_mint_fired = True
    except Exception:  # noqa: BLE001 — a failed mint must not cost the score its record
        logger.exception("stagnation mint failed; recording the score without it")


def _record_experiment_outcome(
    store,
    session_id: str,
    *,
    succeeded: bool,
    error: str = "",
    workspace: Workspace | None = None,
    execution_id: str | None = None,
) -> None:
    """Fold one experiment outcome into the breaker's counters and persist.

    On a success that produced a comparable score, also append it to
    `score_events` and derive `metric_history`/`last_metric` from the series.

    The derivation happens *here*, at the one place the series changes, rather
    than as an invariant every writer must remember to re-establish. A session
    with no events is therefore never touched — which matters, because
    `metric_history` predates this series and a campaign resumed from an older
    session still has values in it.
    """
    session = store.get_session(session_id)
    if session is None:
        return
    budget_cfg, budget_state = load_budget_pair(session)
    budget_state.record_execution(succeeded=succeeded, error=error)
    if succeeded and workspace is not None and execution_id:
        event = score_event_for(workspace, execution_id, fallback_maximize=budget_cfg.maximize)
        if event is not None:
            # A resumed session's stored readings name no metric, so they
            # cannot be compared against a keyed one. They leave the derived
            # view rather than being flattened into it — say so, because it
            # changes what `plateau` sees.
            if not budget_state.score_events and budget_state.metric_history:
                logger.info(
                    "%d stored metric readings name no metric and are excluded from "
                    "the comparison window now that %s is recorded",
                    len(budget_state.metric_history),
                    event.metric_name,
                )
            previous = budget_state.score_events[-1] if budget_state.score_events else None
            budget_state.score_events.append(event)
            # Every event stays in the series — exit criterion 1 and the
            # stagnation mint both cite experiments by id, and evicting them is
            # exactly what the design doc refused to do. Only the *derived*
            # window narrows, to the trailing run measuring one metric, so
            # `plateau` never takes a max-minus-min across scales.
            comparable = comparable_tail(budget_state.score_events)
            if previous is not None and not metric_names_match(
                previous.metric_name, event.metric_name
            ):
                logger.warning(
                    "primary metric changed from %s to %s; %d earlier reading(s) stay "
                    "on record but leave the comparison window",
                    previous.metric_name,
                    event.metric_name,
                    len(budget_state.score_events) - len(comparable),
                )
            budget_state.metric_history = [e.value for e in comparable]
            budget_state.last_metric = event.value
            # Reset here, at the append, and nowhere else. `steps_since_success`
            # resets on any successful execution, which is why it cannot see a
            # run that succeeds and writes a placeholder metric; this counter is
            # keyed on the series it guards.
            budget_state.steps_since_new_score = 0
            logger.info(
                "recorded %s=%s for %s", event.metric_name, event.value, event.experiment_id
            )
            _maybe_mint_on_stagnation(workspace, budget_state, budget_cfg)
    persist_budgets(store, session_id, budget_cfg, budget_state)


def _needs_guidance_reason(config: Any, state: Any) -> str:
    """Which condition asked for a person, in the terms they can act on.

    A bare `stop:needs_guidance` reproduces the complaint this milestone
    exists to answer — a campaign that ended without saying why.
    """
    limit = getattr(config, "max_steps_without_score", None)
    if limit is not None and getattr(state, "steps_since_new_score", 0) >= limit:
        return (
            f"{state.steps_since_new_score} step(s) produced no new comparable "
            "score. The campaign is running and measuring nothing — check that "
            "experiments are writing metrics."
        )
    return (
        f"{getattr(state, 'consecutive_unmapped', 0)} consecutive step(s) chose "
        "an action no registered tool can perform. The campaign has nothing "
        "eligible to run; see the recorded suggestions for what was missing."
    )


def _fail_session_on_degraded_llm(store, session_id, record, decisions, exc) -> None:
    """Record a strict-mode abort before it propagates.

    The session is marked so a later `conduct continue` sees why it ended
    rather than finding a run that simply stopped short.
    """
    record.rationale = f"{record.rationale} | strict LLM abort: {exc}"
    store.append_decision(record)
    decisions.append(record)
    try:
        store.increment_metric(session_id, "tasks_failed")
        store.update_session_status(session_id, "failed")
    except Exception:  # noqa: BLE001 — the abort matters more than the bookkeeping
        logger.warning("could not mark session %s failed", session_id)


def _objective_unmet(config: Any, state: Any) -> bool:
    """True when a metric target was set and the best result has not reached it."""
    target = getattr(config, "target_value", None)
    if getattr(config, "target_metric", None) is None or target is None:
        return False
    last = getattr(state, "last_metric", None)
    if last is None:
        return True
    # Same check `evaluate_stops` makes before firing on a target, and for the
    # same reason: `last_metric` is a bare number. Without it a `cv_rmse` of
    # 1789.7 reads as having met an `mse` target of 5 — 1789.7 > 5 is the
    # *unmet* branch here, but an accuracy of 0.9 against that target is not,
    # and this function's answer is what keeps a campaign going after an
    # advisory stop. Measured on rogii 2026-08-12: the competition metric is
    # mean_squared_error and the pipeline records cv_rmse.
    if not _measures_the_target(config, state):
        return True
    # `getattr(..., True)` matches `BudgetConfig.maximize`'s own default. This
    # read used `False`, so the two disagreed about the same field whenever the
    # attribute was missing — one more place where direction was assumed rather
    # than resolved.
    return last < target if getattr(config, "maximize", True) else last > target


def _measures_the_target(config: Any, state: Any) -> bool:
    """Whether `last_metric` is a reading of the metric the target names.

    Duck-typed like its caller, which takes `Any` so tests can pass stand-ins.
    An unknown metric name keeps the older, looser behaviour, exactly as
    `budgets._last_metric_matches_target` does.
    """
    from labpilot.research_engine.conductor.budgets import metric_names_match

    events = getattr(state, "score_events", None)
    if not events:
        return True
    recorded = events[-1].metric_name
    target_metric = getattr(config, "target_metric", None)
    if metric_names_match(recorded, target_metric):
        return True
    # Said out loud, because the consequence is invisible otherwise: while the
    # names disagree this returns False forever, so `_objective_unmet` is always
    # True, `evaluate_stops` cannot fire `metric_target` either, and a campaign
    # that has genuinely reached its goal can only end on max_steps, plateau or
    # budget. Nothing renames either side, so the mismatch is a permanent
    # property of the workspace rather than a transient one, and
    # `metric_names_match` only bridges a `_MEASUREMENT_PREFIXES` prefix —
    # `holdout_auc` against `auc` does not match. Once per campaign, so a
    # 30-step run logs it once and not thirty times.
    # Latched on the campaign's own state, so a second campaign in the same
    # process is told too. `setattr` is guarded because this function is
    # duck-typed for stand-ins, and a stand-in that refuses the attribute
    # should warn every step rather than crash the loop.
    if not getattr(state, "metric_mismatch_reported", False):
        try:
            state.metric_mismatch_reported = True
        except (AttributeError, ValueError):
            logger.debug("could not latch the metric-mismatch warning on %r", type(state))
        logger.warning(
            "The pipeline records %r but the target names %r, and nothing maps between "
            "them — the objective can never be reported met, so this campaign will run "
            "to its step or time budget. Set `target_metric` to the key the pipeline "
            "writes, or have the pipeline write the key the target names.",
            recorded,
            target_metric,
        )
    return False


def _latest_plan_id(workspace: Workspace) -> str | None:
    """Latest plan the Engineer would accept, or None.

    Taking the highest id outright targeted plans that had already finished —
    the Engineer then refused with "status=done; need ready or in_progress" and
    the campaign lost a step. The first fix narrowed the *preference* but kept a
    fallback to the newest plan of any status, so the refusal still happened
    whenever every plan was done. There is no useful answer in that case:
    returning None lets the caller offer `generate_plan` instead of burning a
    step on a run that cannot succeed.
    """
    from labpilot.research_engine.intelligence.paths import store_is_absent
    from labpilot.research_engine.planner.store import PlanStore

    if store_is_absent(workspace.knowledge_dir, workspace.competition):
        return None
    store = None
    try:
        # Constructed inside the guard: opening the store is where a corrupt
        # database actually fails, and it used to fail *outside* it — so the
        # handler written for this case never saw the case.
        store = PlanStore(workspace.knowledge_dir, workspace.competition)
        selectable = store.selectable_plan_ids()
    except Exception:
        # Absence is answered above, so this handler now holds only genuine
        # faults: a locked database, a schema the code no longer matches, a
        # permissions problem. All three used to return exactly what "no plans
        # yet" returns, and say nothing. The answer below is a guess, and the
        # log is the only place that admits it. M20, 2026-08-09.
        logger.exception("cannot read plans for %s; treating as none", workspace.competition)
        return None
    finally:
        if store is not None:
            store.close()
    return selectable[-1] if selectable else None


def _next_hypothesis_id(workspace: Workspace) -> str | None:
    """Highest-confidence untested hypothesis, or None when there are none.

    This is what lets a campaign iterate: once the baseline plan exists, the
    next plan has to be built against a hypothesis rather than re-requesting an
    idempotent baseline.
    """
    from labpilot.research_engine.intelligence.paths import hypotheses_are_absent
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    if hypotheses_are_absent(workspace.knowledge_dir, workspace.competition):
        return None
    try:
        store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:
        # See `_latest_plan_id`: absence is answered above, so this is a fault.
        logger.exception(
            "cannot read hypotheses for %s; treating as none to test",
            workspace.competition,
        )
        return None
    if not proposed:
        return None
    # Ranked by posterior, not by the prior generation wrote once and never
    # revisited. `confidence` is set at `create()` and updated by nothing, so
    # sorting on it alone kept `hyp:H-010` at 0.99 through the runs that
    # disproved it, and ranked a technique measured as *harmful* above one
    # nobody had tried.
    from labpilot.research_engine.intelligence.hypothesis.selection import rank_hypotheses

    ranked = rank_hypotheses(proposed, workspace.knowledge_dir, workspace.competition)
    return ranked[0].id if ranked else None


def _baseline_is_done(workspace: Workspace) -> bool:
    """Whether the campaign may stop asking for a baseline and start iterating.

    M23 step 8. `_baseline_plan_exists` answered *"was a plan object compiled?"*
    to a caller asking *"has the baseline been done?"* — so a campaign flipped to
    research mode on the strength of a file existing, whatever the pipeline it
    described actually scored. That is how rogii spent two weeks minting
    hypotheses over a pipeline 91x worse than one line of code.

    Under enforcement the answer is the gate's: `passed` or `waived`. Otherwise
    it is the old one, because the rollout's whole shape is that step 8 is a
    config flip and observe-only must not change what a campaign does.
    """
    root = getattr(workspace, "root", None)
    if root is not None:
        try:
            from labpilot.research_engine.execution.baseline.gate import (
                _enforcement_enabled,
                evaluate_gate,
            )

            if _enforcement_enabled():
                # `blocks_research` and not `state == "passed"`: a waived gate is
                # a decision someone recorded, and the campaign proceeds.
                return not evaluate_gate(Path(root), enforced=True).blocks_research
        except Exception as exc:  # noqa: BLE001 — a gate that cannot run must
            # not force a baseline recompile over the top of existing work.
            logger.warning("Baseline gate unavailable, falling back to plan lookup: %s", exc)
    return _baseline_plan_exists(workspace)


def _baseline_plan_exists(workspace: Workspace) -> bool:
    """True when a baseline plan has already been compiled for this competition.

    Retained only as `_baseline_is_done`'s observe-only fallback. It answers a
    different question from the one its caller asks, which is the defect step 8
    exists to fix.
    """
    from labpilot.research_engine.artifacts.plan import PlanArtifacts
    from labpilot.research_engine.intelligence.paths import store_is_absent

    if store_is_absent(workspace.knowledge_dir, workspace.competition):
        return False
    artifacts = None
    try:
        artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
        plans = artifacts.list()
    except Exception:
        # **Raised, not answered.** The other reads here degrade to a negative
        # because the cost of being wrong is a wasted step. This one's negative
        # means *compile a baseline*, over the top of whatever is already there
        # — so a fault that reads as "no baseline" destroys work rather than
        # delaying it. Reported on PR #120: moving the store construction into
        # the guard turned a crash into exactly that. A campaign that cannot
        # read its own plans should stop.
        logger.exception("cannot read plans for %s", workspace.competition)
        raise
    finally:
        if artifacts is not None:
            artifacts.close()
    return any((p.metadata or {}).get("plan_kind") == "baseline" for p in plans)


def _latest_execution_id(workspace: Workspace) -> str | None:
    """Most recent execution id, or None when nothing has run yet."""
    from labpilot.research_engine.shared.experiments.graph import build_graph

    try:
        graph = build_graph(
            workspace.effective_runs_dir,
            workspace.competition,
            knowledge_dir=workspace.knowledge_dir,
        )
    except Exception:
        # "Nothing has run yet" is what the conductor plans against, so a graph
        # it cannot build must not read as an empty one.
        logger.exception(
            "cannot build the execution graph for %s; treating as nothing run",
            workspace.competition,
        )
        return None
    nodes = sorted(graph.nodes.values(), key=lambda e: e.created_at)
    return nodes[-1].id if nodes else None


#: Ask an operator to settle one schema question. Returns the chosen value, or
#: None to leave it open. Injected by the CLI exactly like `approval_prompt` —
#: with one deliberate asymmetry: there is **no `auto_answer` counterpart to
#: `auto_approve`**. Absent a prompt, an approval falls back to auto-approve; a
#: schema question has nothing to fall back to, so it blocks. An option that
#: could answer one unattended must not exist, because `--yes` would eventually
#: be wired to it.
SchemaPrompt = Callable[[SchemaQuestion], str | None]


def _schema_stamp(root: Path) -> tuple[int, int, int, int]:
    """A cheap signature of the two files a schema question is derived from.

    `open_questions` parses `profile.json` — on rogii that is 14 KB carrying
    every column, the evidence plane and up to 200 paths — and this loop can run
    unbounded since M17. Neither file changes *within* a step, so two `stat`
    calls answer "is the previous result still good?" for a fraction of the cost.
    """
    stamps: list[int] = []
    for name in ("profile.json", ANSWERS_FILENAME):
        try:
            info = (root / name).stat()
            stamps.extend((info.st_mtime_ns, info.st_size))
        except OSError:
            stamps.extend((0, 0))
    return (stamps[0], stamps[1], stamps[2], stamps[3])


def _answer_schema_questions(
    questions: list[SchemaQuestion],
    root: Path,
    prompt: SchemaPrompt | None,
    progress: Callable[[str], None],
) -> bool:
    """Ask and record. True when every question is now settled.

    Recording writes `schema_answers.json`, which changes the answers
    fingerprint the profile was built with, so the profile it superseded reads
    stale and `prepare_workspace` re-derives it on the next run.

    A failed write is a blocked campaign, not a crashed one: a read-only
    workspace or a full disk used to let `OSError` escape `run_until_stop`
    entirely, ending the run with a traceback and no decision record — where
    every other operator-facing prompt in this loop ends in one.
    """
    if prompt is None:
        return False
    for question in questions:
        answer = prompt(question)
        if not answer:
            progress(f"{question.field} left unanswered")
            return False
        try:
            record_answer(root, question.field, answer)
        except (OSError, ValueError) as exc:
            progress(f"{question.field} could not be recorded: {exc}")
            return False
        progress(f"{question.field} answered: {answer}")
    return True


def run_until_stop(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
    max_steps: int | None = 8,
    auto_approve: bool = False,
    approval_prompt: ApprovalPrompt | None = None,
    on_progress: ProgressCallback | None = None,
    autonomy: int = 0,
    campaign_mode: bool = True,
    prefer_offline: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
    schema_prompt: SchemaPrompt | None = None,
    branches: int = 1,
) -> list[DecisionRecord]:
    """Run until stop, budget, operator pause, or ``max_steps`` if one is set.

    ``max_steps=None`` runs unbounded, which is what the CLI asks for: a
    campaign should end on its objective, not on a counter. The default stays
    ``8`` here rather than ``None`` so no in-process caller changes behaviour
    by upgrading — a bounded run is what a library caller asking for "some
    steps" means, and an unbounded one is a hang in a test suite with no
    per-test timeout.

    ``branches`` is the fan-out width (M11). The default of 1 is the
    sequential path this loop has always run, unchanged: every fan-out
    decision is guarded on ``branches > 1``, so a caller that does not ask
    for parallel branches cannot get one.

    When online policy fails, asks the operator (allow / deny / retry) before
    using the deterministic offline order — unless ``prefer_offline`` or
    ``auto_approve`` (``--yes``) is set.
    """
    # Every micro-agent invocation in this campaign is recorded: which agent,
    # whether the LLM or its rule engine produced the answer, and on failure
    # what kind. M14 2b and 3 are both blocked on having that as *data* rather
    # than log lines, and it can only be collected while the run happens.
    with recording_provenance(
        workspace.knowledge_dir, workspace.competition, session_id=session_id
    ):
        return _run_until_stop_inner(
            store,
            workspace,
            session_id,
            registry,
            llm_client=llm_client,
            max_steps=max_steps,
            auto_approve=auto_approve,
            approval_prompt=approval_prompt,
            on_progress=on_progress,
            autonomy=autonomy,
            campaign_mode=campaign_mode,
            prefer_offline=prefer_offline,
            offline_fallback_prompt=offline_fallback_prompt,
            schema_prompt=schema_prompt,
            branches=branches,
        )


def _run_until_stop_inner(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
    max_steps: int | None = 8,
    auto_approve: bool = False,
    approval_prompt: ApprovalPrompt | None = None,
    on_progress: ProgressCallback | None = None,
    autonomy: int = 0,
    campaign_mode: bool = True,
    prefer_offline: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
    schema_prompt: SchemaPrompt | None = None,
    branches: int = 1,
) -> list[DecisionRecord]:
    scheduler = Scheduler(store, registry, workspace, llm_client=llm_client)
    decisions: list[DecisionRecord] = []
    #: Cached across iterations: neither file changes within a step, and parsing
    #: the profile every time is real work in an unbounded loop.
    schema_stamp: tuple[int, int, int, int] | None = None
    schema_questions: list[SchemaQuestion] = []
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    branch_agent: Any = None
    if branches > 1:
        # Stamped before the sweep, so this campaign's own worktrees can never
        # be candidates for it.
        claim_session_ownership(store, session_id)
        _reconcile_stale_worktrees(store, workspace)
        branch_agent = _experiment_agent(llm_client=llm_client)

    ensure_metrics(store, session_id)
    budget_cfg, budget_state = load_budget_pair(session)
    budget_state.ensure_wall_start()
    # A new run of the loop clears the guidance counters, and that *is* the
    # resume. Without it a `needs_guidance` pause could never be picked up:
    # the counters that tripped it are persisted at their thresholds, so the
    # first `evaluate_stops` of the resumed run re-fires the same stop before
    # anything is dispatched — the campaign takes zero steps, every time,
    # however thoroughly the operator fixed what it asked about.
    #
    # Cleared here rather than in `conduct continue` so any resumer gets it,
    # and unconditionally rather than on a stored stop reason: invoking the
    # loop again is the operator saying "try again", and a campaign still
    # unable to progress simply spends the counters afresh and pauses again.
    #
    # The M20 breaker's counters are deliberately not cleared. `failing` parks
    # a session in `failed`, which resuming needs `--session` to reach at all,
    # and that friction is the point of the distinction.
    budget_state.steps_since_new_score = 0
    budget_state.consecutive_unmapped = 0
    persist_budgets(store, session_id, budget_cfg, budget_state)

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    consecutive_stop_overrides = 0
    policy_kw: dict[str, Any] = {
        "prefer_offline": prefer_offline,
        "auto_offline_fallback": auto_approve,
        "offline_fallback_prompt": offline_fallback_prompt,
    }

    # Repair research memory before acting on it. A claim no measurement
    # supports steers every decision this loop is about to make, and the repair
    # must not wait for a *successful experiment* — a campaign that completes
    # none is exactly when memory is most likely to be leading it astray.
    # Measured 2026-08-07: a full campaign ran with 45 false `vit` claims intact
    # because revalidation only fired from `record_successful_execution`.
    try:
        from labpilot.research_engine.evidence.belief_repair import (
            rederive_beliefs_from_cards,
        )
        from labpilot.research_engine.evidence.repair import repair_card_directions
        from labpilot.research_engine.execution.outcome import revalidate_outcome_claims

        # Cards first: revalidation reads their verdicts, so an inverted card
        # would otherwise have its inverted conclusion re-confirmed. Measured
        # 2026-08-07, all 15 rogii cards were built as `maximize=True` on an MSE
        # competition, which recorded the one real improvement as `rejected`.
        # `workspace.root` matters for a workspace whose only `metric.direction`
        # is on the run's own competition.json: without it repair falls back to
        # the knowledge copy and the Analyze profile, and silently no-ops when
        # neither carries a direction.
        reoriented = repair_card_directions(
            workspace.knowledge_dir,
            workspace.competition,
            workspace_root=workspace.root,
        )
        if reoriented:
            _progress(
                f"Re-oriented {len(reoriented)} evidence card(s) built with the "
                "wrong metric direction"
            )

        # Beliefs are recomputed from the repaired cards, not stepped again.
        # Repairing a card does not retract the belief step it already caused,
        # so without this the loop plans against the pre-repair compass.
        rebuilt = rederive_beliefs_from_cards(workspace.knowledge_dir, workspace.competition)
        if rebuilt:
            _progress(f"Re-derived {len(rebuilt)} belief(s) from repaired evidence")

        # Overlays reach the *model*, so a stale one costs more than a stale
        # row. `upsert_skill_overlay` returns early on a known lesson id, so a
        # lesson written from an inverted verdict is permanent until this runs.
        # Measured 2026-08-08: every rogii overlay said `Avoid: SWA` — the only
        # technique that ever improved the metric — long after its card was
        # re-oriented to `accepted`.
        from labpilot.research_engine.evidence.overlay_repair import (
            record_references_in_overlays,
            repair_skill_overlays,
        )

        relearned = repair_skill_overlays(
            workspace.root, workspace.knowledge_dir, workspace.competition
        )
        if relearned:
            _progress(f"Rebuilt {len(relearned)} skill overlay(s) from repaired evidence")
        leaking = record_references_in_overlays(workspace.root)
        if leaking:
            # Not repaired here: the write-path guard in `outcome.py` owns this,
            # and a hit means a write site it does not cover.
            logger.warning("record reference still present in overlay(s): %s", ", ".join(leaking))

        from labpilot.research_engine.execution.technique.vocabulary import (
            recompute_technique_status,
        )

        status_changes = recompute_technique_status(workspace.knowledge_dir, workspace.competition)
        if status_changes:
            _progress(f"Recomputed status for {len(status_changes)} technique(s) from evidence")

        contested = revalidate_outcome_claims(
            knowledge_dir=workspace.knowledge_dir, competition=workspace.competition
        )
        if contested:
            _progress(f"Contested {len(contested)} claim(s) no measurement supports")
    except Exception as exc:  # noqa: BLE001 — never block a campaign on repair
        logger.warning("Claim revalidation at session start failed: %s", exc)

    step = -1
    while True:
        step += 1
        # A `for/else` cannot say "bound only when asked" — it would have to
        # iterate `range(max_steps or sys.maxsize)` and then report `max_steps`
        # on a run that never had one.
        if max_steps is not None and step >= max_steps:
            store.update_session_status(session_id, "paused")
            _progress(f"Reached max_steps={max_steps}")
            save_checkpoint(store, session_id, extra={"stop_reason": "max_steps"})
            break
        step_label = f"{step + 1}/{max_steps}" if max_steps is not None else f"{step + 1}"
        # Refresh each iteration so mid-session registration is visible.
        allowlist = set(registry.names())
        session = store.get_session(session_id)
        assert session is not None
        if session.status == "paused":
            _progress("Session paused by operator")
            break

        # A schema question is not a budget: it is the campaign discovering it
        # does not know what it is optimising. Asked where there is someone to
        # ask, and otherwise a stop — never a default, because a guess is frozen
        # into `profile.json` for every later run of this workspace.
        stamp = _schema_stamp(workspace.root)
        if stamp != schema_stamp:
            schema_stamp, schema_questions = stamp, open_questions(workspace.root)
        if schema_questions:
            answered = _answer_schema_questions(
                schema_questions, workspace.root, schema_prompt, _progress
            )
            # Stopping either way, and the reasons are different.
            #
            # Unanswered is the campaign waiting for a person. **Answered is the
            # campaign holding a description that has just been superseded**:
            # the answer changed the fingerprint, so `profile.json` now reads
            # stale, and `prepare_workspace` — a *plan task*, not a tool this
            # loop can dispatch — is what re-derives it. Continuing here would
            # run the rest of the campaign against the column the operator had
            # just rejected, with the question closed so nothing would ask
            # again. That is the silent-wrong-target failure this milestone
            # exists to remove, and it is worth one re-run to avoid.
            field = schema_questions[0].field
            if answered:
                rationale = (
                    f"stop:schema_question — answer recorded for {field}; the profile it "
                    "supersedes is now stale. Re-run to rebuild it and continue."
                )
            else:
                rationale = (
                    f"stop:schema_question — {field} is uncertain "
                    f"({schema_questions[0].context}); answer with "
                    f"`research schema answer {field} <value>`"
                )
            # `waiting`, not `failed`: the campaign is resumable the moment a
            # person answers, and `checkpoint.py` already counts `waiting` among
            # the active sessions. Nothing wrote it until now.
            store.update_session_status(session_id, "waiting")
            store.increment_metric(session_id, "unmet_goal")
            _progress(f"Stop condition: {rationale}")
            decisions.append(
                DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=rationale,
                    stop=True,
                    observe={
                        "stop_reason": "schema_question",
                        "answered": answered,
                        "questions": [{"id": q.id, "field": q.field} for q in schema_questions],
                    },
                )
            )
            store.append_decision(decisions[-1])
            break

        budget_cfg, budget_state = load_budget_pair(session)
        if not submit_tools_allowed(budget_cfg):
            # A campaign told never to submit must not be *offered* the tool.
            # Relying on the approval gate would not hold: `--yes` maps every
            # gated tool to `auto_approve`, so a non-interactive run has no
            # brake between "selected submit_learn" and "uploaded to Kaggle".
            allowlist -= SUBMIT_TOOLS
        # Before the stop evaluation, not after: the step that ends the
        # campaign is the one an operator most wants a progress line for, and
        # a line printed after the `break` is a line never printed.
        progress_line = goal_progress(budget_cfg, budget_state)
        if progress_line:
            _progress(progress_line)

        stop = evaluate_stops(budget_cfg, budget_state)
        if stop != "none":
            # `failing` is not a completion. Recording it as one would put the
            # campaign that could not run a single experiment in the same state
            # as the campaign that met its target, which is the distinction the
            # breaker exists to draw.
            #
            # `needs_guidance` is neither a completion nor a failure: nothing is
            # broken and nothing is finished — the campaign is waiting on a
            # person. It pauses, which is the status `conduct continue` resumes
            # and `latest_active_session` still finds, so picking it back up
            # needs no `--session`.
            status = _STOP_SESSION_STATUS.get(stop, "completed")
            store.update_session_status(session_id, status)
            if stop != "metric_target":
                store.increment_metric(session_id, "unmet_goal")
            rationale = f"stop:{stop}"
            if stop == "needs_guidance":
                why = _needs_guidance_reason(budget_cfg, budget_state)
                rationale = f"stop:{stop} — {why}"
                record_suggestion(
                    store,
                    session_id,
                    why,
                    kind="needs_guidance",
                    context={
                        "step": step,
                        "steps_since_new_score": budget_state.steps_since_new_score,
                        "consecutive_unmapped": budget_state.consecutive_unmapped,
                    },
                )
            if stop == "failing":
                # Say what broke. A bare `stop:failing` reproduces the original
                # complaint — a campaign that ended and did not say why.
                why = "; ".join(budget_state.recent_failures[-2:]) or "no successful experiment"
                rationale = (
                    f"stop:{stop} — {budget_state.consecutive_failures} consecutive "
                    f"failed execution(s), {budget_state.steps_since_success} step(s) "
                    f"since the last success. Last: {why}"
                )
            _progress(f"Stop condition: {rationale}")
            decisions.append(
                DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=rationale,
                    stop=True,
                    observe={
                        "stop_reason": stop,
                        "consecutive_failures": budget_state.consecutive_failures,
                        "steps_since_success": budget_state.steps_since_success,
                        "steps_since_new_score": budget_state.steps_since_new_score,
                        "consecutive_unmapped": budget_state.consecutive_unmapped,
                        "recent_failures": budget_state.recent_failures,
                    },
                )
            )
            store.append_decision(decisions[-1])
            if stop == "needs_guidance":
                # After the stop is on record, so the checkpoint an operator
                # reads counts the decision that ended the run.
                save_checkpoint(store, session_id, extra={"stop_reason": stop})
            break

        # One step is about to be spent. Counted here rather than on dispatch so
        # a step that never reaches a tool still counts: rogii's S-021 spent 30
        # steps without producing an execution, which no per-execution counter
        # would have noticed.
        budget_state.steps_since_success += 1
        budget_state.steps_since_new_score += 1
        persist_budgets(store, session_id, budget_cfg, budget_state)

        if campaign_mode:
            completed = [
                t.tool_name for t in store.list_tasks(session_id) if t.status == "completed"
            ]
            if prefer_offline:
                research = offline_next_research_action(completed, allowlist)
            else:
                # Observe (Context Engine retrieval) + think (LLM) both run
                # before the first dispatch message, so without this a campaign
                # step is several silent minutes on a local model.
                _progress(f"step {step_label}: observing + deciding …")
                started = time.monotonic()
                next_tool, _obs = decide_next(
                    store,
                    workspace,
                    session_id,
                    registry,
                    llm_client=llm_client,
                    # Already loaded for this step's stop evaluation. The
                    # stagnant clause gates the allowlist before the prompt is
                    # built, so the series has to arrive with the decision.
                    budgets=(budget_cfg, budget_state),
                    **policy_kw,
                )
                _progress(
                    f"step {step_label}: chose "
                    f"{next_tool.tool or 'stop'} ({time.monotonic() - started:.1f}s)"
                )
                if next_tool.stop or not next_tool.tool:
                    research = ResearchAction(
                        intent="stop",
                        rationale=next_tool.rationale or "stop",
                        stop=True,
                    )
                else:
                    research = ResearchAction(
                        intent=f"Run tool {next_tool.tool}",
                        rationale=next_tool.rationale,
                        suggested_tools=[next_tool.tool],
                    )
            # Re-read after policy/offline so same-step registration is visible.
            # The submit carve-out has to be re-applied: this is the allowlist
            # that reaches `map_research_action`, so a plain re-read would hand
            # the tools back at exactly the point they get selected.
            allowlist = set(registry.names())
            if not submit_tools_allowed(budget_cfg):
                allowlist -= SUBMIT_TOOLS
            plan = map_research_action(research, allowlist)
            if research.stop and _objective_unmet(budget_cfg, budget_state):
                # Goal persistence. The policy tends to call it done once it has
                # used each tool once ("no immediate next step in the
                # allowlist"), even with the target metric far away. A campaign
                # exists to pursue an objective, so an advisory stop is not
                # honoured while the target is unmet and budget remains —
                # reflection and the next hypothesis are still open moves.
                # Budgets, max_steps and repeated insistence still end the run.
                consecutive_stop_overrides += 1
                if consecutive_stop_overrides <= _MAX_STOP_OVERRIDES:
                    _progress(
                        f"Policy wanted to stop with the objective unmet "
                        f"({consecutive_stop_overrides}/{_MAX_STOP_OVERRIDES}); "
                        "continuing toward the target."
                    )
                    record_suggestion(
                        store,
                        session_id,
                        f"Policy stopped early with the objective unmet: {research.rationale}",
                        context={"step": step},
                    )
                    # Name the tool explicitly. Routing this by intent text let
                    # keyword matching hijack it: the phrase contains
                    # "hypothesis", which matches the ("plan", "baseline",
                    # "hypothesis") template before anything else, so the
                    # override dispatched generate_plan(baseline=True) instead
                    # of reflecting on the result it was reacting to.
                    override_tool = "reflect" if "reflect" in allowlist else "generate_plan"
                    research = ResearchAction(
                        intent=f"objective unmet — {override_tool} and continue",
                        rationale="objective still unmet; continuing",
                        suggested_tools=[override_tool],
                    )
                    plan = map_research_action(research, allowlist)
            else:
                consecutive_stop_overrides = 0

            if research.stop:
                record = DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=research.rationale,
                    stop=True,
                )
                store.append_decision(record)
                decisions.append(record)
                store.update_session_status(session_id, "completed")
                _progress(f"Conductor stop: {research.rationale}")
                break
            if plan.unmapped:
                ctx = build_suggestion_context(
                    intent=research.intent,
                    suggested_tools=research.suggested_tools,
                    missing_tools=plan.missing_tools,
                    competition=store.competition,
                    session_id=session_id,
                    goal=session.goal,
                )
                record_suggestion(
                    store,
                    session_id,
                    plan.suggestion or research.intent,
                    context=ctx,
                )
                record = DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=plan.suggestion or "no_capability",
                    stop=False,
                    observe={
                        "unmapped": True,
                        "intent": research.intent,
                        "missing_tools": list(plan.missing_tools),
                    },
                )
                store.append_decision(record)
                decisions.append(record)
                budget_state.consecutive_unmapped += 1
                persist_budgets(store, session_id, budget_cfg, budget_state)
                _progress(f"No capability: {plan.suggestion}")
                continue

            # The plan maps. Whatever the campaign could not reach before, it
            # can reach something now.
            if budget_state.consecutive_unmapped:
                budget_state.consecutive_unmapped = 0
                persist_budgets(store, session_id, budget_cfg, budget_state)

            prev_id: str | None = None
            for tool_step in plan.steps:
                decision_id = store.new_decision_id()
                deps = [prev_id] if prev_id else []
                # Resolve @latest against state *now*, after any earlier step in
                # this batch created a plan or execution.
                step_args = resolve_step_args(
                    tool_step.tool,
                    tool_step.args,
                    latest_plan_id=_latest_plan_id(workspace),
                    latest_execution_id=_latest_execution_id(workspace),
                    next_hypothesis_id=_next_hypothesis_id(workspace),
                    baseline_plan_exists=_baseline_is_done(workspace),
                )
                task = store.enqueue(
                    session_id,
                    tool_step.tool,
                    args=step_args,
                    decision_id=decision_id,
                    dependencies=deps,
                )
                record = DecisionRecord(
                    id=decision_id,
                    session_id=session_id,
                    tool_name=tool_step.tool,
                    rationale=research.rationale or research.intent,
                    args=step_args,
                    task_id=task.id,
                    observe={"step": step, "intent": research.intent},
                )
                approval = maybe_approve(
                    store,
                    session_id=session_id,
                    tool_name=tool_step.tool,
                    decision_id=decision_id,
                    task_id=task.id,
                    auto=auto_approve,
                    prompt=approval_prompt,
                    autonomy=autonomy,
                )
                if approval is not None:
                    record.approval = approval
                    if approval.decision == "reject":
                        store.update_task_status(task.id, "cancelled", error="operator rejected")
                        store.increment_metric(session_id, "tasks_blocked")
                        store.append_decision(record)
                        decisions.append(record)
                        _progress(f"Rejected {tool_step.tool}")
                        break
                if tool_step.tool in _EXPERIMENT_TOOLS and branches > 1:
                    fanned = _fan_out_with_task_cleanup(
                        store,
                        task.id,
                        workspace,
                        session_id,
                        step=step,
                        branches=branches,
                        rationale=research.rationale or research.intent,
                        llm_client=llm_client,
                        dry_run=bool(
                            step_args.get("dry_run", _DRY_RUN_DEFAULTS.get(tool_step.tool, True))
                        ),
                        submit=bool(step_args.get("submit", False)),
                        agent=branch_agent,
                        auto_approve=auto_approve,
                        approval_prompt=approval_prompt,
                        autonomy=autonomy,
                        progress=_progress,
                    )
                    if fanned is not None:
                        # The task enqueued for the sequential dispatch is not
                        # what ran: the branches did. Cancel it rather than
                        # leaving a pending row no worker will ever claim —
                        # with no `error`, because nothing failed. A non-null
                        # error here makes every fan-out step look like one
                        # more failure to anything scanning os_tasks. The
                        # reason lives on each branch's DecisionRecord instead.
                        store.update_task_status(task.id, "cancelled")
                        decisions.extend(fanned)
                        prev_id = None
                        continue
                _progress(f"Dispatch {tool_step.tool} ({task.id})")
                try:
                    result = scheduler.dispatch(task)
                    record.artifact_refs = [r.model_dump() for r in result.refs]
                    if tool_step.tool in {"submit", "submit_learn"}:
                        store.increment_metric(session_id, "submissions")
                        budget_cfg, budget_state = load_budget_pair(
                            store.get_session(session_id)  # type: ignore[arg-type]
                        )
                        budget_state.submissions += 1
                        persist_budgets(store, session_id, budget_cfg, budget_state)
                    if tool_step.tool in _EXPERIMENT_TOOLS:
                        # Ask what the *execution* did, not whether the call
                        # returned. `run_plan` reports a failed execution in its
                        # result and raises nothing, so counting a clean return
                        # as a success reset the breaker on every one.
                        #
                        # Measured 2026-08-09: a campaign with **8 executions,
                        # all failed** ran its full 8 steps, because 10 of its
                        # 16 dispatches were `run_plan` returning normally. The
                        # breaker built to stop exactly that never reached 3.
                        outcome = _experiment_outcome(result)
                        # `execution_id` is present on `run_plan`'s result and
                        # absent on the specialist `run_experiment` path, which
                        # creates no execution to cite — so the score is
                        # recorded for the runs that have an id to record it
                        # against, and skipped for the ones that do not.
                        _record_experiment_outcome(
                            store,
                            session_id,
                            succeeded=outcome[0],
                            error=outcome[1],
                            workspace=workspace,
                            execution_id=str(
                                (getattr(result, "data", None) or {}).get("execution_id") or ""
                            )
                            or None,
                        )
                except LLMDegradedError as exc:
                    # Strict mode (M14 2b) means fatal, and the generic handler
                    # below would reduce it to one lost step with the campaign
                    # carrying on — which is the silent degradation M14 exists
                    # to remove, just one level up. Stop the session and say why.
                    _fail_session_on_degraded_llm(store, session_id, record, decisions, exc)
                    _progress(f"Campaign stopped: {exc}")
                    raise
                except Exception as exc:
                    store.increment_metric(session_id, "tasks_failed")
                    if tool_step.tool in _EXPERIMENT_TOOLS:
                        _record_experiment_outcome(
                            store, session_id, succeeded=False, error=str(exc)
                        )
                    record.rationale = f"{record.rationale} | dispatch error: {exc}"
                    store.append_decision(record)
                    decisions.append(record)
                    _progress(f"Task failed: {exc}")
                    break
                store.append_decision(record)
                decisions.append(record)
                prev_id = task.id
            save_checkpoint(store, session_id)
            continue

        # Legacy M2 single-tool path
        action, observe = decide_next(
            store,
            workspace,
            session_id,
            registry,
            llm_client=llm_client,
            budgets=(budget_cfg, budget_state),
            **policy_kw,
        )
        decision_id = store.new_decision_id()
        record = DecisionRecord(
            id=decision_id,
            session_id=session_id,
            tool_name=action.tool,
            rationale=action.rationale,
            stop=action.stop,
            args=action.args,
            observe={
                "step": step,
                "completed_tools": observe.get("completed_tools"),
                "operator_feedback": observe.get("operator_feedback"),
            },
        )
        if action.stop or not action.tool:
            store.append_decision(record)
            decisions.append(record)
            store.update_session_status(session_id, "completed")
            _progress(f"Conductor stop: {action.rationale}")
            break
        # Resolve `@latest` here too. Only the multi-step campaign path did,
        # so this path could not use the shared `_default_args` and hand-rolled
        # its own — which is how `offline_next_action` came to pin
        # `plan_id="P-001"` and `dry_run=True`, and how a degraded policy
        # started minting `dry_run_stub` metrics that became evidence cards.
        action_args = resolve_step_args(
            action.tool,
            action.args,
            latest_plan_id=_latest_plan_id(workspace),
            latest_execution_id=_latest_execution_id(workspace),
            next_hypothesis_id=_next_hypothesis_id(workspace),
            baseline_plan_exists=_baseline_is_done(workspace),
        )
        record.args = action_args
        task = store.enqueue(
            session_id,
            action.tool,
            args=action_args,
            decision_id=decision_id,
        )
        record.task_id = task.id
        approval = maybe_approve(
            store,
            session_id=session_id,
            tool_name=action.tool,
            decision_id=decision_id,
            task_id=task.id,
            auto=auto_approve,
            prompt=approval_prompt,
            autonomy=autonomy,
        )
        if approval is not None:
            record.approval = approval
            if approval.decision == "reject":
                store.update_task_status(task.id, "cancelled", error="operator rejected")
                store.append_decision(record)
                decisions.append(record)
                continue
        try:
            result = scheduler.dispatch(task)
            record.artifact_refs = [r.model_dump() for r in result.refs]
        except LLMDegradedError as exc:
            _fail_session_on_degraded_llm(store, session_id, record, decisions, exc)
            _progress(f"Campaign stopped: {exc}")
            raise
        except Exception as exc:
            record.rationale = f"{record.rationale} | dispatch error: {exc}"
            store.append_decision(record)
            decisions.append(record)
            continue
        store.append_decision(record)
        decisions.append(record)
        save_checkpoint(store, session_id)

    return decisions

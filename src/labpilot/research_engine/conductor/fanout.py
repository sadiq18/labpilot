"""Run one campaign step as K parallel branches instead of one (M11).

A branch is one untested hypothesis, given its own plan, its own git worktree
to write code into, and its share of the machine's cores. The K of them run
concurrently through M5's `run_parallel_sync`, and each reports back
individually so the conductor can record a decision and feed the circuit
breaker per branch, exactly as the sequential path does for its one.

Kept out of `loop.py` because none of this is loop control: setup and teardown
here are a transaction — a claim taken must be released, a worktree created
must be removed — and that is much easier to see, and to test, when it is not
interleaved with step sequencing and stop conditions.

Design: docs/research-os/autonomy-roadmap/design/05-parallel-branches.md §6.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.agents.git_worktree import (
    ExperimentWorktree,
    create_experiment_worktree,
    remove_experiment_worktree,
)
from labpilot.research_engine.agents.parallel import ParallelWorkItem, run_parallel_sync
from labpilot.research_engine.execution.training.compute_budget import (
    available_cpus,
    cpu_share,
    reset_branch_cpu_share,
    set_branch_cpu_share,
)
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


class PlanRejected(Exception):
    """An operator declined to compile a plan for a branch.

    Its own type because `prepare_branches` has to tell it apart from a
    planning *failure*: declining is the approval gate being used as intended,
    and reporting it the same way as a broken plan compiler — at ERROR, with a
    traceback — trains operators to ignore the level that carries real faults.

    Raised by the caller's `make_plan`, since the gate and the prompt belong to
    the conductor; caught here, since this is where a branch is dropped.
    """


@dataclass(frozen=True)
class Branch:
    """One hypothesis, set up and ready to run."""

    hypothesis_id: str
    plan_id: str
    worktree: ExperimentWorktree
    workspace: Workspace


@dataclass
class BranchOutcome:
    """What one branch produced, in the shape the conductor records."""

    hypothesis_id: str
    plan_id: str
    ok: bool
    execution_id: str | None = None
    error: str | None = None
    refs: list[Any] = field(default_factory=list)


def resolve_k(requested: int, *, available: int) -> int:
    """How many branches to actually run.

    Clamped by what there is to run: asking for 4 branches with 2 untested
    hypotheses is 2 branches, not 4 with two of them idle or duplicated.

    The result is used as `max_workers` too, so the two cannot disagree —
    `cpu_share` divides the machine by this number, and a K that exceeded the
    worker pool would hand each branch a share of cores that more branches
    than can run are supposedly using. Design §6.

    Both clamps are kept even though the current caller truncates its candidate
    list to `requested` before asking, which makes `available <= requested`
    there. This is public and takes the two counts separately; enforcing only
    the clamp that happens to bind today would make it wrong for a caller that
    passes an untruncated list.

    A third clamp is the machine itself. Nothing else bounds K: `--branches`
    takes any positive integer, so `-k 64` on a ten-core box checked out 64
    worktrees and started 64 training runs, with `cpu_share` quietly handing
    each of them 1 core and no indication the request was degenerate. Capped at
    the CPUs actually available, and said out loud — a ceiling nobody can see is
    the same surprise one step later.
    """
    if requested < 2 or available < 2:
        return 1
    k = min(requested, available)
    cpus = available_cpus()
    if cpus is not None and k > cpus:
        logger.warning(
            "%d branches requested but only %d CPUs are available; running %d "
            "— more branches than cores oversubscribes the machine and gives "
            "each of them a 1-core share",
            k,
            cpus,
            cpus,
        )
        return cpus
    return k


def select_hypotheses(workspace: Workspace, limit: int) -> list[str]:
    """The top `limit` untested hypotheses, best first.

    Same ranking the sequential path's `_next_hypothesis_id` uses — it already
    sorts the whole list and takes `[0]`, so fanning out is that same list,
    less truncated. Ranking by posterior rather than the confidence written at
    creation matters more here, not less: K branches spend K times the compute
    on the answer.
    """
    from labpilot.research_engine.intelligence.hypothesis.selection import rank_hypotheses
    from labpilot.research_engine.intelligence.paths import hypotheses_are_absent
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    if limit < 1 or hypotheses_are_absent(workspace.knowledge_dir, workspace.competition):
        return []
    try:
        store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:
        logger.exception(
            "cannot read hypotheses for %s; not fanning out", workspace.competition
        )
        return []
    if not proposed:
        return []
    ranked = rank_hypotheses(proposed, workspace.knowledge_dir, workspace.competition)
    return [h.id for h in ranked[:limit]]


def _release(workspace: Workspace, hypothesis_id: str) -> None:
    """Hand a claimed hypothesis back, best-effort.

    Never raises: this runs on the failure and teardown paths, where the
    caller is already handling something else and a secondary exception would
    replace the real one.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    try:
        HypothesisStore(workspace.knowledge_dir, workspace.competition).release_claim(
            hypothesis_id
        )
    except Exception:  # noqa: BLE001 — a stuck claim is worth a log, not a crash
        logger.exception("cannot release claim on %s", hypothesis_id)


def prepare_branches(
    workspace: Workspace,
    hypothesis_ids: list[str],
    *,
    session_id: str,
    repo_root: Path,
    make_plan: Callable[[Workspace, str], str],
) -> list[Branch]:
    """Claim, plan and check out a worktree for each hypothesis that allows it.

    Returns only the branches that got all the way through. A hypothesis is
    dropped — never half-set-up — when any step fails, and whatever it had
    already taken is given back before the next one is tried, so a failure
    part-way through a fan-out cannot leave a hypothesis stuck in `testing`
    with nothing running it.

    `make_plan` is injected rather than imported: plan compilation is an LLM
    call the conductor already owns a client for, and taking it as a parameter
    is what lets this module's tests run without one.

    Plans are compiled against the shared `workspace`, before any worktree
    exists, because a plan is research state — every branch has to be able to
    see the others' — and `knowledge_dir` is shared by construction.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    branches: list[Branch] = []
    # What this iteration has taken and not yet handed to `branches`. The
    # `BaseException` handler below is the only reader; tracked here because by
    # the time it runs, the loop variable says nothing about how far the
    # current branch got.
    claimed: str | None = None
    worktree: ExperimentWorktree | None = None
    try:
        for hypothesis_id in hypothesis_ids:
            # `claim_if_proposed`, not `mark_testing_if_proposed`: only the
            # caller that actually made the claim may release it, and the older
            # method hands every racer a `testing` hypothesis and no way to
            # tell which one of them won.
            if store.claim_if_proposed(hypothesis_id) is None:
                logger.info("hypothesis %s already claimed; not branching it", hypothesis_id)
                continue
            claimed, worktree = hypothesis_id, None
            try:
                plan_id = make_plan(workspace, hypothesis_id)
            except PlanRejected as exc:
                # Before the broad handler below, and at info: an operator
                # saying no is the gate working, not a fault to investigate.
                logger.info("not branching %s: %s", hypothesis_id, exc)
                _release(workspace, hypothesis_id)
                claimed = None
                continue
            except Exception:
                logger.exception("cannot plan for %s; releasing the claim", hypothesis_id)
                _release(workspace, hypothesis_id)
                claimed = None
                continue
            try:
                worktree = create_experiment_worktree(
                    repo_root, session_id=session_id, experiment_key=hypothesis_id
                )
            except Exception:
                logger.exception("cannot check out a worktree for %s", hypothesis_id)
                _release(workspace, hypothesis_id)
                claimed = None
                continue
            # Rebound so `Branch` is not handed the handler's Optional: the
            # declaration above is `| None` only because the exception handler
            # reads it before it can be set.
            checked_out = worktree
            branches.append(
                Branch(
                    hypothesis_id=hypothesis_id,
                    plan_id=plan_id,
                    worktree=checked_out,
                    workspace=workspace.for_branch(checked_out.path),
                )
            )
            claimed, worktree = None, None
    except BaseException:
        # `BaseException`, not `Exception`: the one that matters is
        # `KeyboardInterrupt`, and `make_plan` now blocks on an operator
        # approval prompt — so Ctrl-C here is an ordinary keystroke, not a
        # crash. Uncaught, it left every hypothesis claimed so far stuck in
        # `testing`, which selection never lists again, so they were
        # unreachable by any later step or campaign. Measured before this
        # guard: three hypotheses, Ctrl-C at the second prompt, two left
        # `testing` and one worktree registered.
        if worktree is not None:
            _removed(worktree)
        if claimed is not None:
            _release(workspace, claimed)
        teardown_branches(workspace, branches, [])
        raise
    return branches


def _removed(worktree: ExperimentWorktree) -> bool:
    """Drop one worktree; report whether it went.

    The one guarded removal in this module. Both callers need the same
    tolerance — a worktree that will not go must not stop the others, and must
    not displace an exception already being handled — and the caller's response
    differs only in what it then does about the claim, which is why this returns
    a bool instead of deciding for them.
    """
    try:
        remove_experiment_worktree(worktree)
    except Exception:  # noqa: BLE001 — startup reconciliation sweeps the rest
        logger.exception("cannot remove worktree %s", worktree.path)
        return False
    return True


def teardown_branches(
    workspace: Workspace, branches: list[Branch], outcomes: list[BranchOutcome]
) -> None:
    """Remove every worktree; release the claim of every branch that did not
    succeed.

    A successful branch keeps its `testing` claim — the reflection its own run
    files is what resolves the hypothesis, and handing it back here would let
    the next step re-test something already answered.

    Derived from the *successes*, not the failures, so a branch with no outcome
    at all is released. That is the case that matters: the caller passes an
    empty `outcomes` both when it gives up before running (too few branches
    prepared) and when the fan-out raised, and reading the failures instead
    released nobody while still deleting every worktree — leaving each
    hypothesis `testing` with nothing running it and nothing able to pick it
    up, since selection only ever lists `proposed`.

    The removal is attempted after the release and guarded per branch: one
    worktree that will not go must not strand the rest, since a worktree left
    registered keeps its branch checked out and the next run of that
    experiment key cannot claim it.
    """
    succeeded = {o.hypothesis_id for o in outcomes if o.ok}
    for branch in branches:
        if not _removed(branch.worktree):
            # Claim deliberately kept: the branch is still checked out, so
            # `create_experiment_worktree` for this experiment key would hit
            # git's "already checked out". Releasing here would make the
            # hypothesis selectable again and every later step would claim it,
            # fail to check it out, and release it — burning a fan-out slot each
            # time until the next startup sweep clears the registration. Left
            # `testing`, it is skipped instead.
            logger.warning(
                "leaving %s claimed until startup reconciliation clears its worktree",
                branch.hypothesis_id,
            )
            continue
        if branch.hypothesis_id not in succeeded:
            _release(workspace, branch.hypothesis_id)


def run_branches(
    branches: list[Branch],
    *,
    agent: Any,
    cohort_id: str,
    workspace: Workspace,
    context: Any,
    build_task: Callable[[Branch, str], Any],
) -> list[BranchOutcome]:
    """Run every branch concurrently and report each one's outcome.

    The compute share is installed once around the whole fan-out rather than
    per branch: `cpu_share` divides the machine by K, so the number has to be
    the same for all of them, and a contextvar set here is what the worker
    threads each branch's training runs on inherit. The token is reset in a
    `finally` so a raising fan-out cannot leave the next, possibly sequential,
    step running under a divided machine.
    """
    if not branches:
        return []

    items = [
        ParallelWorkItem(
            id=branch.hypothesis_id,
            agent=agent,
            task=build_task(branch, cohort_id),
            # Its own workspace, so the branch writes code into its worktree
            # while data, cache and runs stay on the shared workspace.
            workspace=branch.workspace,
            context=context,
        )
        for branch in branches
    ]

    token = set_branch_cpu_share(cpu_share(len(branches)))
    try:
        results = run_parallel_sync(
            items, workspace, context, max_workers=len(branches)
        )
    finally:
        reset_branch_cpu_share(token)

    by_id = {r.id: r for r in results}
    outcomes: list[BranchOutcome] = []
    for branch in branches:
        result = by_id.get(branch.hypothesis_id)
        if result is None:  # pragma: no cover — run_parallel_sync fills every id
            outcomes.append(
                BranchOutcome(
                    hypothesis_id=branch.hypothesis_id,
                    plan_id=branch.plan_id,
                    ok=False,
                    error="missing",
                )
            )
            continue
        outcomes.append(
            BranchOutcome(
                hypothesis_id=branch.hypothesis_id,
                plan_id=branch.plan_id,
                ok=bool(result.ok),
                execution_id=_execution_id_from(result.refs),
                error=result.error,
                refs=list(result.refs),
            )
        )
    return outcomes


#: `ExperimentSpecialist` names its experiment ref `experiment:<execution_id>`
#: — `ArtifactRef` carries no metadata dict, so the id is where the execution
#: id actually travels.
_EXPERIMENT_REF_KIND = "experiment"


def _execution_id_from(refs: list[Any]) -> str | None:
    """The execution id a branch's artifact refs name, if any.

    Read off the refs because that is all a `ParallelResult` carries — the
    sequential path has the tool's own `ToolResult.data` and this does not.
    Without an id a branch's score is recorded against nothing, and the
    campaign's whole comparison quietly runs on fewer readings than it ran
    branches, so the parsing is pinned by a test rather than trusted.
    """
    for ref in refs:
        if getattr(ref, "kind", None) != _EXPERIMENT_REF_KIND:
            continue
        _, _, execution_id = str(getattr(ref, "id", "")).partition(":")
        if execution_id:
            return execution_id
    return None

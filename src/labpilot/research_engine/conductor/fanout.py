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
    cpu_share,
    reset_branch_cpu_share,
    set_branch_cpu_share,
)
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


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
    """
    if requested < 2 or available < 2:
        return 1
    return min(requested, available)


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
    for hypothesis_id in hypothesis_ids:
        # `claim_if_proposed`, not `mark_testing_if_proposed`: only the caller
        # that actually made the claim may release it, and the older method
        # hands every racer a `testing` hypothesis and no way to tell which
        # one of them won.
        if store.claim_if_proposed(hypothesis_id) is None:
            logger.info("hypothesis %s already claimed; not branching it", hypothesis_id)
            continue
        try:
            plan_id = make_plan(workspace, hypothesis_id)
        except Exception:
            logger.exception("cannot plan for %s; releasing the claim", hypothesis_id)
            _release(workspace, hypothesis_id)
            continue
        try:
            worktree = create_experiment_worktree(
                repo_root, session_id=session_id, experiment_key=hypothesis_id
            )
        except Exception:
            logger.exception("cannot check out a worktree for %s", hypothesis_id)
            _release(workspace, hypothesis_id)
            continue
        branches.append(
            Branch(
                hypothesis_id=hypothesis_id,
                plan_id=plan_id,
                worktree=worktree,
                workspace=workspace.for_branch(worktree.path),
            )
        )
    return branches


def teardown_branches(
    workspace: Workspace, branches: list[Branch], outcomes: list[BranchOutcome]
) -> None:
    """Remove every worktree; release the claims of branches that failed.

    A successful branch keeps its `testing` claim — the reflection its own run
    files is what resolves the hypothesis, and handing it back here would let
    the next step re-test something already answered. A failed branch produced
    no such answer, so its hypothesis returns to the pool.

    Each removal is guarded separately: one worktree that will not go must not
    strand the rest, since a worktree left registered keeps its branch checked
    out and the next run of that experiment key cannot claim it.
    """
    failed = {o.hypothesis_id for o in outcomes if not o.ok}
    for branch in branches:
        if branch.hypothesis_id in failed:
            _release(workspace, branch.hypothesis_id)
        try:
            remove_experiment_worktree(branch.worktree)
        except Exception:  # noqa: BLE001 — reconcile_worktrees sweeps the rest
            logger.exception(
                "cannot remove worktree %s; startup reconciliation will retry",
                branch.worktree.path,
            )


def run_branches(
    branches: list[Branch],
    *,
    agent: Any,
    cohort_id: str,
    workspace: Workspace,
    context: Any,
    dry_run: bool,
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

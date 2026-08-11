"""Thin parallel workers under a sync Conductor facade.

Runs independent specialist tasks with a worker cap and shared budget.
No research branch trees or merge policy — that stays backlog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anyio

from labpilot.research_engine.agents.ports import Agent
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.workspace_facade import Workspace

#: The only runtime this module can actually execute. Remote dispatch
#: (Kaggle, Colab, cloud) is separately tracked — TODO.md "P2 remote
#: execution". Anything else is refused rather than run here, because running
#: it here is silent: the item would finish, report a metric, and leave
#: nothing downstream able to tell the answer came from the wrong machine.
LOCAL_RUNTIME = "local"


@dataclass
class ParallelWorkItem:
    """One independent unit of specialist work."""

    id: str
    agent: Agent
    task: object
    cost: float = 1.0
    context: ContextBundle | None = None
    #: Where this item is meant to run; see `LOCAL_RUNTIME` above.
    runtime: str = LOCAL_RUNTIME


@dataclass
class ParallelResult:
    """Outcome for one work item — success or captured failure."""

    id: str
    ok: bool = True
    refs: list[ArtifactRef] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False


@dataclass
class ParallelBudget:
    """Shared cost budget across concurrent workers."""

    limit: float
    spent: float = 0.0

    def try_reserve(self, cost: float) -> bool:
        """Reserve ``cost`` if remaining budget allows (sync; call under lock)."""
        if cost < 0:
            return False
        if self.spent + cost > self.limit + 1e-9:
            return False
        self.spent += cost
        return True


async def run_parallel_async(
    items: list[ParallelWorkItem],
    workspace: Workspace,
    context: ContextBundle,
    *,
    max_workers: int = 4,
    budget_limit: float | None = None,
) -> list[ParallelResult]:
    """Run work items concurrently; a *failing* item does not cancel siblings.

    ``max_workers`` caps in-flight tasks. When ``budget_limit`` is set, items
    whose ``cost`` would exceed the remaining budget are skipped with an error.

    Raises ``ValueError`` — before any item starts, so nothing runs — for
    ``max_workers < 1`` or an item whose ``runtime`` is not `LOCAL_RUNTIME`.
    Both are the caller's mistake rather than a worker fault, which is why
    they abort the batch instead of being reported per item: a runtime this
    process cannot honour is not something the other branches' results can be
    trusted alongside.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    # Pairs rather than a set of values: naming the item is what makes the
    # error actionable when a fan-out builds a dozen of these, and it avoids
    # sorting a heterogeneous set — a `None` runtime among strings raises
    # TypeError from the sort rather than the ValueError intended here.
    offenders = [(i.id, i.runtime) for i in items if i.runtime != LOCAL_RUNTIME]
    if offenders:
        detail = ", ".join(f"{item_id}={runtime!r}" for item_id, runtime in offenders)
        raise ValueError(f"unsupported runtime(s): {detail}; only {LOCAL_RUNTIME!r} runs today")

    budget = ParallelBudget(limit=budget_limit) if budget_limit is not None else None
    limiter = anyio.CapacityLimiter(max_workers)
    lock = anyio.Lock()
    by_id: dict[str, ParallelResult] = {}

    async def _run_one(item: ParallelWorkItem) -> None:
        async with limiter:
            if budget is not None:
                async with lock:
                    reserved = budget.try_reserve(item.cost)
                if not reserved:
                    by_id[item.id] = ParallelResult(
                        id=item.id,
                        ok=False,
                        skipped=True,
                        error="budget_exceeded",
                    )
                    return
            try:
                ctx = item.context if item.context is not None else context
                refs = await item.agent.execute(item.task, workspace, ctx)
                by_id[item.id] = ParallelResult(id=item.id, ok=True, refs=list(refs))
            except Exception as exc:  # noqa: BLE001 — isolate worker faults
                by_id[item.id] = ParallelResult(
                    id=item.id,
                    ok=False,
                    error=str(exc) or exc.__class__.__name__,
                )

    async with anyio.create_task_group() as tg:
        for item in items:
            tg.start_soon(_run_one, item)

    return [by_id.get(item.id, ParallelResult(id=item.id, ok=False, error="missing")) for item in items]


def run_parallel_sync(
    items: list[ParallelWorkItem],
    workspace: Workspace,
    context: ContextBundle,
    *,
    max_workers: int = 4,
    budget_limit: float | None = None,
) -> list[ParallelResult]:
    """Sync facade — Conductor callers need no event loop.

    Raises whatever `run_parallel_async` validates, notably a ``ValueError``
    for an item whose ``runtime`` this process cannot honour.
    """

    async def _main() -> list[ParallelResult]:
        return await run_parallel_async(
            items,
            workspace,
            context,
            max_workers=max_workers,
            budget_limit=budget_limit,
        )

    return anyio.run(_main)


def parallel_summary(results: list[ParallelResult]) -> dict[str, Any]:
    """Compact counts for logs / observe hooks."""
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok and not r.skipped),
        "skipped": sum(1 for r in results if r.skipped),
        "errors": {r.id: r.error for r in results if r.error},
    }

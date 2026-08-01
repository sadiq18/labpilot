"""Tests for thin parallel specialist workers."""

from __future__ import annotations

import time
from pathlib import Path

from labpilot.research_engine.agents.models import AgentTask
from labpilot.research_engine.agents.parallel import (
    ParallelWorkItem,
    parallel_summary,
    run_parallel_sync,
)
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.workspace_facade import Workspace


def _ws(tmp_path: Path) -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", "par", code_root=tmp_path / "ws"
    ).ensure_roots()


def _bundle() -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition="par", goal="parallel"))


class _FakeAgent:
    """Test double that tracks in-flight concurrency."""

    name = "fake"
    capabilities = ["fake"]

    def __init__(self, *, fail_ids: set[str] | None = None, hold_s: float = 0.05) -> None:
        self.fail_ids = fail_ids or set()
        self.hold_s = hold_s
        self.max_in_flight = 0
        self._in_flight = 0

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        del workspace, context
        task_id = getattr(task, "id", "T")
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            # Sync sleep inside async to stress the worker pool without needing
            # a real event-loop sleep dependency for the assertion window.
            await __import__("anyio").sleep(self.hold_s)
            if task_id in self.fail_ids:
                raise RuntimeError(f"boom:{task_id}")
            return [
                ArtifactRef(
                    kind="echo",
                    id=f"echo:{task_id}",
                    schema_id="labpilot.artifact.echo/v1",
                    path=None,
                    competition="par",
                )
            ]
        finally:
            self._in_flight -= 1


def test_run_parallel_sync_caps_workers(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.08)
    items = [
        ParallelWorkItem(
            id=f"w{i}",
            agent=agent,
            task=AgentTask(id=f"T{i}", capability="fake"),
            cost=1.0,
        )
        for i in range(4)
    ]
    t0 = time.perf_counter()
    results = run_parallel_sync(items, ws, _bundle(), max_workers=2)
    elapsed = time.perf_counter() - t0
    assert len(results) == 4
    assert all(r.ok for r in results)
    assert agent.max_in_flight <= 2
    # With hold 0.08 and 4 tasks / 2 workers, expect >1 wave.
    assert elapsed >= 0.12


def test_sibling_failure_does_not_drop_others(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    agent = _FakeAgent(fail_ids={"T-bad"}, hold_s=0.02)
    items = [
        ParallelWorkItem(id="a", agent=agent, task=AgentTask(id="T-ok1", capability="fake")),
        ParallelWorkItem(id="b", agent=agent, task=AgentTask(id="T-bad", capability="fake")),
        ParallelWorkItem(id="c", agent=agent, task=AgentTask(id="T-ok2", capability="fake")),
    ]
    results = run_parallel_sync(items, ws, _bundle(), max_workers=3)
    by_id = {r.id: r for r in results}
    assert by_id["a"].ok and by_id["c"].ok
    assert not by_id["b"].ok
    assert "boom:T-bad" in (by_id["b"].error or "")
    summary = parallel_summary(results)
    assert summary["ok"] == 2
    assert summary["failed"] == 1


def test_shared_budget_skips_overflow(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.01)
    items = [
        ParallelWorkItem(
            id="cheap1",
            agent=agent,
            task=AgentTask(id="T1", capability="fake"),
            cost=2.0,
        ),
        ParallelWorkItem(
            id="cheap2",
            agent=agent,
            task=AgentTask(id="T2", capability="fake"),
            cost=2.0,
        ),
        ParallelWorkItem(
            id="expensive",
            agent=agent,
            task=AgentTask(id="T3", capability="fake"),
            cost=2.0,
        ),
    ]
    results = run_parallel_sync(items, ws, _bundle(), max_workers=3, budget_limit=4.0)
    ok = [r for r in results if r.ok]
    skipped = [r for r in results if r.skipped]
    assert len(ok) == 2
    assert len(skipped) == 1
    assert skipped[0].error == "budget_exceeded"


def test_max_workers_validation(tmp_path: Path) -> None:
    import pytest

    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="max_workers"):
        run_parallel_sync([], ws, _bundle(), max_workers=0)

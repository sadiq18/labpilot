"""Tests for thin parallel specialist workers."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from labpilot.research_engine.agents.models import AgentTask
from labpilot.research_engine.agents.parallel import (
    LOCAL_RUNTIME,
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
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="max_workers"):
        run_parallel_sync([], ws, _bundle(), max_workers=0)


def test_runtime_defaults_to_local() -> None:
    """M11 task 5: the field exists so remote dispatch has somewhere to land."""
    item = ParallelWorkItem(id="w", agent=_FakeAgent(), task=AgentTask(id="T", capability="fake"))
    assert item.runtime == LOCAL_RUNTIME


def test_an_equal_but_distinct_local_runtime_is_allowed(tmp_path: Path) -> None:
    """The guard's own boundary, exercised with a value it cannot shortcut.

    Every other passing item leans on the dataclass default, which *is* the
    module constant — so `!= LOCAL_RUNTIME` and `is not LOCAL_RUNTIME` behave
    identically and the comparison is never really tested. Building the string
    at runtime is what separates them: this is what a runtime value arriving
    from config or a JSON payload looks like, and an identity check would
    refuse it while accepting the constant. A containment guard in
    `git_worktree.py` shipped with a clause that never fired on equality for
    want of exactly this test.
    """
    from_config = "".join(["loc", "al"])
    assert from_config == LOCAL_RUNTIME
    assert from_config is not LOCAL_RUNTIME, "the test needs a non-identical equal string"

    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.0)
    items = [
        ParallelWorkItem(
            id="explicit",
            agent=agent,
            task=AgentTask(id="T0", capability="fake"),
            runtime=from_config,
        ),
    ]
    results = run_parallel_sync(items, ws, _bundle(), max_workers=1)
    assert results[0].ok
    assert [r.id for r in results[0].refs] == ["echo:T0"]


def test_a_runtime_that_cannot_run_here_is_refused(tmp_path: Path) -> None:
    """Refused, not silently run locally.

    Nothing dispatches remotely yet. Executing a Kaggle-bound item on this
    machine would succeed, report a metric, and leave no trace that the
    answer came from the wrong place — so the value is checked instead of
    ignored. The day remote dispatch lands, this test names what changes.
    """
    ws = _ws(tmp_path)
    items = [
        ParallelWorkItem(
            id="remote",
            agent=_FakeAgent(hold_s=0.0),
            task=AgentTask(id="T0", capability="fake"),
            runtime="kaggle",
        ),
    ]
    with pytest.raises(ValueError, match="unsupported runtime"):
        run_parallel_sync(items, ws, _bundle(), max_workers=1)


def test_the_refusal_happens_before_any_sibling_runs(tmp_path: Path) -> None:
    """A pre-flight check, not a per-item failure.

    Budget and cost are spent by running work; discovering the bad item after
    three siblings have already trained would waste exactly what the check is
    cheap enough to prevent.
    """
    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.0)
    items = [
        ParallelWorkItem(id="ok", agent=agent, task=AgentTask(id="T0", capability="fake")),
        ParallelWorkItem(
            id="bad",
            agent=agent,
            task=AgentTask(id="T1", capability="fake"),
            runtime="colab",
        ),
    ]
    with pytest.raises(ValueError, match="colab"):
        run_parallel_sync(items, ws, _bundle(), max_workers=2)
    assert agent.max_in_flight == 0, "a sibling ran before the runtime was validated"


def test_the_refusal_names_the_item_not_just_the_runtime(tmp_path: Path) -> None:
    """A dozen-item fan-out needs to say which branch was misconfigured."""
    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.0)
    items = [
        ParallelWorkItem(id=f"ok{i}", agent=agent, task=AgentTask(id=f"T{i}", capability="fake"))
        for i in range(3)
    ]
    items.append(
        ParallelWorkItem(
            id="branch-7",
            agent=agent,
            task=AgentTask(id="T7", capability="fake"),
            runtime="colab",
        )
    )
    with pytest.raises(ValueError, match="branch-7"):
        run_parallel_sync(items, ws, _bundle(), max_workers=2)


def test_a_none_runtime_is_refused_not_a_type_error(tmp_path: Path) -> None:
    """Dataclasses do not enforce annotations, so `None` is constructible.

    Collecting the offenders as (id, runtime) pairs rather than sorting a set
    of the values is what keeps this a ValueError: sorting `{None, "kaggle"}`
    raises TypeError from the comparison, burying the real problem under an
    error about `<` that the caller never wrote.
    """
    ws = _ws(tmp_path)
    agent = _FakeAgent(hold_s=0.0)
    items = [
        ParallelWorkItem(
            id="none-runtime",
            agent=agent,
            task=AgentTask(id="T0", capability="fake"),
            runtime=None,  # type: ignore[arg-type]
        ),
        ParallelWorkItem(
            id="str-runtime",
            agent=agent,
            task=AgentTask(id="T1", capability="fake"),
            runtime="kaggle",
        ),
    ]
    with pytest.raises(ValueError, match="unsupported runtime"):
        run_parallel_sync(items, ws, _bundle(), max_workers=2)

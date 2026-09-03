"""`gather_once` — the evidence routine as a callable unit (M16).

The gate `should_gather_evidence` has shipped since M21; what did not exist was
a way to call it, and the pipeline behind it, without going through the policy
step this milestone exists to bypass. These tests pin the unit's two jobs —
honour the gate, report what happened — and the boundary that keeps it usable
outside Kaggle.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from labpilot.research_engine.conductor.producer import (
    GatherPlan,
    default_gather_plan,
    gather_once,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _registry(calls: list[dict], *, name: str = "gather_stub", created: int = 0) -> ToolRegistry:
    class _Report:
        summary = {"hypothesis_count": created}

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        calls.append(dict(kwargs))
        return ToolResult(refs=[], data={"report": _Report()})

    registry = ToolRegistry()
    registry.register(ToolDescriptor(name=name, handler=handler, capability_status="fixed"))
    return registry


def _quiet_workspace(tmp_path: Path, *, hypotheses: int = 6, evidence_age_hours: float = 1.0):
    """A workspace the gate has no reason to gather in.

    Both open clauses have to be closed at once: enough viable hypotheses that
    the pool is not thin, *and* an artifact recent enough that the evidence is
    not stale — the clauses are independent, so satisfying one leaves the other
    firing. `evidence_age_hours` sits above `_MIN_RESWEEP_HOURS` (0.5) and
    below the 24h cooldown.
    """
    ws = _ws(tmp_path)
    store = HypothesisStore(ws.knowledge_dir, ws.competition)
    for i in range(hypotheses):
        store.create(
            observation=f"o{i}",
            reason=f"r{i}",
            prediction=f"p{i}",
            confidence=0.5,
            technique=f"tech-{i}",
        )
    stamp = (datetime.now(UTC) - timedelta(hours=evidence_age_hours)).isoformat()
    with KnowledgeStore(ws.knowledge_dir, ws.competition) as knowledge:
        knowledge.upsert_artifact(
            ResearchArtifact(
                id="art:test:1",
                type=ResearchArtifactType.NOTE,
                source="user",
                title="gathered evidence",
                competition_slug=ws.competition,
            )
        )
        knowledge._conn.execute(  # noqa: SLF001 — dating the row is the point
            "UPDATE research_artifacts SET created_at = ?", (stamp,)
        )
        knowledge._conn.commit()  # noqa: SLF001
    return ws


def _stale_workspace(tmp_path: Path):
    """A **thin** pool with evidence old enough to gather on anyway.

    The `stale` clause is what the producer's own work *does* move — 158h to
    0.01h in the measured run — so it has to keep firing while `thin` is
    latched.

    Thin on purpose, and this is the whole point of the fixture. It read
    `hypotheses=40`, which put the pool over target, so `thin` could not fire
    and the suppression under test was never exercised: the gate returns at the
    first clause that fires, and with a full pool it simply reached `stale` on
    its own. The test passed against a producer that skipped every tick while
    latched. One hypothesis makes `thin` fire *and* be suppressed, which is the
    only arrangement that can tell the two behaviours apart.
    """
    return _quiet_workspace(tmp_path, hypotheses=1, evidence_age_hours=48.0)


def _stagnant_budgets():
    """A campaign whose last two experiments beat nothing."""
    from labpilot.research_engine.conductor.budgets import BudgetConfig, BudgetState, ScoreEvent

    events = [
        ScoreEvent(experiment_id=f"E-{i}", metric_name="rmse", value=value, maximize=False)
        for i, value in enumerate((1.0, 2.0, 3.0))
    ]
    return BudgetConfig(plateau_window=1), BudgetState(score_events=events)


def test_an_empty_pool_gathers_and_reports_the_gate_reason(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    calls: list[dict] = []

    outcome = gather_once(ws, _registry(calls, created=3), GatherPlan(tool="gather_stub"))

    assert outcome.gathered is True
    assert "viable" in outcome.reason
    assert outcome.hypotheses_created == 3
    assert len(calls) == 1


def test_a_full_pool_skips_without_invoking_the_tool(tmp_path: Path) -> None:
    """Exit criterion 2: the tick no-ops, and says why."""
    ws = _quiet_workspace(tmp_path)
    calls: list[dict] = []

    outcome = gather_once(ws, _registry(calls), GatherPlan(tool="gather_stub"))

    assert outcome.gathered is False
    assert outcome.reason
    assert calls == []
    assert outcome.hypotheses_created == 0


def test_the_plan_decides_what_is_gathered_not_the_producer(tmp_path: Path) -> None:
    """§5.4: no source name is readable from `producer.py`.

    Driving it with a stub tool and arbitrary args must work exactly as well as
    the Kaggle plan does. This fails the moment someone reaches for
    `ANALYZE_ARGS` inside the unit.
    """
    ws = _ws(tmp_path)
    calls: list[dict] = []
    plan = GatherPlan(tool="gather_stub", args={"sources": ["lab-notebook"], "limit": 2})

    outcome = gather_once(ws, _registry(calls), plan)

    assert outcome.gathered is True
    assert calls == [{"sources": ["lab-notebook"], "limit": 2}]


def test_the_llm_client_reaches_a_handler_that_asks_for_one(tmp_path: Path) -> None:
    """A gathering pipeline running without a client degrades silently."""
    ws = _ws(tmp_path)
    seen: list[object] = []

    def handler(workspace: Workspace, llm_client: object = None, **kwargs: object) -> ToolResult:
        seen.append(llm_client)
        return ToolResult(refs=[], data={})

    registry = ToolRegistry()
    registry.register(
        # A stub returns the same thing whatever it is asked, and M15 makes
        # a descriptor say so rather than let the catalog overstate it.
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    sentinel = object()

    gather_once(ws, registry, GatherPlan(tool="gather_stub"), llm_client=sentinel)

    assert seen == [sentinel]


def test_a_handler_that_takes_no_client_is_not_handed_one(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    calls: list[dict] = []

    gather_once(ws, _registry(calls), GatherPlan(tool="gather_stub"), llm_client=object())

    assert calls == [{}]


def test_a_failing_tool_raises_rather_than_reporting_success(tmp_path: Path) -> None:
    """Isolating a bad tick belongs to the runner; a unit that swallows its own
    failures cannot be tested for them.
    """
    ws = _ws(tmp_path)

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        raise RuntimeError("network died")

    registry = ToolRegistry()
    registry.register(
        # A stub returns the same thing whatever it is asked, and M15 makes
        # a descriptor say so rather than let the catalog overstate it.
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )

    with pytest.raises(RuntimeError, match="network died"):
        gather_once(ws, registry, GatherPlan(tool="gather_stub"))


def test_the_default_plan_is_the_campaigns_own_gathering_budget(tmp_path: Path) -> None:
    """The one quarantined domain site — pinned so a move is deliberate."""
    from labpilot.research_engine.conductor.actions import ANALYZE_ARGS

    plan = default_gather_plan(_ws(tmp_path))

    assert plan.tool == "analyze_competition"
    assert plan.args == ANALYZE_ARGS
    assert plan.args is not ANALYZE_ARGS  # a copy: a tick must not mutate the constant


# --- the runner, the allowlist, and what the policy sees ----------------------


def test_the_consumer_loses_the_tool_while_a_producer_owns_it(tmp_path: Path) -> None:
    """Exit criterion 1, as a property of the allowlist rather than of timing.

    The gate here says *gather* — an empty pool, no artifacts — which is
    exactly the case where leaving the tool gated on the predicate would have
    both components sweep in the same second.
    """
    from labpilot.research_engine.conductor.policy import (
        available_tools,
        should_gather_evidence,
    )

    ws = _ws(tmp_path)
    catalog = {"analyze_competition", "generate_plan", "query_memory"}

    assert should_gather_evidence(ws)[0] is True
    assert "analyze_competition" in available_tools(ws, catalog)
    assert "analyze_competition" not in available_tools(ws, catalog, external_gathering=True)


def test_a_tick_that_raises_is_recorded_and_survived(tmp_path: Path) -> None:
    """Exit criterion 6: the consumer's work is the thing with a deadline."""
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    ws = _ws(tmp_path)

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        raise RuntimeError("network died")

    registry = ToolRegistry()
    registry.register(
        # A stub returns the same thing whatever it is asked, and M15 makes
        # a descriptor say so rather than let the catalog overstate it.
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    producer = EvidenceProducer(ws, registry, plan=GatherPlan(tool="gather_stub"))

    assert producer.tick_once() is None

    status = producer.status()
    assert status["ticks"] == 1
    assert "network died" in status["last_error"]


def test_status_reports_the_gate_reason_a_skipped_tick_gave(tmp_path: Path) -> None:
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    ws = _quiet_workspace(tmp_path)
    calls: list[dict] = []
    producer = EvidenceProducer(ws, _registry(calls), plan=GatherPlan(tool="gather_stub"))

    producer.tick_once()
    status = producer.status()

    assert status["last_decision"] == "skipped"
    assert status["last_reason"]
    assert status["last_error"] is None
    assert calls == []


def test_the_thread_ticks_and_stops_without_waiting_out_its_interval(tmp_path: Path) -> None:
    """`wait` on an event, not `sleep`: an idle producer must not hold shutdown."""
    import time as _time

    from labpilot.research_engine.conductor.producer import EvidenceProducer

    ws = _quiet_workspace(tmp_path)
    calls: list[dict] = []
    producer = EvidenceProducer(
        ws, _registry(calls), plan=GatherPlan(tool="gather_stub"), tick_seconds=30.0
    )

    producer.start()
    deadline = _time.monotonic() + 5.0
    while producer.status()["ticks"] < 1 and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert producer.status()["ticks"] >= 1

    started = _time.monotonic()
    producer.stop(timeout=5.0)
    assert _time.monotonic() - started < 2.0  # not the 30s interval
    assert producer.is_running() is False


def test_the_observe_bundle_carries_producer_state_when_given_one(tmp_path: Path) -> None:
    """§9: the policy should not watch the pool grow with no account of why."""
    from labpilot.research_engine.conductor.policy import build_observe_bundle
    from labpilot.research_engine.conductor.store import ConductorStore

    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("beat baseline")
        without = build_observe_bundle(store, ws, session.id, include_context=False)
        with_status = build_observe_bundle(
            store,
            ws,
            session.id,
            include_context=False,
            producer_status={"running": True, "last_decision": "skipped"},
        )
    finally:
        store.close()

    assert "evidence_producer" not in without
    assert with_status["evidence_producer"]["last_decision"] == "skipped"


# --- the producer is the lower claim on the LLM budget ------------------------


class _Router:
    """Enough of the gateway to answer `preview(role, reserve=…)`."""

    def __init__(self, *, available_at_reserve: float) -> None:
        self.available_at_reserve = available_at_reserve
        self.asked: list[tuple[str, float]] = []

    def preview(self, role: str, *, reserve: float = 0.0):
        self.asked.append((role, reserve))
        provider = object() if reserve <= self.available_at_reserve else None
        return type("Route", (), {"provider": provider, "reason": "daily limit reached"})()


def test_the_producer_yields_when_the_reserve_leaves_it_no_room(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    calls: list[dict] = []
    router = _Router(available_at_reserve=0.0)  # room at 0%, none at 20%

    outcome = gather_once(
        ws,
        _registry(calls),
        GatherPlan(tool="gather_stub"),
        llm_client=router,
        reserve=0.2,
    )

    assert outcome.gathered is False
    assert "holding 20%" in outcome.reason
    assert calls == []
    assert router.asked == [("reasoning", 0.2)]


def test_the_same_budget_still_lets_the_consumer_through() -> None:
    """The point of a reserve: the producer runs out first, not both."""
    router = _Router(available_at_reserve=0.0)

    assert router.preview("reasoning", reserve=0.0).provider is not None
    assert router.preview("reasoning", reserve=0.2).provider is None


def test_no_reserve_means_no_pre_flight_question(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    calls: list[dict] = []
    router = _Router(available_at_reserve=0.0)

    outcome = gather_once(ws, _registry(calls), GatherPlan(tool="gather_stub"), llm_client=router)

    assert outcome.gathered is True
    assert router.asked == []


def test_a_client_with_no_router_behind_it_does_not_block_gathering(tmp_path: Path) -> None:
    """A legacy provider path has no ledger to ask; refusing on that basis
    would be inventing a limit rather than respecting one.
    """
    ws = _ws(tmp_path)
    calls: list[dict] = []

    class _PlainClient:
        def complete(self, system: str, user: str, **kw: object) -> str:
            return ""

    outcome = gather_once(
        ws,
        _registry(calls),
        GatherPlan(tool="gather_stub"),
        llm_client=_PlainClient(),
        reserve=0.9,
    )

    assert outcome.gathered is True
    assert len(calls) == 1


# --- lifecycle and instrumentation the review caught -------------------------


def test_the_producer_thread_inherits_the_provenance_context(tmp_path: Path) -> None:
    """A fresh thread gets a fresh empty context, so the provenance sink the
    campaign installed is invisible inside it and every micro-agent call the
    sweep makes records nothing. Fourth instance of this bug in this codebase
    — `SqliteInvocationSink` names the first three.
    """
    from labpilot.accessor.common import provenance
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    ws = _ws(tmp_path)  # empty pool, so the gate opens and the tool actually runs
    seen: list[object] = []

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        seen.append(provenance._sink.get())  # noqa: SLF001 — the point of the test
        return ToolResult(refs=[], data={})

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    producer = EvidenceProducer(ws, registry, plan=GatherPlan(tool="gather_stub"), tick_seconds=1.0)

    sentinel = object()
    token = provenance.set_sink(sentinel)  # type: ignore[arg-type]
    try:
        producer.start()
        deadline = __import__("time").monotonic() + 5.0
        while producer.status()["ticks"] < 1 and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.01)
        producer.stop(timeout=5.0)
    finally:
        provenance.reset_sink(token)

    # The gate is open on a fresh workspace, so the tool ran and recorded what
    # the sink looked like from inside the producer thread.
    assert seen == [sentinel]


def test_a_second_start_is_refused_while_the_first_thread_lives(tmp_path: Path) -> None:
    """`start()` clears the stop event, so stacking one on a live thread would
    un-stop the loop that was told to finish and leave two sweeping at once.
    """
    import threading as _threading

    from labpilot.research_engine.conductor.producer import EvidenceProducer

    entered = _threading.Event()
    release = _threading.Event()

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        entered.set()
        release.wait(5.0)
        return ToolResult(refs=[], data={})

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    # An empty pool, so the gate opens and the tick blocks inside the handler.
    ws_open = _ws(tmp_path, slug="open")
    producer = EvidenceProducer(
        ws_open, registry, plan=GatherPlan(tool="gather_stub"), tick_seconds=1.0
    )

    producer.start()
    assert entered.wait(5.0)
    first = producer._thread  # noqa: SLF001 — identity is the assertion

    producer.start()
    assert producer._thread is first  # noqa: SLF001

    release.set()
    producer.stop(timeout=5.0)


def test_a_thread_that_outlives_the_timeout_is_kept_not_dropped(tmp_path: Path) -> None:
    """Dropping the handle reported `is_running() is False` while a sweep was
    still writing, and let a later `start()` resurrect the old loop.
    """
    import threading as _threading

    from labpilot.research_engine.conductor.producer import EvidenceProducer

    entered = _threading.Event()
    release = _threading.Event()

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        entered.set()
        release.wait(10.0)
        return ToolResult(refs=[], data={})

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    producer = EvidenceProducer(
        _ws(tmp_path), registry, plan=GatherPlan(tool="gather_stub"), tick_seconds=1.0
    )

    producer.start()
    assert entered.wait(5.0)

    assert producer.stop(timeout=0.1) is False
    assert producer.is_running() is True  # still sweeping, and still says so

    release.set()
    assert producer.stop(timeout=5.0) is True
    assert producer.is_running() is False


def test_a_tick_interval_below_the_floor_is_raised_to_it(tmp_path: Path) -> None:
    """`Event.wait(0)` returns immediately: a zero interval is a spin that
    globs the hypothesis directory as fast as it can, hidden by the re-sweep
    floor making every tick a no-op.
    """
    from labpilot.research_engine.conductor.producer import (
        _MIN_TICK_SECONDS,
        EvidenceProducer,
    )

    ws = _quiet_workspace(tmp_path)
    calls: list[dict] = []

    assert EvidenceProducer(ws, _registry(calls), tick_seconds=0).tick_seconds == _MIN_TICK_SECONDS
    assert EvidenceProducer(ws, _registry(calls), tick_seconds=-5).tick_seconds == _MIN_TICK_SECONDS
    assert EvidenceProducer(ws, _registry(calls), tick_seconds=30).tick_seconds == 30


def test_a_failed_tick_does_not_leave_the_previous_decision_showing(
    tmp_path: Path, monkeypatch
) -> None:
    """`status()` reported `last_decision="gathered"` beside a fresh
    `last_error`, describing a sweep that did not happen on the tick reported.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    # Unreadable viable count => no hysteresis verdict, so the second tick still
    # reaches the tool. The latch has its own tests below.
    monkeypatch.setattr(producer_mod, "_viable", lambda _ws: None)

    ws = _ws(tmp_path)
    outcomes = [ToolResult(refs=[], data={}), RuntimeError("network died")]

    def handler(workspace: Workspace, **kwargs: object) -> ToolResult:
        nxt = outcomes.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(name="gather_stub", handler=handler, capability_status="fixed")
    )
    producer = EvidenceProducer(ws, registry, plan=GatherPlan(tool="gather_stub"))

    producer.tick_once()
    assert producer.status()["last_decision"] == "gathered"

    producer.tick_once()
    status = producer.status()
    assert status["last_decision"] is None
    assert "network died" in status["last_error"]


def test_the_default_plan_does_not_share_its_nested_lists(tmp_path: Path) -> None:
    """A shallow copy left `args["exclude"]` pointing at the module constant,
    so one handler mutating it in place edits every later tick and step.
    """
    from labpilot.research_engine.conductor.actions import ANALYZE_ARGS

    plan = default_gather_plan(_ws(tmp_path))
    plan.args["exclude"].append("dataset")

    assert ANALYZE_ARGS["exclude"] == ["papers"]


def test_a_client_whose_preview_keeps_raising_is_only_reported_once(tmp_path: Path) -> None:
    """A signature mismatch does not heal; a traceback every tick for twelve
    hours buries the campaign's own log.
    """
    ws = _ws(tmp_path)
    calls: list[dict] = []
    attempts: list[int] = []

    class _Broken:
        def preview(self, role: str, **kw: object):
            attempts.append(1)
            raise TypeError("preview() got an unexpected keyword argument 'reserve'")

    client = _Broken()
    for _ in range(3):
        outcome = gather_once(
            ws, _registry(calls), GatherPlan(tool="gather_stub"), llm_client=client, reserve=0.2
        )
        assert outcome.gathered is True

    assert len(attempts) == 1


def test_the_env_switch_only_accepts_boolean_words(monkeypatch) -> None:
    """A denylist turned every value it did not recognise into *on*, so an
    operator writing `disabled` switched the feature on and moved gathering to
    a different component without being told.
    """
    from labpilot.cli.conduct import _gather_background_enabled

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("LABPILOT_GATHER_BACKGROUND", value)
        assert _gather_background_enabled(False) is True, value

    for value in ("", "0", "false", "no", "off", "disabled", "none", "n", "2"):
        monkeypatch.setenv("LABPILOT_GATHER_BACKGROUND", value)
        assert _gather_background_enabled(False) is False, value

    monkeypatch.setenv("LABPILOT_GATHER_BACKGROUND", "off")
    assert _gather_background_enabled(True) is True  # the flag still wins


def _campaign_registry(order: list[str], gathered: threading.Event) -> ToolRegistry:
    """Two handlers: the producer's sweep, and everything the campaign can call.

    The campaign's tools block until the producer has swept, so the ordering
    assertions below do not depend on how the scheduler happens to interleave
    two threads. `analyze_competition` is only ever reached by the producer —
    while one is running it is off the consumer's allowlist by design.
    """

    def sweep(workspace: Workspace, **kwargs: object) -> ToolResult:
        order.append("gather")
        gathered.set()
        return ToolResult(refs=[], data={})

    def campaign_step(workspace: Workspace, **kwargs: object) -> ToolResult:
        gathered.wait(10.0)
        return ToolResult(refs=[], data={})

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(name="analyze_competition", handler=sweep, capability_status="fixed")
    )
    for name in ("query_memory", "generate_plan", "search_papers"):
        registry.register(
            ToolDescriptor(name=name, handler=campaign_step, capability_status="fixed")
        )
    return registry


def test_the_campaign_starts_the_producer_only_after_repairing_memory(
    tmp_path: Path, monkeypatch
) -> None:
    """Its first tick fires immediately and mints from beliefs and overlays —
    the things the repair chain at the top of the loop has just corrected.
    Started before it, the first sweep reads the pre-repair compass: a campaign
    once ran with 45 false `vit` claims intact, and every rogii overlay said
    `Avoid: SWA` about the only technique that had ever improved the metric.

    Driven through the real loop rather than by reading its source, so a
    refactor that keeps the call and moves it cannot pass.
    """
    import time as _time

    from labpilot.research_engine.conductor.loop import run_until_stop
    from labpilot.research_engine.conductor.store import ConductorStore
    from labpilot.research_engine.execution import outcome as outcome_mod

    order: list[str] = []
    gathered = threading.Event()

    def slow_repair(**kwargs: object) -> list[str]:
        order.append("repair")
        # Long enough that a producer started ahead of the chain would have
        # swept before this returns.
        _time.sleep(0.5)
        return []

    monkeypatch.setattr(outcome_mod, "revalidate_outcome_claims", slow_repair)

    ws = _ws(tmp_path, slug="ordering")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("improve score", metadata={"max_steps": 2})
        run_until_stop(
            store,
            ws,
            session.id,
            _campaign_registry(order, gathered),
            llm_client=None,
            max_steps=2,
            auto_approve=True,
            prefer_offline=True,
            gather_background=True,
        )
    finally:
        store.close()

    assert "repair" in order, "the repair chain did not run"
    assert "gather" in order, "the producer never swept"
    assert order.index("repair") < order.index("gather")


def test_gathering_returns_to_the_campaign_if_the_producer_is_not_running(
    tmp_path: Path, monkeypatch
) -> None:
    """The allowlist handover has to follow the thread, not the object.

    Keyed on existence, a producer whose thread never started would take
    `analyze_competition` off the consumer's allowlist for the whole run and
    leave nothing gathering in its place. Both directions are driven here: a
    live producer owns the sweep, a dead one hands it back.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.loop import run_until_stop
    from labpilot.research_engine.conductor.store import ConductorStore

    def _run(slug: str, *, start_the_thread: bool) -> list[str]:
        order: list[str] = []
        gathered = threading.Event()
        if not start_the_thread:
            # Nothing sweeps, so the campaign's tools must not wait for one.
            gathered.set()
            monkeypatch.setattr(
                producer_mod.EvidenceProducer, "start", lambda self: None, raising=True
            )
        ws = _ws(tmp_path, slug=slug)
        store = ConductorStore(ws.knowledge_dir, ws.competition)
        try:
            session = store.create_session("improve score", metadata={"max_steps": 3})
            decisions = run_until_stop(
                store,
                ws,
                session.id,
                _campaign_registry(order, gathered),
                llm_client=None,
                max_steps=3,
                auto_approve=True,
                prefer_offline=True,
                gather_background=True,
            )
        finally:
            store.close()
        return [d.tool_name for d in decisions if d.tool_name and not d.stop]

    with monkeypatch.context():
        live = _run("live-producer", start_the_thread=True)
    dead = _run("dead-producer", start_the_thread=False)

    assert "analyze_competition" not in live, "a running producer must own the sweep"
    assert "analyze_competition" in dead, "a dead producer must hand gathering back"


def test_a_producer_with_a_session_can_actually_read_its_budgets(tmp_path: Path) -> None:
    """Every tick failed, silently, whenever a session id was set.

    `_budgets` imported `load_budget_pair` from `budgets` — it lives in
    `checkpoint` — and the imports sat outside the guard, so the ImportError
    went past `_budgets`' own handler to `tick_once`, which counted the whole
    tick as failed. The producer swept nothing for the life of the campaign
    while `last_error` explained why to a field nobody reads. Every unit test
    built a producer without a session id and never reached the line.
    """
    from labpilot.research_engine.conductor.budgets import BudgetConfig, BudgetState
    from labpilot.research_engine.conductor.producer import EvidenceProducer
    from labpilot.research_engine.conductor.store import ConductorStore

    ws = _ws(tmp_path)
    calls: list[dict] = []
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("improve score")
    finally:
        store.close()

    producer = EvidenceProducer(
        ws,
        _registry(calls),
        session_id=session.id,
        plan=GatherPlan(tool="gather_stub"),
    )

    pair = producer._budgets()  # noqa: SLF001 — the line that was unreachable
    assert pair is not None
    assert isinstance(pair[0], BudgetConfig)
    assert isinstance(pair[1], BudgetState)

    outcome = producer.tick_once()
    assert outcome is not None, producer.status()["last_error"]
    assert producer.status()["last_error"] is None
    assert len(calls) == 1


# --- hysteresis: the producer must not chase a signal it cannot move ---------


def _counting_viable(values: list[int]):
    """Stand-in for `_viable` that walks a scripted sequence."""

    def read(_workspace) -> int:
        return values.pop(0) if len(values) > 1 else values[0]

    return read


def test_a_sweep_that_does_not_move_the_viable_count_stops_the_producer_sweeping(
    tmp_path: Path, monkeypatch
) -> None:
    """Measured on rogii 2026-08-20: two complete sweeps, ten hypotheses each,
    and the gate still read "only 10 viable hypotheses queued" before the third.
    `viable_hypothesis_count` ages a row out after two selections that picked
    something else, so with 153 proposals the producer's own output aged out as
    fast as it arrived — ~40 minutes of LLM work in a 51-minute campaign.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    monkeypatch.setattr(producer_mod, "_viable", lambda _ws: 3)  # never moves

    calls: list[dict] = []
    # A workspace with recent, non-stale evidence: `_ws` has no artifact at
    # all, so the gate answers `first` — a real second reason to sweep — and
    # the second tick would gather on it rather than on the latched clause.
    # The case under test is a pool that stays thin, not one that has never
    # gathered.
    producer = EvidenceProducer(
        _quiet_workspace(tmp_path, hypotheses=1, evidence_age_hours=1.0),
        _registry(calls, created=10),
        plan=GatherPlan(tool="gather_stub"),
    )

    first = producer.tick_once()
    assert first is not None and first.gathered is True
    assert first.clause == "thin"
    assert producer.status()["thin_clause_unmovable"] is True

    second = producer.tick_once()
    assert second is not None
    assert second.gathered is False
    assert "did not change that" in second.reason
    assert len(calls) == 1, "the second tick must not sweep again"


def test_the_latch_lifts_when_the_count_actually_moves(tmp_path: Path, monkeypatch) -> None:
    """A pool that genuinely drains is the case the clause exists for, and it
    has to keep working — the latch is hysteresis, not an off switch.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    # Reads, in order: tick 1 before=3, after=3 (latches at 3); tick 2's
    # skip-check sees 5 (> 3, so the latch lifts), then before=5, after=9 —
    # a sweep that did move the count, so nothing re-latches.
    monkeypatch.setattr(producer_mod, "_viable", _counting_viable([3, 3, 5, 5, 9]))

    calls: list[dict] = []
    producer = EvidenceProducer(
        _ws(tmp_path), _registry(calls, created=10), plan=GatherPlan(tool="gather_stub")
    )

    producer.tick_once()
    assert producer.status()["thin_clause_unmovable"] is True

    second = producer.tick_once()
    assert producer.status()["thin_clause_unmovable"] is False
    assert second is not None and second.gathered is True
    assert len(calls) == 2


def test_a_sweep_that_does_move_the_count_never_latches(tmp_path: Path, monkeypatch) -> None:
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    monkeypatch.setattr(producer_mod, "_viable", _counting_viable([1, 4]))

    calls: list[dict] = []
    producer = EvidenceProducer(
        _ws(tmp_path), _registry(calls, created=10), plan=GatherPlan(tool="gather_stub")
    )

    outcome = producer.tick_once()
    assert outcome is not None and outcome.viable_before == 1
    assert outcome.viable_after == 4
    assert producer.status()["thin_clause_unmovable"] is False


def test_the_latch_only_silences_the_clause_it_was_earned_on(tmp_path: Path, monkeypatch) -> None:
    """Staleness and stagnation are signals the producer's own work *does*
    move (158h → 0.01h in the measured run), so they must keep firing.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import gather_once

    monkeypatch.setattr(producer_mod, "_viable", lambda _ws: 3)
    calls: list[dict] = []

    # `thin` is latched, but this workspace's gate opens on staleness instead.
    outcome = gather_once(
        _stale_workspace(tmp_path),
        _registry(calls),
        GatherPlan(tool="gather_stub"),
        skip_clauses=("thin",),
    )

    assert outcome.gathered is True
    # `stale` exactly. `first` was in here as an alternative, and it hid the
    # difference: on a workspace with no artifact at all the gate answers
    # `first` whether or not suppression works.
    assert outcome.clause == "stale"
    assert len(calls) == 1


def test_a_latched_clause_falls_through_to_stagnation(tmp_path: Path, monkeypatch) -> None:
    """The other clause below `thin`, and the one a thin pool masks longest.

    `gather_verdict` returns at the first clause that fires and `thin` is
    evaluated before `stagnant`, so a caller that filtered the *returned*
    clause suppressed stagnation as well without ever evaluating it.
    """
    from labpilot.research_engine.conductor import producer as producer_mod
    from labpilot.research_engine.conductor.producer import gather_once

    monkeypatch.setattr(producer_mod, "_viable", lambda _ws: 3)
    calls: list[dict] = []

    outcome = gather_once(
        _quiet_workspace(tmp_path, hypotheses=1, evidence_age_hours=1.0),
        _registry(calls),
        GatherPlan(tool="gather_stub"),
        budgets=_stagnant_budgets(),
        skip_clauses=("thin",),
    )

    assert outcome.gathered is True
    assert outcome.clause == "stagnant"
    assert len(calls) == 1


def test_a_latched_clause_with_nothing_below_it_is_held_not_satisfied(tmp_path: Path) -> None:
    """When suppression really is the last word, say which signal is stuck.

    `satisfied` would report "5 viable hypotheses queued" for a campaign
    holding one, into the log an operator reads to work out why nothing is
    gathering.
    """
    from labpilot.research_engine.conductor.producer import gather_once

    calls: list[dict] = []
    outcome = gather_once(
        _quiet_workspace(tmp_path, hypotheses=1, evidence_age_hours=1.0),
        _registry(calls),
        GatherPlan(tool="gather_stub"),
        skip_clauses=("thin",),
    )

    assert outcome.gathered is False
    assert outcome.clause == "held"
    assert "1 viable" in outcome.reason and "did not change that" in outcome.reason
    assert not calls


# --- the two guards that fail quietly ----------------------------------------


def test_a_malformed_interval_warns_instead_of_killing_the_campaign(monkeypatch) -> None:
    """`producer.py` is imported lazily from inside `run_until_stop`, so a
    `float()` raising at module scope took the campaign down several steps in,
    with a traceback naming neither the variable nor the value.
    """
    from labpilot.research_engine.conductor.producer import _env_float

    monkeypatch.setenv("LABPILOT_GATHER_TICK_S", "5m")
    assert _env_float("LABPILOT_GATHER_TICK_S", 300.0) == 300.0

    monkeypatch.setenv("LABPILOT_GATHER_TICK_S", "  60  ")
    assert _env_float("LABPILOT_GATHER_TICK_S", 300.0) == 60.0

    monkeypatch.delenv("LABPILOT_GATHER_TICK_S")
    assert _env_float("LABPILOT_GATHER_TICK_S", 300.0) == 300.0


class _PreviewClient:
    """Minimal stand-in for the routed LLM client `_budget_headroom` asks."""

    def __init__(self, *, raises: bool) -> None:
        self.raises = raises
        self.previews = 0

    def preview(self, _role, *, reserve):
        self.previews += 1
        if self.raises:
            raise RuntimeError("no such parameter: reserve")
        return type("R", (), {"provider": None, "reason": "daily limit reached"})()


def test_a_refusal_does_not_outlive_the_client_it_describes() -> None:
    """The refusal set held `id(client)` — an address, and nothing more.

    So the entry survived the client and the set grew for the life of the
    process; and because CPython hands a freed address to the next object of
    that size, a fresh client could inherit the refusal. `_budget_headroom`
    would then answer "go ahead" without ever calling `preview`, leaving the
    reserve off in the campaign it exists to protect, with no log line.

    The address-reuse half cannot be forced without depending on the allocator,
    so what is pinned here is the invariant underneath it: an entry that cannot
    outlive its client cannot be inherited by another.

    Recorded through `_remember_refusal` rather than by provoking a real
    failure, deliberately. `_budget_headroom` logs the exception, the traceback
    holds the frame that holds the client, and pytest keeps log records for the
    length of the test — so a client refused the realistic way stays reachable
    and the collection below would measure the harness instead of the code.
    """
    import gc

    from labpilot.research_engine.conductor.producer import _PREVIEW_REFUSED, _remember_refusal

    tracked = _PreviewClient(raises=False)
    _remember_refusal(tracked)

    # Fails outright against a `set[int]` of ids: a client is not its address.
    assert tracked in _PREVIEW_REFUSED
    held = len(_PREVIEW_REFUSED)

    del tracked
    gc.collect()
    assert len(_PREVIEW_REFUSED) == held - 1, "the refusal outlived the client it described"


def test_a_broken_preview_is_recorded_once_and_never_blocks_gathering() -> None:
    """A signature mismatch does not heal, so it is asked once; and it must not
    be read as "no budget", which would stop the producer on a bad answer."""
    from labpilot.research_engine.conductor.producer import _PREVIEW_REFUSED, _budget_headroom

    broken = _PreviewClient(raises=True)
    assert _budget_headroom(broken, 0.2) is None
    assert _budget_headroom(broken, 0.2) is None
    assert broken.previews == 1, "a client whose preview raised must not be asked again"
    assert broken in _PREVIEW_REFUSED

    healthy = _PreviewClient(raises=False)
    reason = _budget_headroom(healthy, 0.2)
    assert healthy.previews == 1, "a different client must be consulted on its own terms"
    assert reason is not None and "holding 20%" in reason

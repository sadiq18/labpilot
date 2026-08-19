"""`gather_once` — the evidence routine as a callable unit (M16).

The gate `should_gather_evidence` has shipped since M21; what did not exist was
a way to call it, and the pipeline behind it, without going through the policy
step this milestone exists to bypass. These tests pin the unit's two jobs —
honour the gate, report what happened — and the boundary that keeps it usable
outside Kaggle.
"""

from __future__ import annotations

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
    registry.register(ToolDescriptor(name=name, handler=handler))
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
    registry.register(ToolDescriptor(name="gather_stub", handler=handler))
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
    registry.register(ToolDescriptor(name="gather_stub", handler=handler))

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
    registry.register(ToolDescriptor(name="gather_stub", handler=handler))
    producer = EvidenceProducer(ws, registry, plan=GatherPlan(tool="gather_stub"))

    assert producer.tick_once() is None

    status = producer.status()
    assert status["ticks"] == 1
    assert "network died" in status["last_error"]


def test_status_reports_the_gate_reason_a_skipped_tick_gave(tmp_path: Path) -> None:
    from labpilot.research_engine.conductor.producer import EvidenceProducer

    ws = _quiet_workspace(tmp_path)
    calls: list[dict] = []
    producer = EvidenceProducer(
        ws, _registry(calls), plan=GatherPlan(tool="gather_stub")
    )

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

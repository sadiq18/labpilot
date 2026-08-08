"""The instrument must reach the agents it was built to measure.

The sink is installed once per process on the main thread; the experiment path
runs through `anyio.to_thread.run_sync`. So `CodeEngineerAgent`, `AiderAgent`
and `DeltaBriefAgent` all record from a worker thread.

Thread-confined, every one of those writes raised inside `record_invocation` —
which swallows failures at debug level, because telemetry must never break a
run. The instrument therefore had a hole exactly where the most important agent
runs, and the hole was silent.

Measured on rogii 2026-08-09: a campaign in which aider ran three times recorded
**three `ConductorPolicy` rows and nothing else**, ConductorPolicy being the one
caller on the main thread. It is also why the 08-08 evidence log records
`CodeEngineerAgent` invoked "exactly once" — it ran far more often and the rows
were dropped.

Third instance of this bug, after `BudgetLedger` and `PromptCache`.
"""

from __future__ import annotations

import sqlite3
import threading

from labpilot.accessor.common.provenance import (
    AgentInvocation,
    record_invocation,
    reset_sink,
    set_sink,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.telemetry.agent_provenance import SqliteInvocationSink

_COMP = "rogii-wellbore-geology-prediction"


def _sink(tmp_path) -> SqliteInvocationSink:
    return SqliteInvocationSink(tmp_path / "knowledge", _COMP)


def _rows(tmp_path) -> list[str]:
    db = ResearchPaths(tmp_path / "knowledge", _COMP).db_path
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute("SELECT agent FROM agent_invocations")]
    finally:
        conn.close()


def _in_thread(fn):
    box: dict[str, BaseException] = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            box["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=10)
    if "error" in box:
        raise box["error"]


def test_an_agent_on_a_worker_thread_is_recorded(tmp_path):
    """The exact call that was being dropped for every experiment-path agent."""
    sink = _sink(tmp_path)
    try:
        _in_thread(lambda: sink.record(AgentInvocation(agent="aider", generated_by="aider")))
        assert _rows(tmp_path) == ["aider"]
    finally:
        sink.close()


def test_record_invocation_reaches_the_sink_from_a_worker_thread(tmp_path):
    """Through the *production* path, not a lookalike.

    `anyio.to_thread.run_sync` is what `agents/experiment.py` uses, and it
    copies the calling context into the worker so the sink ContextVar is
    visible there. A bare `threading.Thread` does not — an earlier version of
    this test used one, saw nothing recorded, and would have reported a
    ContextVar problem that production does not have. Verification has to call
    what production calls.
    """
    import anyio

    sink = _sink(tmp_path)
    token = set_sink(sink)

    async def go():
        await anyio.to_thread.run_sync(
            lambda: record_invocation(agent="DeltaBriefAgent", generated_by="llm")
        )

    try:
        anyio.run(go)
        assert _rows(tmp_path) == ["DeltaBriefAgent"]
    finally:
        reset_sink(token)
        sink.close()


def test_the_main_thread_still_works(tmp_path):
    """ConductorPolicy records from the main thread and must not regress."""
    sink = _sink(tmp_path)
    try:
        sink.record(AgentInvocation(agent="ConductorPolicy", generated_by="llm"))
        assert _rows(tmp_path) == ["ConductorPolicy"]
    finally:
        sink.close()


def test_concurrent_agents_do_not_lose_rows(tmp_path):
    """`allow_cross_thread` makes cross-thread use possible; the lock is what
    makes it safe. sqlite tolerates cross-thread, not concurrent."""
    sink = _sink(tmp_path)
    try:
        threads = [
            threading.Thread(
                target=lambda i=i: sink.record(
                    AgentInvocation(agent=f"agent-{i}", generated_by="llm")
                )
            )
            for i in range(16)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert sorted(_rows(tmp_path)) == sorted(f"agent-{i}" for i in range(16))
    finally:
        sink.close()


def test_cross_thread_is_opt_in_for_everyone_else(tmp_path):
    """Domain stores run SQL against `conn` without taking any lock, so a global
    flip would make cross-thread use possible everywhere and safe nowhere."""
    from labpilot.accessor.sqlite import SqliteClient

    client = SqliteClient(tmp_path / "plain.db")
    try:
        with_raise: dict[str, BaseException] = {}

        def touch():
            try:
                client.conn.execute("SELECT 1").fetchone()
            except BaseException as exc:  # noqa: BLE001
                with_raise["error"] = exc

        thread = threading.Thread(target=touch)
        thread.start()
        thread.join(timeout=10)

        assert isinstance(with_raise.get("error"), sqlite3.ProgrammingError)
    finally:
        client.close()

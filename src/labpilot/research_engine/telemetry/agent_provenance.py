"""SQLite sink and reports for micro-agent provenance.

The sink is installed for the duration of a campaign; the reports are what M14's
remaining phases actually consult:

* :func:`llm_failure_report` answers 2b — how often does the LLM path fail, and
  with which kind of error? Making failure fatal is only safe if the dominant
  failure mode is one that retrying or re-asking fixes.
* :func:`rule_engine_fire_report` answers 3 — which rule engines ever fire? One
  that never fires is dead code; one that fires constantly is either load-bearing
  domain logic to promote, or it is masking a persistent LLM failure, and the
  ``failure_kind`` column says which.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labpilot.accessor.common.provenance import (
    AgentInvocation,
    record_invocation,  # noqa: F401 — re-exported for callers
    reset_run_context,
    reset_sink,
    set_run_context,
    set_sink,
)
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.intelligence.paths import ResearchPaths

logger = logging.getLogger(__name__)


class SqliteInvocationSink:
    """Append-only writer for ``agent_invocations``.

    Cross-thread by necessity. The sink is installed once per process in the
    CLI callback, on the main thread, while the experiment path runs through
    `anyio.to_thread.run_sync` — so `CodeEngineerAgent`, `AiderAgent` and
    `DeltaBriefAgent` all record from a worker thread.

    Thread-confined, every one of those writes raised inside
    `record_invocation`, which swallows failures at debug level because
    telemetry must never break a run. The result was an instrument with a hole
    exactly where the most important agent runs: measured on rogii 2026-08-09, a
    campaign in which aider ran three times recorded **three `ConductorPolicy`
    rows and nothing else** — ConductorPolicy being the one caller on the main
    thread. It is also why the 08-08 log records `CodeEngineerAgent` invoked
    "exactly once": it was invoked far more often and the rows were dropped.

    Third instance of this bug in the same codebase, after `BudgetLedger` and
    `PromptCache`. The lock lives here rather than in the caller for the reason
    `budget.py` records: a rule that holds only while every caller remembers to
    take a lock is a rule that lapses the first time one does not.
    """

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        paths = ResearchPaths(Path(knowledge_dir), competition).ensure()
        self._client = SqliteClient(paths.db_path, allow_cross_thread=True)
        self._lock = threading.RLock()

    def record(self, invocation: AgentInvocation) -> None:
        with self._lock:
            self._record_locked(invocation)

    def _record_locked(self, invocation: AgentInvocation) -> None:
        self._client.conn.execute(
            """
            INSERT INTO agent_invocations (
                competition_slug, session_id, agent, llm_role, generated_by,
                failure_reason, failure_kind, attempts, provider, model,
                latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invocation.competition_slug,
                invocation.session_id,
                invocation.agent,
                invocation.llm_role,
                invocation.generated_by,
                invocation.failure_reason,
                invocation.failure_kind,
                invocation.attempts,
                invocation.provider,
                invocation.model,
                invocation.latency_ms,
                invocation.created_at,
            ),
        )
        self._client.conn.commit()

    def close(self) -> None:
        self._client.close()


@contextmanager
def recording_provenance(
    knowledge_dir: Path, competition: str, *, session_id: str | None = None
) -> Iterator[SqliteInvocationSink | None]:
    """Install the sink for the duration of a campaign.

    Yields None and records nothing if the store cannot be opened. Telemetry
    that can abort a run is worse than no telemetry.
    """
    sink: SqliteInvocationSink | None = None
    try:
        sink = SqliteInvocationSink(knowledge_dir, competition)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent provenance disabled: %s", exc)
    sink_token = ctx_tokens = None
    if sink is not None:
        sink_token = set_sink(sink)
        ctx_tokens = set_run_context(competition=competition, session_id=session_id)
    try:
        yield sink
    finally:
        # Restore rather than clear: `cli/main.py` installs a process-wide sink,
        # and setting None here would silently stop recording any agent work
        # that continues after the campaign in the same process.
        if sink_token is not None:
            reset_sink(sink_token)
        if ctx_tokens is not None:
            reset_run_context(ctx_tokens)
        if sink is not None:
            sink.close()


# --- reports ----------------------------------------------------------------


@dataclass
class AgentStat:
    agent: str
    total: int
    llm: int
    rule_engine: int

    @property
    def fallback_rate(self) -> float:
        return (self.rule_engine / self.total) if self.total else 0.0


def _rows(knowledge_dir: Path, competition: str, sql: str, args: tuple = ()) -> list[Any]:
    paths = ResearchPaths(Path(knowledge_dir), competition)
    if not paths.db_path.is_file():
        return []
    client = SqliteClient(paths.db_path)
    try:
        return list(client.conn.execute(sql, args))
    except Exception as exc:  # noqa: BLE001
        logger.warning("provenance query failed: %s", exc)
        return []
    finally:
        client.close()


def rule_engine_fire_report(
    knowledge_dir: Path, competition: str, *, session_id: str | None = None
) -> list[AgentStat]:
    """Per-agent counts, most-fallback-first. Phase 3's triage input."""
    where = "WHERE competition_slug = ?"
    args: tuple = (competition,)
    if session_id:
        where += " AND session_id = ?"
        args = (competition, session_id)
    rows = _rows(
        knowledge_dir,
        competition,
        f"""
        SELECT agent,
               COUNT(*) AS total,
               SUM(CASE WHEN generated_by = 'llm' THEN 1 ELSE 0 END) AS llm,
               SUM(CASE WHEN generated_by != 'llm' THEN 1 ELSE 0 END) AS rule_engine
        FROM agent_invocations {where}
        GROUP BY agent
        """,
        args,
    )
    stats = [
        AgentStat(
            agent=r["agent"], total=r["total"], llm=r["llm"] or 0, rule_engine=r["rule_engine"] or 0
        )
        for r in rows
    ]
    return sorted(stats, key=lambda s: (-s.fallback_rate, -s.total, s.agent))


def llm_failure_report(
    knowledge_dir: Path, competition: str, *, session_id: str | None = None
) -> dict[str, int]:
    """Counts by ``failure_kind``. Phase 2b's go/no-go input.

    The key that matters is ``json_shape``: a model answering in prose is the
    failure 2b would turn into a hard abort, and unlike a rate limit it does not
    clear by waiting.
    """
    where = "WHERE competition_slug = ? AND failure_kind IS NOT NULL"
    args: tuple = (competition,)
    if session_id:
        where += " AND session_id = ?"
        args = (competition, session_id)
    rows = _rows(
        knowledge_dir,
        competition,
        f"SELECT failure_kind, COUNT(*) AS n FROM agent_invocations {where} GROUP BY failure_kind",
        args,
    )
    return {r["failure_kind"]: r["n"] for r in rows}


def invocation_totals(knowledge_dir: Path, competition: str) -> dict[str, int]:
    rows = _rows(
        knowledge_dir,
        competition,
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN generated_by = 'llm' THEN 1 ELSE 0 END) AS llm
        FROM agent_invocations WHERE competition_slug = ?
        """,
        (competition,),
    )
    if not rows:
        return {"total": 0, "llm": 0, "rule_engine": 0}
    total, llm = rows[0]["total"] or 0, rows[0]["llm"] or 0
    return {"total": total, "llm": llm, "rule_engine": total - llm}

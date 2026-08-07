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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labpilot.accessor.common.provenance import (
    AgentInvocation,
    record_invocation,  # noqa: F401 — re-exported for callers
    set_run_context,
    set_sink,
)
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.intelligence.paths import ResearchPaths

logger = logging.getLogger(__name__)


class SqliteInvocationSink:
    """Append-only writer for ``agent_invocations``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        paths = ResearchPaths(Path(knowledge_dir), competition).ensure()
        self._client = SqliteClient(paths.db_path)

    def record(self, invocation: AgentInvocation) -> None:
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
    if sink is not None:
        set_sink(sink)
        set_run_context(competition=competition, session_id=session_id)
    try:
        yield sink
    finally:
        set_sink(None)
        set_run_context()
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

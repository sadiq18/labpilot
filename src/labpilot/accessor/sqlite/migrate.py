"""Idempotent migrator for the unified SQLite schema.

There is exactly **one** ``schema.sql`` and one migrator for ``knowledge.db``.
Every pillar (intelligence / planner / execution / reflection) reaches SQLite through
:class:`~labpilot.accessor.sqlite.client.SqliteClient`, which runs this migration
on open. Migration is ``CREATE TABLE IF NOT EXISTS`` throughout, so re-running on
an existing DB adds any new tables without touching existing data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Single source of record for the schema DDL.
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Bumped when the unified schema gains tables/columns. Recorded in
#: ``schema_meta``. v3 introduces ``research_plans`` / ``research_tasks`` /
#: ``research_task_deps`` (Research Planner). v4 adds ``research_executions``
#: (Research Engineer). v5 adds reflection tables (``experiment_evidence``,
#: ``belief_updates``, ``lessons``, ``research_claims``, ``claim_evidence``).
#: v6 adds Conductor tables (``os_sessions``, ``os_tasks``, ``os_decisions``,
#: ``os_operator_feedback``).
SCHEMA_VERSION = "6"


def run_migration(conn: sqlite3.Connection) -> None:
    """Apply the unified schema to ``conn`` and record the schema version."""
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()

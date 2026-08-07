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
#: ``os_operator_feedback``). v7 adds Campaign Engine tables
#: (``os_suggestions``, ``os_campaign_metrics``). v8 adds ``experience_records``.
#: v9 adds capability gap ledger (``os_capability_gaps``, ``os_capability_decisions``).
#: v10 adds ``agent_invocations`` — durable micro-agent provenance, without
#: which M14 phases 2b and 3 have no data to decide on.
#: v11 adds ``techniques.status`` and ``technique_status_history`` (M-25 step 1).
SCHEMA_VERSION = "11"


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _apply_incremental_migrations(conn: sqlite3.Connection) -> None:
    """DDL that ``CREATE TABLE IF NOT EXISTS`` cannot apply to existing tables."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "techniques" in tables and not _table_has_column(conn, "techniques", "status"):
        conn.execute(
            "ALTER TABLE techniques ADD COLUMN status TEXT NOT NULL DEFAULT 'candidate'"
        )
    if "techniques" in tables:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_techniques_status ON techniques(status)"
        )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS technique_status_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            technique_id     TEXT NOT NULL,
            competition_slug TEXT NOT NULL DEFAULT '',
            from_status      TEXT,
            to_status        TEXT NOT NULL,
            reason           TEXT NOT NULL,
            evidence_card_id TEXT,
            observations     INTEGER NOT NULL DEFAULT 0,
            signed_net       REAL,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_technique_status_history_technique
            ON technique_status_history(technique_id);
        """
    )
    if "technique_status_history" in tables and _table_has_column(
        conn, "technique_status_history", "net_effect"
    ) and not _table_has_column(conn, "technique_status_history", "signed_net"):
        # Step-1 preview DBs may still have the misnamed column.
        conn.execute(
            "ALTER TABLE technique_status_history RENAME COLUMN net_effect TO signed_net"
        )


def run_migration(conn: sqlite3.Connection) -> None:
    """Apply the unified schema to ``conn`` and record the schema version."""
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_incremental_migrations(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()

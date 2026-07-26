"""Unified SQLite access: client, migrator, and the single ``schema.sql``."""

from labpilot.accessor.sqlite.client import SqliteClient
from labpilot.accessor.sqlite.migrate import SCHEMA_PATH, SCHEMA_VERSION, run_migration

__all__ = ["SqliteClient", "SCHEMA_PATH", "SCHEMA_VERSION", "run_migration"]

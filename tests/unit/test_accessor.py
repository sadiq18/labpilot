from pathlib import Path

from labpilot.accessor.common.ids import allocate_sequential_id, task_id
from labpilot.accessor.common.json_utils import dumps, loads
from labpilot.accessor.sqlite import SCHEMA_VERSION, SqliteClient


def test_sqlite_client_migrates_and_enables_foreign_keys(tmp_path: Path):
    client = SqliteClient(tmp_path / "sub" / "knowledge.db")
    try:
        assert client.schema_version() == SCHEMA_VERSION
        fk = client.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        tables = {
            row["name"]
            for row in client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"research_plans", "research_tasks", "research_task_deps", "research_executions"} <= tables
        # Layer-3 knowledge ontology table is untouched.
        assert "tasks" in tables
    finally:
        client.close()


def test_migrate_is_idempotent(tmp_path: Path):
    db = tmp_path / "knowledge.db"
    SqliteClient(db).close()
    client = SqliteClient(db)
    try:
        assert client.schema_version() == SCHEMA_VERSION
    finally:
        client.close()


def test_allocate_sequential_id():
    assert allocate_sequential_id("P", []) == "P-001"
    assert allocate_sequential_id("P", ["P-001", "P-002"]) == "P-003"
    assert allocate_sequential_id("P", ["P-009", "noise", "H-050"]) == "P-010"


def test_task_id():
    assert task_id("P-001", 1) == "P-001-T01"
    assert task_id("P-001", 12) == "P-001-T12"


def test_json_helpers_round_trip():
    assert loads(dumps(["a", "b"]), []) == ["a", "b"]
    assert loads("", []) == []
    assert loads("not-json", {"fallback": 1}) == {"fallback": 1}

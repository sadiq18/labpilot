"""ExecutionStore — durable CRUD for ``research_executions`` (``E-xxx``).

The DB is the source of record for execution attempts. Task status remains on
``research_tasks`` (updated via :class:`PlanStore`); evidence lives on disk under
``…/executions/E-xxx/evidence/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.accessor.common import allocate_sequential_id
from labpilot.accessor.common.json_utils import dumps, loads
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.execution.evidence import ensure_execution_layout
from labpilot.research_engine.execution.schemas import ExecutionStatus, ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.workspace import competition_workspace_path

_EXEC_ID_PREFIX = "E"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class ExecutionStore:
    """CRUD for ``research_executions``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self.competition = competition
        self._client = SqliteClient(self.paths.db_path)
        self._conn = self._client.conn

    def close(self) -> None:
        self._client.close()

    def default_workspace_path(self) -> Path:
        return competition_workspace_path(self.knowledge_dir, self.competition)

    def new_execution_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM research_executions").fetchall()
        return allocate_sequential_id(_EXEC_ID_PREFIX, (row["id"] for row in rows))

    def create_execution(
        self,
        plan_id: str,
        *,
        workspace_path: str | None = None,
        runtime_target: str | None = None,
        metadata: dict | None = None,
        status: ExecutionStatus = "pending",
    ) -> ResearchExecution:
        """Allocate ``E-xxx``, insert row, and ensure on-disk layout.

        Default ``workspace_path`` is the competition code root (client
        ``labpilot.yaml`` workspace, or legacy ``competitions/<slug>/``).
        Execution evidence stays under ``executions/E-xxx/``.
        """
        # FK: plan must exist.
        plan_row = self._conn.execute(
            "SELECT id FROM research_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if plan_row is None:
            raise ValueError(f"unknown plan_id: {plan_id}")

        exec_id = self.new_execution_id()
        now = _now()
        ensure_execution_layout(self.paths, exec_id)
        workspace = Path(workspace_path) if workspace_path else self.default_workspace_path()
        workspace.mkdir(parents=True, exist_ok=True)

        self._conn.execute(
            """
            INSERT INTO research_executions (
                id, plan_id, competition_slug, status, workspace_path,
                runtime_target, experiment_id, error, metadata,
                created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, NULL, NULL)
            """,
            (
                exec_id,
                plan_id,
                self.competition,
                status,
                str(workspace),
                runtime_target,
                dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        execution = self.get_execution(exec_id)
        assert execution is not None
        return execution

    def get_execution(self, execution_id: str) -> ResearchExecution | None:
        row = self._conn.execute(
            "SELECT * FROM research_executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_execution(row)

    def list_executions(
        self,
        *,
        plan_id: str | None = None,
        status: ExecutionStatus | str | None = None,
    ) -> list[ResearchExecution]:
        clauses: list[str] = []
        params: list[str] = []
        if plan_id is not None:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM research_executions{where} ORDER BY id",  # noqa: S608
            params,
        ).fetchall()
        return [self._row_to_execution(row) for row in rows]

    def update_status(
        self,
        execution_id: str,
        status: ExecutionStatus | str,
        *,
        error: str | None = None,
        experiment_id: str | None = None,
        metadata_patch: dict | None = None,
    ) -> None:
        now = _now()
        row = self._conn.execute(
            "SELECT * FROM research_executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown execution_id: {execution_id}")

        started_at = row["started_at"]
        completed_at = row["completed_at"]
        status_s = str(status)
        if status_s == "running" and not started_at:
            started_at = now
        if status_s in {"succeeded", "failed", "cancelled"} and not completed_at:
            completed_at = now

        metadata = loads(row["metadata"], {})
        if metadata_patch:
            metadata.update(metadata_patch)

        self._conn.execute(
            """
            UPDATE research_executions SET
                status = ?, error = ?, experiment_id = COALESCE(?, experiment_id),
                metadata = ?, updated_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status_s,
                error if error is not None else row["error"],
                experiment_id,
                dumps(metadata),
                now,
                started_at,
                completed_at,
                execution_id,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_execution(row) -> ResearchExecution:
        return ResearchExecution(
            id=row["id"],
            plan_id=row["plan_id"],
            competition=row["competition_slug"] or "",
            status=row["status"],
            workspace_path=row["workspace_path"],
            runtime_target=row["runtime_target"],
            experiment_id=row["experiment_id"],
            error=row["error"],
            metadata=loads(row["metadata"], {}),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
        )

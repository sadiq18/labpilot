"""SQLite persistence for Conductor sessions, tasks, decisions, feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.accessor.common import allocate_sequential_id
from labpilot.accessor.common.json_utils import dumps, loads
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.conductor.models import (
    ApprovalResult,
    ConductSession,
    DecisionRecord,
    OperatorFeedback,
    OsTask,
    TaskStatus,
    _now,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths

_SESSION_PREFIX = "S"
_TASK_PREFIX = "T"
_DECISION_PREFIX = "D"
_FEEDBACK_PREFIX = "F"


class ConductorStore:
    """CRUD for OS session / queue / decision log under a competition knowledge DB."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self._client = SqliteClient(self.paths.db_path)
        self._conn = self._client.conn

    def close(self) -> None:
        self._client.close()

    # -- sessions ----------------------------------------------------------

    def create_session(
        self,
        goal: str,
        *,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> ConductSession:
        sid = self._new_id(_SESSION_PREFIX, "os_sessions")
        now = _now()
        meta = metadata or {}
        self._conn.execute(
            """
            INSERT INTO os_sessions (id, competition, goal, status, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sid, self.competition, goal, status, dumps(meta), now, now),
        )
        self._conn.commit()
        session = self.get_session(sid)
        assert session is not None
        return session

    def get_session(self, session_id: str) -> ConductSession | None:
        row = self._conn.execute(
            "SELECT * FROM os_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(self) -> list[ConductSession]:
        rows = self._conn.execute(
            "SELECT * FROM os_sessions WHERE competition = ? ORDER BY id",
            (self.competition,),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_session_status(self, session_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE os_sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), session_id),
        )
        self._conn.commit()

    # -- tasks -------------------------------------------------------------

    def enqueue(
        self,
        session_id: str,
        tool_name: str,
        *,
        args: dict[str, Any] | None = None,
        priority: int = 0,
        dependencies: list[str] | None = None,
        decision_id: str | None = None,
        max_retries: int = 1,
    ) -> OsTask:
        tid = self._new_id(_TASK_PREFIX, "os_tasks")
        now = _now()
        self._conn.execute(
            """
            INSERT INTO os_tasks (
                id, session_id, tool_name, status, priority, retry_count, max_retries,
                args_json, dependencies_json, artifact_refs_json, error, decision_id,
                created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, 'pending', ?, 0, ?, ?, ?, '[]', NULL, ?, ?, ?, NULL, NULL)
            """,
            (
                tid,
                session_id,
                tool_name,
                priority,
                max_retries,
                dumps(args or {}),
                dumps(dependencies or []),
                decision_id,
                now,
                now,
            ),
        )
        self._conn.commit()
        task = self.get_task(tid)
        assert task is not None
        return task

    def get_task(self, task_id: str) -> OsTask | None:
        row = self._conn.execute(
            "SELECT * FROM os_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(
        self,
        session_id: str,
        *,
        status: TaskStatus | str | None = None,
    ) -> list[OsTask]:
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM os_tasks WHERE session_id = ? ORDER BY priority DESC, id",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM os_tasks
                WHERE session_id = ? AND status = ?
                ORDER BY priority DESC, id
                """,
                (session_id, str(status)),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def ready_tasks(self, session_id: str) -> list[OsTask]:
        """Pending tasks whose dependencies are all completed."""
        pending = self.list_tasks(session_id, status="pending")
        completed_ids = {
            t.id for t in self.list_tasks(session_id) if t.status == "completed"
        }
        ready: list[OsTask] = []
        for task in pending:
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return ready

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus | str,
        *,
        error: str | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> OsTask:
        now = _now()
        row = self._conn.execute(
            "SELECT * FROM os_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown task_id: {task_id}")
        started_at = row["started_at"]
        completed_at = row["completed_at"]
        status_s = str(status)
        if status_s == "running" and not started_at:
            started_at = now
        if status_s in {"completed", "failed", "cancelled", "blocked"} and not completed_at:
            completed_at = now
        refs = (
            dumps(artifact_refs)
            if artifact_refs is not None
            else row["artifact_refs_json"]
        )
        err = error if error is not None else row["error"]
        retry = row["retry_count"]
        if status_s == "retry":
            retry = int(row["retry_count"]) + 1
            status_s = "pending"
            completed_at = None
        self._conn.execute(
            """
            UPDATE os_tasks SET
                status = ?, error = ?, artifact_refs_json = ?, retry_count = ?,
                updated_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (status_s, err, refs, retry, now, started_at, completed_at, task_id),
        )
        self._conn.commit()
        task = self.get_task(task_id)
        assert task is not None
        return task

    # -- decisions / feedback ----------------------------------------------

    def append_decision(self, record: DecisionRecord) -> DecisionRecord:
        self._conn.execute(
            """
            INSERT INTO os_decisions (
                id, session_id, tool_name, rationale, stop, args_json, observe_json,
                approval_json, artifact_refs_json, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.tool_name,
                record.rationale,
                1 if record.stop else 0,
                dumps(record.args),
                dumps(record.observe),
                dumps(record.approval.model_dump()) if record.approval else None,
                dumps(record.artifact_refs),
                record.task_id,
                record.created_at,
            ),
        )
        self._conn.commit()
        return record

    def new_decision_id(self) -> str:
        return self._new_id(_DECISION_PREFIX, "os_decisions")

    def list_decisions(self, session_id: str) -> list[DecisionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM os_decisions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def append_feedback(self, feedback: OperatorFeedback) -> OperatorFeedback:
        self._conn.execute(
            """
            INSERT INTO os_operator_feedback (
                id, session_id, gated_tool, decision, comment, decision_id, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.id,
                feedback.session_id,
                feedback.gated_tool,
                feedback.decision,
                feedback.comment,
                feedback.decision_id,
                feedback.task_id,
                feedback.created_at,
            ),
        )
        self._conn.commit()
        return feedback

    def new_feedback_id(self) -> str:
        return self._new_id(_FEEDBACK_PREFIX, "os_operator_feedback")

    def list_feedback(self, session_id: str, *, limit: int = 20) -> list[OperatorFeedback]:
        rows = self._conn.execute(
            """
            SELECT * FROM os_operator_feedback
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [self._row_to_feedback(r) for r in reversed(rows)]

    # -- helpers -----------------------------------------------------------

    def _new_id(self, prefix: str, table: str) -> str:
        rows = self._conn.execute(f"SELECT id FROM {table}").fetchall()  # noqa: S608
        return allocate_sequential_id(prefix, (row["id"] for row in rows))

    def _row_to_session(self, row: Any) -> ConductSession:
        return ConductSession(
            id=row["id"],
            competition=row["competition"],
            goal=row["goal"],
            status=row["status"],
            metadata=loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_task(self, row: Any) -> OsTask:
        return OsTask(
            id=row["id"],
            session_id=row["session_id"],
            tool_name=row["tool_name"],
            status=row["status"],
            priority=row["priority"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            args=loads(row["args_json"], {}),
            dependencies=loads(row["dependencies_json"], []),
            artifact_refs=loads(row["artifact_refs_json"], []),
            error=row["error"],
            decision_id=row["decision_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_decision(self, row: Any) -> DecisionRecord:
        approval_raw = loads(row["approval_json"], None) if row["approval_json"] else None
        approval = ApprovalResult.model_validate(approval_raw) if approval_raw else None
        return DecisionRecord(
            id=row["id"],
            session_id=row["session_id"],
            tool_name=row["tool_name"],
            rationale=row["rationale"] or "",
            stop=bool(row["stop"]),
            args=loads(row["args_json"], {}),
            observe=loads(row["observe_json"], {}),
            approval=approval,
            artifact_refs=loads(row["artifact_refs_json"], []),
            task_id=row["task_id"],
            created_at=row["created_at"],
        )

    def _row_to_feedback(self, row: Any) -> OperatorFeedback:
        return OperatorFeedback(
            id=row["id"],
            session_id=row["session_id"],
            gated_tool=row["gated_tool"],
            decision=row["decision"],
            comment=row["comment"] or "",
            decision_id=row["decision_id"],
            task_id=row["task_id"],
            created_at=row["created_at"],
        )

"""PlanStore — durable read/write for research plans and their task DAGs.

The DB is the source of record. PlanStore persists a :class:`ResearchPlan` in a
single transaction (plan row + replace tasks + replace dependency edges) and
reassembles it on read. It reaches SQLite through the shared accessor
:class:`SqliteClient` — the planner never imports the intelligence pillar for
infrastructure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.accessor.commons import allocate_sequential_id
from labpilot.accessor.commons.json_utils import dumps, loads
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
    TaskVerification,
)
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    RuntimeTarget,
    TaskStatus,
    TaskType,
)

_PLAN_ID_PREFIX = "P"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PlanStore:
    """CRUD for ``research_plans`` / ``research_tasks`` / ``research_task_deps``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self.competition = competition
        self._client = SqliteClient(self.paths.db_path)
        self._conn = self._client.conn

    def close(self) -> None:
        self._client.close()

    # -- id allocation -----------------------------------------------------

    def new_plan_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM research_plans").fetchall()
        return allocate_sequential_id(_PLAN_ID_PREFIX, (row["id"] for row in rows))

    # -- write -------------------------------------------------------------

    def upsert_plan(self, plan: ResearchPlan) -> str:
        """Persist a plan + tasks + dependency edges in one transaction."""
        now = _now()
        conn = self._conn
        existing = conn.execute(
            "SELECT created_at FROM research_plans WHERE id = ?", (plan.id,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now

        conn.execute(
            """
            INSERT INTO research_plans (
                id, competition_slug, hypothesis_id, goal, current_state,
                expected_outcome, status, priority, estimated_gain, risk,
                estimated_cost, estimated_duration, runtime_target,
                success_criteria, rollback, artifacts, generated_by, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                competition_slug=excluded.competition_slug,
                hypothesis_id=excluded.hypothesis_id, goal=excluded.goal,
                current_state=excluded.current_state,
                expected_outcome=excluded.expected_outcome, status=excluded.status,
                priority=excluded.priority, estimated_gain=excluded.estimated_gain,
                risk=excluded.risk, estimated_cost=excluded.estimated_cost,
                estimated_duration=excluded.estimated_duration,
                runtime_target=excluded.runtime_target,
                success_criteria=excluded.success_criteria, rollback=excluded.rollback,
                artifacts=excluded.artifacts, generated_by=excluded.generated_by,
                metadata=excluded.metadata, updated_at=excluded.updated_at
            """,
            (
                plan.id,
                plan.competition or self.competition,
                plan.hypothesis_id,
                plan.goal,
                plan.current_state,
                plan.expected_outcome,
                str(plan.status),
                plan.priority,
                plan.estimated_gain,
                plan.risk,
                plan.estimated_cost,
                plan.estimated_duration,
                str(plan.runtime_target) if plan.runtime_target else None,
                dumps(plan.success_criteria),
                plan.rollback,
                dumps(plan.artifacts),
                plan.generated_by,
                dumps(plan.metadata),
                created_at,
                now,
            ),
        )

        # Replace the task set; cascade drops old dependency edges.
        conn.execute("DELETE FROM research_tasks WHERE plan_id = ?", (plan.id,))
        for task in plan.tasks:
            conn.execute(
                """
                INSERT INTO research_tasks (
                    id, plan_id, parent_task_id, task_type, description, inputs,
                    outputs, status, verification, retry_policy, order_index,
                    estimated_cost, estimated_time, metadata, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    plan.id,
                    str(task.type),
                    task.description,
                    dumps(task.inputs),
                    dumps(task.outputs),
                    str(task.status),
                    dumps(task.verification.model_dump()),
                    dumps(task.retry_policy.model_dump()),
                    task.order,
                    task.estimated_cost,
                    task.estimated_time,
                    dumps(task.metadata),
                    now,
                    now,
                ),
            )
        # Second pass: wire parents now that every row exists (FK-safe).
        for task in plan.tasks:
            if task.parent_task_id:
                conn.execute(
                    "UPDATE research_tasks SET parent_task_id = ? WHERE id = ?",
                    (task.parent_task_id, task.id),
                )
        for task in plan.tasks:
            for dep in task.dependencies:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO research_task_deps (task_id, depends_on)
                    VALUES (?, ?)
                    """,
                    (task.id, dep),
                )
        conn.commit()
        return plan.id

    def update_plan_status(self, plan_id: str, status: PlanStatus | str) -> None:
        self._conn.execute(
            "UPDATE research_plans SET status = ?, updated_at = ? WHERE id = ?",
            (str(status), _now(), plan_id),
        )
        self._conn.commit()

    def update_task_status(self, task_id: str, status: TaskStatus | str) -> None:
        self._conn.execute(
            "UPDATE research_tasks SET status = ?, updated_at = ? WHERE id = ?",
            (str(status), _now(), task_id),
        )
        self._conn.commit()

    # -- read --------------------------------------------------------------

    def get_plan(self, plan_id: str) -> ResearchPlan | None:
        row = self._conn.execute(
            "SELECT * FROM research_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if row is None:
            return None
        tasks = self._load_tasks(plan_id)
        return self._row_to_plan(row, tasks)

    def list_plans(
        self,
        *,
        status: PlanStatus | str | None = None,
        hypothesis_id: str | None = None,
    ) -> list[ResearchPlan]:
        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM research_plans{where} ORDER BY id",  # noqa: S608 - bound params
            params,
        ).fetchall()
        return [self._row_to_plan(row, self._load_tasks(row["id"])) for row in rows]

    # -- helpers -----------------------------------------------------------

    def _load_tasks(self, plan_id: str) -> list[ResearchTask]:
        rows = self._conn.execute(
            "SELECT * FROM research_tasks WHERE plan_id = ? ORDER BY order_index, id",
            (plan_id,),
        ).fetchall()
        tasks: list[ResearchTask] = []
        for row in rows:
            dep_rows = self._conn.execute(
                "SELECT depends_on FROM research_task_deps WHERE task_id = ? ORDER BY depends_on",
                (row["id"],),
            ).fetchall()
            tasks.append(
                ResearchTask(
                    id=row["id"],
                    plan_id=row["plan_id"],
                    parent_task_id=row["parent_task_id"],
                    type=TaskType(row["task_type"]),
                    description=row["description"],
                    inputs=loads(row["inputs"], []),
                    outputs=loads(row["outputs"], []),
                    dependencies=[dep["depends_on"] for dep in dep_rows],
                    status=TaskStatus(row["status"]),
                    order=row["order_index"],
                    verification=TaskVerification.model_validate(
                        loads(row["verification"], {})
                    ),
                    retry_policy=RetryPolicy.model_validate(
                        loads(row["retry_policy"], {})
                    ),
                    estimated_cost=row["estimated_cost"],
                    estimated_time=row["estimated_time"],
                    metadata=loads(row["metadata"], {}),
                )
            )
        return tasks

    @staticmethod
    def _row_to_plan(row, tasks: list[ResearchTask]) -> ResearchPlan:
        runtime_target = row["runtime_target"]
        return ResearchPlan(
            id=row["id"],
            competition=row["competition_slug"] or "",
            hypothesis_id=row["hypothesis_id"] or "",
            goal=row["goal"],
            current_state=row["current_state"],
            expected_outcome=row["expected_outcome"],
            status=PlanStatus(row["status"]),
            priority=row["priority"],
            estimated_gain=row["estimated_gain"],
            risk=row["risk"],
            estimated_cost=row["estimated_cost"],
            estimated_duration=row["estimated_duration"],
            runtime_target=RuntimeTarget(runtime_target) if runtime_target else None,
            tasks=tasks,
            success_criteria=loads(row["success_criteria"], []),
            artifacts=loads(row["artifacts"], []),
            rollback=row["rollback"],
            generated_by=row["generated_by"],
            metadata=loads(row["metadata"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

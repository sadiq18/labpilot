"""ExperienceStore — cross-competition SQLite SoR for Experience Records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.accessor.common import allocate_sequential_id
from labpilot.accessor.common.json_utils import dumps, loads
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.memory.models import (
    ExperienceArtifacts,
    ExperienceFacet,
    ExperienceOutcome,
    ExperienceRecord,
)

_ID_PREFIX = "XR"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class ExperienceStore:
    """CRUD for ``experience_records`` in a shared DB outside competition workspaces.

    Path resolution (see :func:`labpilot.workspace.resolve_experience_db_path`):
    env → labpilot.yaml → parent research root → ``~/.labpilot/experiences.db``.

    Competition-local SoR remains ``knowledge.db`` under the workspace.
    """

    def __init__(
        self,
        knowledge_dir: Path,
        *,
        db_path: Path | None = None,
        workspace: object | None = None,
    ) -> None:
        from labpilot.workspace import CompetitionWorkspace, resolve_experience_db_path

        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        ws = workspace if isinstance(workspace, CompetitionWorkspace) else None
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else resolve_experience_db_path(knowledge_dir=self.knowledge_dir, workspace=ws)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = SqliteClient(self.db_path)
        self._conn = self._client.conn

    def close(self) -> None:
        self._client.close()

    def schema_version(self) -> str:
        return self._client.schema_version()

    def new_experience_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM experience_records").fetchall()
        return allocate_sequential_id(_ID_PREFIX, (row["id"] for row in rows))

    def get(self, experience_id: str) -> ExperienceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM experience_records WHERE id = ?",
            (experience_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_idempotency_key(self, key: str) -> ExperienceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM experience_records WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def upsert(self, record: ExperienceRecord) -> ExperienceRecord:
        """Insert or update by ``idempotency_key``; preserve ``id`` / ``created_at``."""
        now = _now()
        existing = self._conn.execute(
            "SELECT id, created_at FROM experience_records WHERE idempotency_key = ?",
            (record.idempotency_key,),
        ).fetchone()
        if existing is not None:
            experience_id = existing["id"]
            created_at = existing["created_at"]
        else:
            experience_id = record.id or self.new_experience_id()
            created_at = (
                record.created_at.isoformat() if record.created_at else now
            )

        self._conn.execute(
            """
            INSERT INTO experience_records (
                id, source_competition, goal, hypothesis, hypothesis_id,
                action, result, outcome, artifacts, tags, idempotency_key,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
                source_competition=excluded.source_competition,
                goal=excluded.goal,
                hypothesis=excluded.hypothesis,
                hypothesis_id=excluded.hypothesis_id,
                action=excluded.action,
                result=excluded.result,
                outcome=excluded.outcome,
                artifacts=excluded.artifacts,
                tags=excluded.tags,
                updated_at=excluded.updated_at
            """,
            (
                experience_id,
                record.source_competition,
                record.goal,
                record.hypothesis,
                record.hypothesis_id,
                record.action,
                record.result,
                record.outcome,
                dumps(record.artifacts.model_dump(mode="json")),
                dumps([f.model_dump(mode="json") for f in record.facets]),
                record.idempotency_key,
                created_at,
                now,
            ),
        )
        self._conn.commit()
        fetched = self.get(experience_id)
        assert fetched is not None
        return fetched

    def create(
        self,
        *,
        source_competition: str,
        idempotency_key: str,
        goal: str = "",
        hypothesis: str = "",
        hypothesis_id: str | None = None,
        action: str = "",
        result: str = "",
        outcome: ExperienceOutcome = "fail",
        artifacts: ExperienceArtifacts | dict[str, Any] | None = None,
        facets: list[ExperienceFacet] | list[dict[str, Any]] | list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ExperienceRecord:
        """Allocate id and upsert (create-or-update by idempotency key).

        ``tags`` is accepted as a convenience alias for facet names (legacy tests /
        callers); prefer structured ``facets``.
        """
        art = (
            artifacts
            if isinstance(artifacts, ExperienceArtifacts)
            else ExperienceArtifacts.model_validate(artifacts or {})
        )
        facet_list = _normalize_facets(facets, tags)
        now = datetime.now(UTC)
        existing = self.get_by_idempotency_key(idempotency_key)
        draft = ExperienceRecord(
            id=existing.id if existing else self.new_experience_id(),
            source_competition=source_competition,
            goal=goal,
            hypothesis=hypothesis,
            hypothesis_id=hypothesis_id,
            action=action,
            result=result,
            outcome=outcome,
            artifacts=art,
            facets=facet_list,
            idempotency_key=idempotency_key,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        return self.upsert(draft)

    def list(
        self,
        *,
        source_competition: str | None = None,
        outcome: ExperienceOutcome | str | None = None,
        facet: str | None = None,
        facets: list[str] | None = None,
        tag: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ExperienceRecord]:
        """List experiences; filters are cross-competition unless slug set."""
        clauses: list[str] = []
        params: list[Any] = []
        if source_competition is not None:
            clauses.append("source_competition = ?")
            params.append(source_competition)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(str(outcome))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experience_records{where} ORDER BY updated_at DESC, id"  # noqa: S608
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        records = [self._row_to_record(row) for row in rows]

        required = [t for t in (facets or tags or []) if t]
        if facet:
            required.append(facet)
        if tag:
            required.append(tag)
        if required:
            needed = {t.lower() for t in required}
            records = [
                r
                for r in records
                if needed <= {name.lower() for name in r.facet_names()}
            ]
        return records

    def _row_to_record(self, row: Any) -> ExperienceRecord:
        artifacts_raw = loads(row["artifacts"]) if row["artifacts"] else {}
        tags_raw = loads(row["tags"]) if row["tags"] else []
        return ExperienceRecord(
            id=row["id"],
            source_competition=row["source_competition"],
            goal=row["goal"] or "",
            hypothesis=row["hypothesis"] or "",
            hypothesis_id=row["hypothesis_id"],
            action=row["action"] or "",
            result=row["result"] or "",
            outcome=row["outcome"] if row["outcome"] in {"success", "fail"} else "fail",
            artifacts=ExperienceArtifacts.model_validate(artifacts_raw or {}),
            facets=tags_raw or [],
            idempotency_key=row["idempotency_key"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )


def _normalize_facets(
    facets: list[ExperienceFacet] | list[dict[str, Any]] | list[str] | None,
    tags: list[str] | None,
) -> list[ExperienceFacet]:
    if facets:
        return [
            item
            if isinstance(item, ExperienceFacet)
            else ExperienceFacet.model_validate(
                {"facet": item, "confidence": 0.5, "evidence": [], "source": "legacy"}
                if isinstance(item, str)
                else item
            )
            for item in facets
        ]
    if tags:
        return [
            ExperienceFacet(facet=t, confidence=0.5, evidence=[], source="legacy")
            for t in tags
            if t
        ]
    return []

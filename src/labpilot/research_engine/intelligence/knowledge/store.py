"""Knowledge Store — the storage system of record (knowledge-system.md §4).

Persists explored intelligence under ``knowledge/<slug>/research/`` with three
distinct layers (``raw/`` → ``extracted/`` → ``knowledge/``) plus a per-competition
``knowledge.db`` SQLite database. This is a **storage API only** — no LLM, no
retrieval ranking (that is Plans 8–9). ``knowledge.db`` wins for joins;
``reports/analyze.json`` stays a projection.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"

# Which extracted/ subfolder a per-source card is written to.
_EXTRACTED_BUCKET = {
    ResearchArtifactType.PAPER: "papers",
    ResearchArtifactType.REPOSITORY: "repositories",
    ResearchArtifactType.DISCUSSION: "forums",
}

_LIST_FIELDS = ("techniques", "models", "datasets", "claims")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    """File-system safe slug for an artifact id (``repo:owner/name`` → ``repo_owner_name``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "artifact"


def technique_id(name: str) -> str:
    """Deterministic id so the same technique name always merges to one row."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"tech_{slug}" if slug else "tech_unknown"


class KnowledgeStore:
    """Read/write access to one competition's research tree + ``knowledge.db``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self.competition = competition
        self._conn = sqlite3.connect(self.paths.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def _migrate(self) -> None:
        self._conn.executescript(SCHEMA_PATH.read_text())
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- artifacts ---------------------------------------------------------

    def _extracted_path(self, artifact: ResearchArtifact) -> Path:
        bucket = _EXTRACTED_BUCKET.get(artifact.type, "misc")
        directory = self.paths.extracted_dir / bucket
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{_safe_name(artifact.id)}.json"

    def upsert_artifact(self, artifact: ResearchArtifact) -> ResearchArtifact:
        """Persist a per-source card: a ``research_artifacts`` row + ``extracted/`` JSON."""
        now = _now()
        existing = self._conn.execute(
            "SELECT created_at FROM research_artifacts WHERE id = ?", (artifact.id,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now

        self._conn.execute(
            """
            INSERT INTO research_artifacts (
                id, type, source, title, summary, confidence, competition_slug,
                metadata, techniques, models, datasets, claims, refs,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type, source=excluded.source, title=excluded.title,
                summary=excluded.summary, confidence=excluded.confidence,
                competition_slug=excluded.competition_slug, metadata=excluded.metadata,
                techniques=excluded.techniques, models=excluded.models,
                datasets=excluded.datasets, claims=excluded.claims, refs=excluded.refs,
                updated_at=excluded.updated_at
            """,
            (
                artifact.id,
                str(artifact.type),
                artifact.source,
                artifact.title,
                artifact.summary,
                artifact.confidence,
                artifact.competition_slug or self.competition,
                json.dumps(artifact.metadata),
                json.dumps(artifact.techniques),
                json.dumps(artifact.models),
                json.dumps(artifact.datasets),
                json.dumps(artifact.claims),
                json.dumps(artifact.references),
                created_at,
                now,
            ),
        )
        self._conn.commit()
        self._extracted_path(artifact).write_text(artifact.model_dump_json(indent=2) + "\n")
        return artifact

    def get_artifact(self, artifact_id: str) -> ResearchArtifact | None:
        row = self._conn.execute(
            "SELECT * FROM research_artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return self._row_to_artifact(row) if row else None

    def list_artifacts(
        self, *, type: ResearchArtifactType | str | None = None
    ) -> list[ResearchArtifact]:
        if type is not None:
            rows = self._conn.execute(
                "SELECT * FROM research_artifacts WHERE type = ? ORDER BY id",
                (str(type),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM research_artifacts ORDER BY id"
            ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> ResearchArtifact:
        return ResearchArtifact(
            id=row["id"],
            type=ResearchArtifactType(row["type"]),
            source=row["source"],
            title=row["title"],
            summary=row["summary"],
            confidence=row["confidence"],
            competition_slug=row["competition_slug"],
            metadata=json.loads(row["metadata"]),
            techniques=json.loads(row["techniques"]),
            models=json.loads(row["models"]),
            datasets=json.loads(row["datasets"]),
            claims=json.loads(row["claims"]),
            references=json.loads(row["refs"]),
        )

    # -- techniques (merged knowledge objects) -----------------------------

    def merge_technique(
        self,
        name: str,
        *,
        category: str = "",
        domain: str = "",
        summary: str = "",
        known_issues: str = "",
        confidence: float = 0.5,
        evidence: list[str] | None = None,
        relation: str = "supports",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert one merged technique by name and link supporting artifacts.

        Stub merge policy (Plan 8 owns real merging): non-empty text fields win,
        confidence keeps the max seen. Returns the deterministic technique id.
        """
        tid = technique_id(name)
        now = _now()
        row = self._conn.execute(
            "SELECT * FROM techniques WHERE id = ?", (tid,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO techniques (
                    id, name, category, domain, summary, known_issues,
                    confidence, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    name,
                    category,
                    domain,
                    summary,
                    known_issues,
                    confidence,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
        else:
            merged_meta = json.loads(row["metadata"])
            merged_meta.update(metadata or {})
            self._conn.execute(
                """
                UPDATE techniques SET
                    category = ?, domain = ?, summary = ?, known_issues = ?,
                    confidence = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    category or row["category"],
                    domain or row["domain"],
                    summary or row["summary"],
                    known_issues or row["known_issues"],
                    max(confidence, row["confidence"]),
                    json.dumps(merged_meta),
                    now,
                    tid,
                ),
            )

        for artifact_id in evidence or []:
            self.link_artifact_technique(artifact_id, tid, relation=relation)
        self._conn.commit()
        return tid

    def get_technique(self, tid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM techniques WHERE id = ?", (tid,)).fetchone()
        return dict(row) if row else None

    def link_artifact_technique(
        self, artifact_id: str, tid: str, *, relation: str = "mentions", weight: float = 1.0
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO artifact_techniques (artifact_id, technique_id, relation, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(artifact_id, technique_id) DO UPDATE SET
                relation = excluded.relation, weight = excluded.weight
            """,
            (artifact_id, tid, relation, weight),
        )
        self._conn.commit()

    def artifacts_for_technique(
        self, tid: str, *, type: ResearchArtifactType | str | None = None
    ) -> list[str]:
        """Ids of artifacts linked to a technique, optionally filtered by type.

        Backs the flagship join: technique ↔ papers / experiments.
        """
        if type is not None:
            rows = self._conn.execute(
                """
                SELECT a.id FROM research_artifacts a
                JOIN artifact_techniques at ON at.artifact_id = a.id
                WHERE at.technique_id = ? AND a.type = ?
                ORDER BY a.id
                """,
                (tid, str(type)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT a.id FROM research_artifacts a
                JOIN artifact_techniques at ON at.artifact_id = a.id
                WHERE at.technique_id = ?
                ORDER BY a.id
                """,
                (tid,),
            ).fetchall()
        return [row["id"] for row in rows]

    def techniques_for_artifact(self, artifact_id: str) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT technique_id FROM artifact_techniques
            WHERE artifact_id = ? ORDER BY technique_id
            """,
            (artifact_id,),
        ).fetchall()
        return [row["technique_id"] for row in rows]

    # -- evidence links ----------------------------------------------------

    def add_evidence_link(
        self,
        *,
        target_kind: str,
        target_id: str,
        artifact_id: str | None = None,
        relation: str = "supports",
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO evidence_links (
                artifact_id, target_kind, target_id, relation, weight, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                target_kind,
                target_id,
                relation,
                weight,
                json.dumps(metadata or {}),
                _now(),
            ),
        )
        self._conn.commit()

    # -- introspection -----------------------------------------------------

    def schema_version(self) -> str:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return row["value"] if row else ""

    def count(self, table: str) -> int:
        # table names are internal / not user input
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"])

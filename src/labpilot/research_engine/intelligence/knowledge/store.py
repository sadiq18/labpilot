"""Knowledge Store — the storage system of record (knowledge-system.md §4).

Persists explored intelligence under ``knowledge/<slug>/research/`` with three
distinct layers (``raw/`` → ``extracted/`` → ``knowledge/``) plus a per-competition
``knowledge.db`` SQLite database. This is a **storage API only** — no LLM, no
retrieval ranking (that is Plans 8–9). ``knowledge.db`` wins for joins;
``reports/analyze.json`` stays a projection.
"""

from __future__ import annotations

import hashlib
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
SCHEMA_VERSION = "2"

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


#: Per-entity table, id prefix, and ``research/knowledge/`` bucket. Adding a new
#: entity type is a row here, not new store methods.
_ENTITY_TABLES: dict[str, tuple[str, str, str]] = {
    "technique": ("techniques", "tech", "techniques"),
    "dataset": ("datasets", "ds", "datasets"),
    "architecture": ("architectures", "arch", "architectures"),
    "task": ("tasks", "task", "tasks"),
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def technique_id(name: str) -> str:
    """Deterministic id so the same technique name always merges to one row."""
    slug = _slug(name)
    return f"tech_{slug}" if slug else "tech_unknown"


def entity_id(entity_type: str, name: str) -> str:
    """Deterministic id for any merged knowledge object."""
    table = _ENTITY_TABLES.get(str(entity_type))
    if table is None:
        raise ValueError(f"unknown entity_type {entity_type!r}")
    prefix = table[1]
    slug = _slug(name)
    return f"{prefix}_{slug}" if slug else f"{prefix}_unknown"


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

    # -- Knowledge Hub ingestion receipts ---------------------------------

    @staticmethod
    def artifact_fingerprint(artifact: ResearchArtifact) -> str:
        """Stable digest of the complete normalized Layer-2 artifact."""
        payload = json.dumps(
            artifact.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def pending_artifacts(
        self,
        artifacts: list[ResearchArtifact],
        *,
        signature: str,
    ) -> list[ResearchArtifact]:
        """Artifacts without a matching successful Hub receipt."""
        if not artifacts:
            return []
        ids = [artifact.id for artifact in artifacts]
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT artifact_id, fingerprint, signature
            FROM artifact_ingestions
            WHERE artifact_id IN ({placeholders})
            """,  # noqa: S608 - placeholders contain only bound parameters
            ids,
        ).fetchall()
        receipts = {row["artifact_id"]: row for row in rows}
        return [
            artifact
            for artifact in artifacts
            if (receipt := receipts.get(artifact.id)) is None
            or receipt["signature"] != signature
            or receipt["fingerprint"] != self.artifact_fingerprint(artifact)
        ]

    def mark_artifacts_ingested(
        self,
        artifacts: list[ResearchArtifact],
        *,
        signature: str,
    ) -> int:
        """Record successful Hub processing for persisted artifacts only."""
        existing = self.existing_artifact_ids([artifact.id for artifact in artifacts])
        now = _now()
        rows = [
            (
                artifact.id,
                self.artifact_fingerprint(artifact),
                signature,
                now,
            )
            for artifact in artifacts
            if artifact.id in existing
        ]
        self._conn.executemany(
            """
            INSERT INTO artifact_ingestions (
                artifact_id, fingerprint, signature, ingested_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                signature=excluded.signature,
                ingested_at=excluded.ingested_at
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

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

    # -- generic merged entities (technique / dataset / architecture / task) ---

    def merge_entity(
        self,
        entity_type: str,
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
        """Upsert any merged knowledge object; techniques also get join links.

        Non-technique tables carry only name/domain/summary/metadata today, so
        technique-specific fields are folded into ``metadata`` for them.
        """
        entity_type = str(entity_type)
        if entity_type == "technique":
            return self.merge_technique(
                name,
                category=category,
                domain=domain,
                summary=summary,
                known_issues=known_issues,
                confidence=confidence,
                evidence=evidence,
                relation=relation,
                metadata=metadata,
            )

        table, _prefix, _bucket = self._entity_table(entity_type)
        eid = entity_id(entity_type, name)
        now = _now()
        extra = {"category": category, "confidence": confidence, **(metadata or {})}
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (eid,)  # noqa: S608 - internal table map
        ).fetchone()
        if row is None:
            self._conn.execute(
                f"""
                INSERT INTO {table} (id, name, domain, summary, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,  # noqa: S608 - internal table map
                (eid, name, domain, summary, json.dumps(extra), now, now),
            )
        else:
            merged_meta = json.loads(row["metadata"])
            merged_meta.update(extra)
            self._conn.execute(
                f"""
                UPDATE {table} SET domain = ?, summary = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,  # noqa: S608 - internal table map
                (
                    domain or row["domain"],
                    summary or row["summary"],
                    json.dumps(merged_meta),
                    now,
                    eid,
                ),
            )
        for artifact_id in evidence or []:
            if self._has_evidence_link(entity_type, eid, artifact_id, relation):
                continue
            self.add_evidence_link(
                target_kind=entity_type,
                target_id=eid,
                artifact_id=artifact_id,
                relation=relation,
            )
        self._conn.commit()
        return eid

    def _has_evidence_link(
        self, target_kind: str, target_id: str, artifact_id: str, relation: str
    ) -> bool:
        """Keep re-ingest idempotent (``evidence_links`` has no natural key)."""
        row = self._conn.execute(
            """
            SELECT 1 FROM evidence_links
            WHERE target_kind = ? AND target_id = ? AND artifact_id = ? AND relation = ?
            LIMIT 1
            """,
            (target_kind, target_id, artifact_id, relation),
        ).fetchone()
        return row is not None

    def existing_artifact_ids(self, ids: list[str]) -> set[str]:
        """Subset of ``ids`` present in ``research_artifacts``.

        Join tables have a foreign key on artifacts, so callers must filter out
        artifacts an analyzer chose not to persist before linking evidence.
        """
        if not ids:
            return set()
        unique = sorted(set(ids))
        placeholders = ",".join("?" for _ in unique)
        rows = self._conn.execute(
            f"SELECT id FROM research_artifacts WHERE id IN ({placeholders})",  # noqa: S608 - bound params
            unique,
        ).fetchall()
        return {row["id"] for row in rows}

    def get_entity(self, entity_type: str, eid: str) -> dict[str, Any] | None:
        table, _prefix, _bucket = self._entity_table(str(entity_type))
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (eid,)  # noqa: S608 - internal table map
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _entity_table(entity_type: str) -> tuple[str, str, str]:
        table = _ENTITY_TABLES.get(entity_type)
        if table is None:
            raise ValueError(f"unknown entity_type {entity_type!r}")
        return table

    def write_knowledge_unit(self, entity_type: str, unit_id: str, payload: str) -> Path:
        """Persist a Layer-3 card under ``research/knowledge/<bucket>/``."""
        _table, _prefix, bucket = self._entity_table(str(entity_type))
        directory = self.paths.knowledge_dir / bucket
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_name(unit_id)}.json"
        path.write_text(payload if payload.endswith("\n") else payload + "\n")
        return path

    # -- beliefs (Layer 4) -------------------------------------------------

    def upsert_belief(
        self,
        *,
        belief_id: str,
        technique: str,
        status: str = "suggested",
        effect: str = "unknown",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert one belief row. Status policy is owned by the Knowledge Hub."""
        now = _now()
        existing = self._conn.execute(
            "SELECT created_at FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, competition_slug, technique, effect, status, confidence,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                competition_slug=excluded.competition_slug, technique=excluded.technique,
                effect=excluded.effect, status=excluded.status,
                confidence=excluded.confidence, metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                belief_id,
                self.competition,
                technique,
                effect,
                status,
                confidence,
                json.dumps(metadata or {}),
                created_at,
                now,
            ),
        )
        self._conn.commit()
        return belief_id

    def get_belief(self, belief_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_beliefs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM beliefs WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM beliefs ORDER BY id").fetchall()
        return [dict(row) for row in rows]

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

"""ReflectionStore — durable CRUD for reflection SoR tables.

Owns ``experiment_evidence``, ``belief_updates``, ``lessons``,
``research_claims``, and ``claim_evidence``. Belief/hypothesis row mutation
belongs to BeliefUpdater / HypothesisEvaluator (Plans 4+); this store only
persists reflection artifacts and the belief-update audit trail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.accessor.common import allocate_sequential_id
from labpilot.accessor.common.json_utils import dumps, loads
from labpilot.accessor.sqlite import SqliteClient
from labpilot.research_engine.intelligence.paths import ResearchPaths

_EVIDENCE_PREFIX = "EE"
_LESSON_PREFIX = "L"
_CLAIM_PREFIX = "C"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReflectionStore:
    """CRUD for Research Reflection tables in ``knowledge.db``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self._client = SqliteClient(self.paths.db_path)
        self._conn = self._client.conn

    def close(self) -> None:
        self._client.close()

    # --- experiment_evidence -------------------------------------------------

    def new_evidence_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM experiment_evidence").fetchall()
        return allocate_sequential_id(_EVIDENCE_PREFIX, (row["id"] for row in rows))

    def create_evidence(
        self,
        *,
        execution_id: str | None = None,
        experiment_id: str | None = None,
        plan_id: str | None = None,
        hypothesis_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        config_summary: dict[str, Any] | None = None,
        runtime_summary: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        strength: str = "moderate",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_id = self.new_evidence_id()
        now = _now()
        self._conn.execute(
            """
            INSERT INTO experiment_evidence (
                id, competition_slug, execution_id, experiment_id, plan_id,
                hypothesis_id, metrics, config_summary, runtime_summary,
                comparison, strength, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                self.competition,
                execution_id,
                experiment_id,
                plan_id,
                hypothesis_id,
                dumps(metrics or {}),
                dumps(config_summary or {}),
                dumps(runtime_summary or {}),
                dumps(comparison or {}),
                strength,
                dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self.get_evidence(evidence_id)
        assert row is not None
        return row

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM experiment_evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
        return None if row is None else self._evidence_from_row(row)

    def list_evidence(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM experiment_evidence
            WHERE competition_slug = ?
            ORDER BY created_at ASC
            """,
            (self.competition,),
        ).fetchall()
        return [self._evidence_from_row(row) for row in rows]

    def _evidence_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "competition_slug": row["competition_slug"],
            "execution_id": row["execution_id"],
            "experiment_id": row["experiment_id"],
            "plan_id": row["plan_id"],
            "hypothesis_id": row["hypothesis_id"],
            "metrics": loads(row["metrics"], {}),
            "config_summary": loads(row["config_summary"], {}),
            "runtime_summary": loads(row["runtime_summary"], {}),
            "comparison": loads(row["comparison"], {}),
            "strength": row["strength"],
            "metadata": loads(row["metadata"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- belief_updates (audit) ----------------------------------------------

    def append_belief_update(
        self,
        *,
        belief_id: str,
        prior_confidence: float,
        new_confidence: float,
        prior_status: str = "",
        new_status: str = "",
        reason: str = "",
        execution_id: str | None = None,
        experiment_id: str | None = None,
        evidence_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert an audit row; returns the new ``belief_updates.id``."""
        # FK: belief must exist.
        belief = self._conn.execute(
            "SELECT id FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        if belief is None:
            raise ValueError(f"unknown belief_id: {belief_id}")

        cur = self._conn.execute(
            """
            INSERT INTO belief_updates (
                belief_id, competition_slug, execution_id, experiment_id,
                prior_confidence, new_confidence, prior_status, new_status,
                reason, evidence_id, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief_id,
                self.competition,
                execution_id,
                experiment_id,
                prior_confidence,
                new_confidence,
                prior_status,
                new_status,
                reason,
                evidence_id,
                dumps(metadata or {}),
                _now(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_belief_updates(self, belief_id: str | None = None) -> list[dict[str, Any]]:
        if belief_id is not None:
            rows = self._conn.execute(
                """
                SELECT * FROM belief_updates
                WHERE belief_id = ?
                ORDER BY id ASC
                """,
                (belief_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM belief_updates
                WHERE competition_slug = ?
                ORDER BY id ASC
                """,
                (self.competition,),
            ).fetchall()
        return [self._belief_update_from_row(row) for row in rows]

    def _belief_update_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "belief_id": row["belief_id"],
            "competition_slug": row["competition_slug"],
            "execution_id": row["execution_id"],
            "experiment_id": row["experiment_id"],
            "prior_confidence": row["prior_confidence"],
            "new_confidence": row["new_confidence"],
            "prior_status": row["prior_status"],
            "new_status": row["new_status"],
            "reason": row["reason"],
            "evidence_id": row["evidence_id"],
            "metadata": loads(row["metadata"], {}),
            "created_at": row["created_at"],
        }

    # --- lessons -------------------------------------------------------------

    def new_lesson_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM lessons").fetchall()
        return allocate_sequential_id(_LESSON_PREFIX, (row["id"] for row in rows))

    def create_lesson(
        self,
        summary: str,
        *,
        category: str = "",
        confidence: float = 0.5,
        source_execution: str | None = None,
        competition_slug: str | None = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a lesson. Pass ``competition_slug=None`` for cross-competition."""
        lesson_id = self.new_lesson_id()
        now = _now()
        slug = self.competition if competition_slug == "" else competition_slug
        self._conn.execute(
            """
            INSERT INTO lessons (
                id, competition_slug, summary, category, confidence,
                source_execution, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson_id,
                slug,
                summary,
                category,
                confidence,
                source_execution,
                dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self.get_lesson(lesson_id)
        assert row is not None
        return row

    def get_lesson(self, lesson_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        return None if row is None else self._lesson_from_row(row)

    def _lesson_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "competition_slug": row["competition_slug"],
            "summary": row["summary"],
            "category": row["category"],
            "confidence": row["confidence"],
            "source_execution": row["source_execution"],
            "metadata": loads(row["metadata"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- research_claims -----------------------------------------------------

    def new_claim_id(self) -> str:
        rows = self._conn.execute("SELECT id FROM research_claims").fetchall()
        return allocate_sequential_id(_CLAIM_PREFIX, (row["id"] for row in rows))

    def create_claim(
        self,
        statement: str,
        *,
        status: str = "candidate",
        confidence: float = 0.5,
        technique: str = "",
        effect: str = "",
        promoted_from: str | None = None,
        contradictions: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        claim_id = self.new_claim_id()
        now = _now()
        self._conn.execute(
            """
            INSERT INTO research_claims (
                id, competition_slug, statement, status, confidence,
                technique, effect, promoted_from, contradictions, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id,
                self.competition,
                statement,
                status,
                confidence,
                technique,
                effect,
                promoted_from,
                dumps(contradictions or []),
                dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self.get_claim(claim_id)
        assert row is not None
        return row

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM research_claims WHERE id = ?", (claim_id,)
        ).fetchone()
        return None if row is None else self._claim_from_row(row)

    def list_claims(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM research_claims
            WHERE competition_slug = ?
            ORDER BY created_at ASC
            """,
            (self.competition,),
        ).fetchall()
        return [self._claim_from_row(row) for row in rows]

    def link_claim_evidence(
        self,
        claim_id: str,
        evidence_id: str,
        *,
        relation: str = "supports",
        weight: float = 1.0,
    ) -> None:
        claim = self.get_claim(claim_id)
        if claim is None:
            raise ValueError(f"unknown claim_id: {claim_id}")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO claim_evidence (
                claim_id, evidence_id, relation, weight
            ) VALUES (?, ?, ?, ?)
            """,
            (claim_id, evidence_id, relation, weight),
        )
        self._conn.commit()

    def list_claim_evidence(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM claim_evidence WHERE claim_id = ?
            """,
            (claim_id,),
        ).fetchall()
        return [
            {
                "claim_id": row["claim_id"],
                "evidence_id": row["evidence_id"],
                "relation": row["relation"],
                "weight": row["weight"],
            }
            for row in rows
        ]

    def _claim_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "competition_slug": row["competition_slug"],
            "statement": row["statement"],
            "status": row["status"],
            "confidence": row["confidence"],
            "technique": row["technique"],
            "effect": row["effect"],
            "promoted_from": row["promoted_from"],
            "contradictions": loads(row["contradictions"], []),
            "metadata": loads(row["metadata"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- helpers for tests / BeliefUpdater -----------------------------------

    def ensure_belief(
        self,
        belief_id: str,
        *,
        technique: str = "",
        effect: str = "unknown",
        status: str = "suggested",
        confidence: float = 0.5,
    ) -> None:
        """Insert a belief row if missing (test / bootstrap helper)."""
        existing = self._conn.execute(
            "SELECT id FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        if existing is not None:
            return
        now = _now()
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, competition_slug, technique, effect, status, confidence,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?)
            """,
            (
                belief_id,
                self.competition,
                technique,
                effect,
                status,
                confidence,
                now,
                now,
            ),
        )
        self._conn.commit()

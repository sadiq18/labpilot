"""Read/write adapters for Evidence Cards (``EV-xxx.json``)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.evidence.models import EvidenceCard
from labpilot.research_engine.evidence.store import EvidenceCardStore

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["evidence_card"]


class EvidenceArtifacts:
    """Typed access to Evidence Cards for one competition."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self._store = EvidenceCardStore(knowledge_dir, competition)

    @property
    def store(self) -> EvidenceCardStore:
        """Underlying :class:`EvidenceCardStore` for APIs not mirrored here."""
        return self._store

    def save(
        self,
        card: EvidenceCard,
        *,
        produced_by: str = "compare",
    ) -> tuple[EvidenceCard, ArtifactRef]:
        """Persist a card (assigning an id when needed) and return it with a ref."""
        saved = self._store.save(card)
        _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
        path = self._store.dir / f"{saved.id}.json"
        ref = ArtifactRef(
            kind="evidence_card",
            id=saved.id,
            schema_id=SCHEMA_ID,
            path=str(path),
            competition=self.competition,
        )
        return saved, ref

    def get(self, card_id: str) -> EvidenceCard | None:
        """Return a card by id, or ``None`` if it does not exist."""
        return self._store.get(card_id)

    def get_for_execution(self, execution_id: str) -> EvidenceCard | None:
        """Return the card linked to a treatment execution, if any."""
        return self._store.get_for_execution(execution_id)

    def list(self) -> list[EvidenceCard]:
        """Return all Evidence Cards for this competition."""
        return self._store.list()

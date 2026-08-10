"""Durable Evidence Card JSON under ``research/evidence/EV-xxx.json``."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.accessor.common import allocate_sequential_id
from labpilot.research_engine.evidence.models import EvidenceCard
from labpilot.research_engine.intelligence.paths import ResearchPaths

logger = logging.getLogger(__name__)

_EV_PREFIX = "EV"


class EvidenceCardStore:
    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self.dir = self.paths.root / "evidence"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, card_id: str) -> Path:
        return self.dir / f"{card_id}.json"

    def new_id(self) -> str:
        existing = [p.stem for p in self.dir.glob("EV-*.json")]
        return allocate_sequential_id(_EV_PREFIX, existing)

    def save(self, card: EvidenceCard) -> EvidenceCard:
        if not card.id:
            card = card.model_copy(update={"id": self.new_id()})
        if not card.competition:
            card = card.model_copy(update={"competition": self.competition})
        path = self._path(card.id)
        path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return card

    def get(self, card_id: str) -> EvidenceCard | None:
        path = self._path(card_id)
        if not path.is_file():
            return None
        try:
            return EvidenceCard.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A card that will not parse is not a card that does not exist.
            # Returning `None` for both means a corrupted verdict reads as *no
            # verdict*, and the promoter, the belief updater and the planner all
            # act on that difference. M20, 2026-08-09.
            logger.exception("evidence card at %s could not be read", path)
            return None

    def get_for_execution(self, execution_id: str) -> EvidenceCard | None:
        for path in sorted(self.dir.glob("EV-*.json"), reverse=True):
            try:
                card = EvidenceCard.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                # Skipped, but no longer silently: a corrupt card disappearing
                # from a listing is evidence going missing without a trace.
                logger.exception("evidence card at %s could not be read; skipping", path)
                continue
            if card.treatment_experiment == execution_id:
                return card
        return None

    def get_for_hypothesis(self, hypothesis_id: str) -> EvidenceCard | None:
        for path in sorted(self.dir.glob("EV-*.json"), reverse=True):
            try:
                card = EvidenceCard.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                # Skipped, but no longer silently: a corrupt card disappearing
                # from a listing is evidence going missing without a trace.
                logger.exception("evidence card at %s could not be read; skipping", path)
                continue
            if card.hypothesis_id == hypothesis_id:
                return card
        return None

    def list(self) -> list[EvidenceCard]:
        out: list[EvidenceCard] = []
        for path in sorted(self.dir.glob("EV-*.json")):
            try:
                out.append(
                    EvidenceCard.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                # Skipped, but no longer silently: a corrupt card disappearing
                # from a listing is evidence going missing without a trace.
                logger.exception("evidence card at %s could not be read; skipping", path)
                continue
        return out

"""Durable Evidence Card JSON under ``research/evidence/EV-xxx.json``."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.accessor.common import allocate_sequential_id, atomic_write_text, locked
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
        if not card.competition:
            card = card.model_copy(update={"competition": self.competition})
        if not card.id:
            # Locked across allocate-then-write, not just the allocate (M11):
            # two concurrent saves of new cards must not both glob the same
            # max EV-NNN before either file lands on disk — the id doesn't
            # exist yet, so this locks the directory's allocation slot, not
            # a per-card id the way `HypothesisStore` locks a known one.
            with locked(self.dir / ".alloc.lock"):
                card = card.model_copy(update={"id": self.new_id()})
                self._write(card)
            return card
        self._write(card)
        return card

    def _write(self, card: EvidenceCard) -> None:
        # Atomic (M11), same reason as HypothesisStore._write_json: readers
        # (get/list/get_for_execution/get_for_hypothesis) take no lock, so a
        # truncate-then-write here is a torn-read window they'd hit directly.
        atomic_write_text(self._path(card.id), card.model_dump_json(indent=2) + "\n")

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

    def list(self, *, strict: bool = False) -> list[EvidenceCard]:
        """Every readable card. With `strict`, refuse to answer over a short set.

        Skipping a corrupt card and returning the survivors is right for a
        listing and wrong for a *measurement*: `measured_effect` reports
        "observed N times, net X" over whatever came back, so a card that would
        not parse silently changed the number rather than the answer. Logging it
        was not enough — the promoter's own handler could never fire, because
        the corruption was already swallowed here. Reported on PR #120.

        So the caller says which it is. A reader that only wants the cards it
        can show keeps the default; a caller computing a figure asks for
        `strict` and gets the fault.
        """
        out: list[EvidenceCard] = []
        for path in sorted(self.dir.glob("EV-*.json")):
            try:
                out.append(EvidenceCard.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                logger.exception("evidence card at %s could not be read", path)
                if strict:
                    raise ValueError(
                        f"evidence card at {path} could not be read, so any figure "
                        f"computed over this set would be short by at least one: {exc}"
                    ) from exc
                continue
        return out

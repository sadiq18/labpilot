from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from labpilot.experiments.graph import ExperimentGraph
from labpilot.experiments.models import Experiment, Hypothesis, HypothesisStatus

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^H-(\d+)$")


def _now() -> datetime:
    return datetime.now()


class HypothesisStore:
    """File-backed CRUD for per-competition hypotheses under `knowledge/`."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.hypotheses_dir = self.knowledge_dir / competition / "hypotheses"

    def _path_for(self, hypothesis_id: str) -> Path:
        return self.hypotheses_dir / f"{hypothesis_id}.json"

    def _allocate_id(self) -> str:
        self.hypotheses_dir.mkdir(parents=True, exist_ok=True)
        max_n = 0
        for path in self.hypotheses_dir.glob("H-*.json"):
            match = _ID_PATTERN.match(path.stem)
            if match:
                max_n = max(max_n, int(match.group(1)))
        next_n = max_n + 1
        width = max(3, len(str(next_n)))
        return f"H-{next_n:0{width}d}"

    def create(
        self,
        *,
        observation: str,
        reason: str,
        prediction: str,
        confidence: float,
        tags: Iterable[str] = (),
        source: Literal["manual", "reflection", "llm"] = "manual",
    ) -> Hypothesis:
        now = _now()
        hypothesis = Hypothesis(
            id=self._allocate_id(),
            competition=self.competition,
            observation=observation,
            reason=reason,
            prediction=prediction,
            confidence=confidence,
            tags=list(tags),
            source=source,
            created_at=now,
            updated_at=now,
        )
        self._save(hypothesis)
        return hypothesis

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        path = self._path_for(hypothesis_id)
        if not path.is_file():
            return None
        try:
            return Hypothesis.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return None

    def list(self, *, status: HypothesisStatus | None = None) -> list[Hypothesis]:
        if not self.hypotheses_dir.is_dir():
            return []
        results: list[Hypothesis] = []
        for path in sorted(self.hypotheses_dir.glob("H-*.json")):
            try:
                hypothesis = Hypothesis.model_validate_json(path.read_text())
            except (OSError, ValueError) as exc:
                logger.debug("Skipping unreadable hypothesis %s: %s", path, exc)
                continue
            if status is not None and hypothesis.status != status:
                continue
            results.append(hypothesis)
        return results

    def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        *,
        evidence_run_id: str | None = None,
    ) -> Hypothesis:
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            raise FileNotFoundError(
                f"Hypothesis '{hypothesis_id}' not found for competition "
                f"'{self.competition}' under {self.hypotheses_dir}."
            )

        evidence_for = list(hypothesis.evidence_for)
        evidence_against = list(hypothesis.evidence_against)
        if evidence_run_id:
            if status == HypothesisStatus.CONFIRMED:
                if evidence_run_id not in evidence_for:
                    evidence_for.append(evidence_run_id)
            elif status == HypothesisStatus.REJECTED:
                if evidence_run_id not in evidence_against:
                    evidence_against.append(evidence_run_id)

        updated = hypothesis.model_copy(
            update={
                "status": status,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "updated_at": _now(),
            }
        )
        self._save(updated)
        return updated

    def mark_testing_if_proposed(self, hypothesis_id: str) -> Hypothesis:
        """Flip `proposed` → `testing` when a run attaches this hypothesis.

        Leaves any other status untouched (already testing/confirmed/...).
        """
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            raise FileNotFoundError(
                f"Hypothesis '{hypothesis_id}' not found for competition "
                f"'{self.competition}' under {self.hypotheses_dir}."
            )
        if hypothesis.status != HypothesisStatus.PROPOSED:
            return hypothesis
        return self.update_status(hypothesis_id, HypothesisStatus.TESTING)

    def _save(self, hypothesis: Hypothesis) -> None:
        self.hypotheses_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(hypothesis.id)
        path.write_text(hypothesis.model_dump_json(indent=2))


def linked_experiments(hypothesis_id: str, graph: ExperimentGraph) -> list[Experiment]:
    """Return every experiment in `graph` whose manifest linked this hypothesis."""
    return [
        exp
        for exp in graph.nodes.values()
        if exp.hypothesis_id == hypothesis_id
    ]

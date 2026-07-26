from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from labpilot.research_engine.shared.experiments.graph import ExperimentGraph
from labpilot.research_engine.shared.experiments.models import (
    Experiment,
    Hypothesis,
    HypothesisCreatedBy,
    HypothesisEvidenceRef,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)

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
        expected_impact: float = 0.0,
        tags: Iterable[str] = (),
        source: Literal["manual", "reflection", "llm", "analyze"] = "manual",
        created_by: HypothesisCreatedBy | str | None = None,
        generator: HypothesisGenerator | str | None = None,
        origin: HypothesisOrigin | str | None = None,
        origins: Iterable[HypothesisOrigin | str] = (),
        evidence: Iterable[HypothesisEvidenceRef | dict] = (),
    ) -> Hypothesis:
        now = _now()
        resolved_created_by = _coerce_created_by(created_by, source)
        resolved_generator = _coerce_generator(generator, source)
        resolved_origin = _coerce_origin(origin, source)
        resolved_origins = [
            HypothesisOrigin(str(item)) for item in origins if str(item).strip()
        ]
        if not resolved_origins and resolved_origin is not None:
            resolved_origins = [resolved_origin]
        evidence_refs = [
            item
            if isinstance(item, HypothesisEvidenceRef)
            else HypothesisEvidenceRef.model_validate(item)
            for item in evidence
        ]
        hypothesis = Hypothesis(
            id=self._allocate_id(),
            competition=self.competition,
            observation=observation,
            reason=reason,
            prediction=prediction,
            confidence=confidence,
            expected_impact=expected_impact,
            tags=list(tags),
            source=source,
            created_by=resolved_created_by,
            generator=resolved_generator,
            origin=resolved_origin,
            origins=resolved_origins,
            evidence=evidence_refs,
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
        pending: list[Hypothesis] = []
        for path in sorted(self.hypotheses_dir.glob("H-*.json")):
            try:
                hypothesis = Hypothesis.model_validate_json(path.read_text())
            except (OSError, ValueError) as exc:
                logger.debug("Skipping unreadable hypothesis %s: %s", path, exc)
                continue
            pending.append(hypothesis)
            if status is not None and hypothesis.status != status:
                continue
            results.append(hypothesis)
        # Backfill knowledge.db for file-only hypotheses created before dual-write.
        self._mirror_many_to_db(pending)
        return results

    def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        *,
        evidence_run_id: str | None = None,
        why: str | None = None,
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

        reason = hypothesis.reason
        if why:
            note = f"[reflection] {why.strip()}"
            reason = f"{reason}\n\n{note}".strip() if reason else note

        updated = hypothesis.model_copy(
            update={
                "status": status,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "reason": reason,
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
        self._mirror_to_db(hypothesis)

    def _mirror_to_db(self, hypothesis: Hypothesis) -> None:
        """Dual-write into ``knowledge.db`` hypotheses table (M3 KnowledgeStore)."""
        self._mirror_many_to_db([hypothesis])

    def _mirror_many_to_db(self, hypotheses: list[Hypothesis]) -> None:
        if not hypotheses:
            return
        from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

        with KnowledgeStore(self.knowledge_dir, self.competition) as store:
            for hypothesis in hypotheses:
                metadata = {
                    "tags": list(hypothesis.tags),
                    "source": hypothesis.source,
                    "created_by": (
                        str(hypothesis.created_by)
                        if hypothesis.created_by is not None
                        else None
                    ),
                    "generator": (
                        str(hypothesis.generator)
                        if hypothesis.generator is not None
                        else None
                    ),
                    "origin": (
                        str(hypothesis.origin) if hypothesis.origin is not None else None
                    ),
                    "origins": [str(item) for item in hypothesis.origins],
                    "evidence": [
                        item.model_dump(mode="json") for item in hypothesis.evidence
                    ],
                    "evidence_for": list(hypothesis.evidence_for),
                    "evidence_against": list(hypothesis.evidence_against),
                    "file_created_at": hypothesis.created_at.isoformat(),
                    "file_updated_at": hypothesis.updated_at.isoformat(),
                }
                store.upsert_hypothesis(
                    hypothesis_id=hypothesis.id,
                    observation=hypothesis.observation,
                    prediction=hypothesis.prediction,
                    rationale=hypothesis.reason,
                    expected_impact=hypothesis.expected_impact,
                    confidence=hypothesis.confidence,
                    status=str(hypothesis.status),
                    metadata=metadata,
                )


def _coerce_created_by(
    value: HypothesisCreatedBy | str | None,
    source: str,
) -> HypothesisCreatedBy:
    if value is not None:
        return HypothesisCreatedBy(str(value))
    mapping = {
        "manual": HypothesisCreatedBy.MANUAL,
        "reflection": HypothesisCreatedBy.REFLECTION,
        "llm": HypothesisCreatedBy.MANUAL,
        "analyze": HypothesisCreatedBy.ANALYZE,
    }
    return mapping.get(source, HypothesisCreatedBy.MANUAL)


def _coerce_generator(
    value: HypothesisGenerator | str | None,
    source: str,
) -> HypothesisGenerator:
    if value is not None:
        return HypothesisGenerator(str(value))
    mapping = {
        "manual": HypothesisGenerator.HUMAN,
        "reflection": HypothesisGenerator.LLM,
        "llm": HypothesisGenerator.LLM,
        "analyze": HypothesisGenerator.RULE_ENGINE,
    }
    return mapping.get(source, HypothesisGenerator.HUMAN)


def _coerce_origin(
    value: HypothesisOrigin | str | None,
    source: str,
) -> HypothesisOrigin:
    if value is not None:
        return HypothesisOrigin(str(value))
    mapping = {
        "manual": HypothesisOrigin.USER,
        "reflection": HypothesisOrigin.EXPERIMENT,
        "llm": HypothesisOrigin.USER,
        "analyze": HypothesisOrigin.MIXED,
    }
    return mapping.get(source, HypothesisOrigin.USER)


def linked_experiments(hypothesis_id: str, graph: ExperimentGraph) -> list[Experiment]:
    """Return every experiment in `graph` whose manifest linked this hypothesis."""
    return [
        exp
        for exp in graph.nodes.values()
        if exp.hypothesis_id == hypothesis_id
    ]

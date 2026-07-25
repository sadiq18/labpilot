"""Persist Hypothesis Assistant recommendations into M2 HypothesisStore."""

from __future__ import annotations

from pathlib import Path

from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)
from labpilot.research_engine.intelligence.hypothesis.models import HypothesisRecommendation


def persist_recommendations(
    recommendations: list[HypothesisRecommendation],
    *,
    knowledge_dir: Path,
    competition: str,
) -> list[HypothesisRecommendation]:
    """Create Suggested (proposed) M2 hypotheses; fill hypothesis_id on cards."""
    store = HypothesisStore(knowledge_dir, competition)
    updated: list[HypothesisRecommendation] = []
    for card in recommendations:
        hyp = store.create(
            observation=card.observation or card.title,
            reason=card.reason or card.title,
            prediction=card.prediction,
            confidence=card.confidence,
            tags=card.tags,
            source="analyze",
            created_by=card.created_by,
            generator=card.generator,
            origin=card.origin,
            origins=card.origins,
            evidence=card.supporting_evidence,
        )
        updated.append(card.model_copy(update={"hypothesis_id": hyp.id}))
    return updated


def write_hypotheses_report(
    recommendations: list[HypothesisRecommendation],
    *,
    path: Path,
    notes: list[str] | None = None,
) -> Path:
    """Write ``reports/hypotheses.json`` for the CLI path."""
    from labpilot.research_engine.intelligence.hypothesis.models import (
        HypothesisAssistantResult,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = HypothesisAssistantResult(
        recommendations=recommendations,
        notes=list(notes or []),
    )
    path.write_text(payload.model_dump_json(indent=2) + "\n")
    return path


def default_created_by(cli: bool) -> HypothesisCreatedBy:
    return HypothesisCreatedBy.HYPOTHESIZE if cli else HypothesisCreatedBy.ANALYZE


def default_origin(origins: list[HypothesisOrigin]) -> HypothesisOrigin:
    unique = list(dict.fromkeys(origins))
    if not unique:
        return HypothesisOrigin.MIXED
    if len(unique) == 1:
        return unique[0]
    return HypothesisOrigin.MIXED


def as_generator(used_llm: bool) -> HypothesisGenerator:
    return HypothesisGenerator.LLM if used_llm else HypothesisGenerator.RULE_ENGINE


# Statuses that mean the technique was run or explicitly dispositioned —
# ``proposed`` alone must not block new suggestions.
_TRIED_HYPOTHESIS_STATUSES = frozenset(
    {
        HypothesisStatus.TESTING,
        HypothesisStatus.CONFIRMED,
        HypothesisStatus.REJECTED,
        HypothesisStatus.INCONCLUSIVE,
    }
)

# Statuses that mean an idea is still live in the backlog, so re-generating it
# would only create a duplicate. ``rejected`` / ``inconclusive`` stay eligible
# for a fresh hypothesis when new evidence arrives.
_OPEN_HYPOTHESIS_STATUSES = frozenset(
    {
        HypothesisStatus.PROPOSED,
        HypothesisStatus.TESTING,
        HypothesisStatus.CONFIRMED,
    }
)


def load_existing_technique_tags(knowledge_dir: Path, competition: str) -> set[str]:
    """Techniques already tried — used to avoid duplicate suggestions.

    A hypothesis counts only after a run attaches it (``testing``) or it is
    explicitly marked (``confirmed`` / ``rejected`` / ``inconclusive``).
    ``proposed`` backlog items are not treated as tried.

    Also includes techniques from local experiment artifacts in the knowledge
    store (Plan 11 / README §1 Q5 — subtract already-tried history).
    """
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.intelligence.models import ResearchArtifactType
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    tried: set[str] = set()
    store = HypothesisStore(knowledge_dir, competition)
    for hyp in store.list():
        if hyp.status not in _TRIED_HYPOTHESIS_STATUSES:
            continue
        for tag in hyp.tags:
            tried.add(normalize_label(tag))

    try:
        with KnowledgeStore(knowledge_dir, competition) as kstore:
            for artifact in kstore.list_artifacts(type=ResearchArtifactType.EXPERIMENT):
                for technique in artifact.techniques:
                    tried.add(normalize_label(technique))
    except Exception:
        # Store may be absent on early hypothesize CLI calls — status tags suffice.
        pass
    return tried


def load_open_hypothesis_tags(knowledge_dir: Path, competition: str) -> set[str]:
    """Technique tags already covered by a live hypothesis (proposed/testing/confirmed).

    Generation subtracts these so re-running only produces genuinely new
    hypotheses instead of duplicating the existing backlog.
    """
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    open_tags: set[str] = set()
    for hyp in HypothesisStore(knowledge_dir, competition).list():
        if hyp.status not in _OPEN_HYPOTHESIS_STATUSES:
            continue
        for tag in hyp.tags:
            open_tags.add(normalize_label(tag))
    return open_tags

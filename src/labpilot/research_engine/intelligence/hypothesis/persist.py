"""Persist Hypothesis Assistant recommendations into M2 HypothesisStore."""

from __future__ import annotations

from pathlib import Path

from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
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


def load_existing_technique_tags(knowledge_dir: Path, competition: str) -> set[str]:
    """Techniques already proposed/rejected — used to avoid duplicate suggestions."""
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    store = HypothesisStore(knowledge_dir, competition)
    tried: set[str] = set()
    for hyp in store.list():
        for tag in hyp.tags:
            tried.add(normalize_label(tag))
        # Also fold observation/prediction tokens lightly via tags only.
    return tried

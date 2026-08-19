"""Persist Hypothesis Assistant recommendations into M2 HypothesisStore."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)
from labpilot.research_engine.intelligence.hypothesis.models import HypothesisRecommendation


def _mint_identity(hyp_or_card: Any) -> tuple[str, str, str, str]:
    """What makes two proposals the same idea, for the write-race check only.

    Same technique, same combination, same parent **and the same words**. That
    last clause is what keeps it narrow, and it is not paranoia: one analyze
    batch legitimately carries five cards for `SpecAugment` — from a belief,
    from the untried ledger, from a pipeline diff — and the system's own policy
    says those are distinct proposals worth ranking separately. A
    technique-level identity here silently collapsed ten recommendations to
    five.

    The *policy* filter — which techniques the backlog already covers, across
    proposed/testing/confirmed — runs upstream in `HypothesisAssistant.recommend`
    via `load_open_hypothesis_tags`, and is unchanged. This closes a different
    hole: that filter reads the pool before the LLM drafts, and the rows land
    tens of seconds later, so a second writer can create the same card in
    between. Two writers racing on the same evidence produce the same text, so
    matching on it is enough — and only a *newly created* row can appear in
    that window, which is why the `proposed` snapshot
    `create_unless_covered` passes is the right pool to search.

    Compared on the *stored* technique, via `derive_technique`, not the one
    the card arrived with. A combination proposal carries its members in
    `combo_techniques` and an empty `technique`, which the store fills in on
    write — so comparing the raw fields put `("", "alpha+beta", …)` next to
    `("alphabeta", "alpha+beta", …)` and called them different ideas. Every
    combination card duplicated freely while every other kind was caught.

    Narrow on purpose: a false positive here loses an idea, while a false
    negative leaves one duplicate for the upstream filter to catch next pass.
    """
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
    from labpilot.research_engine.shared.experiments.hypothesis import derive_technique

    combo = [str(t).strip() for t in (getattr(hyp_or_card, "combo_techniques", None) or [])]
    stored_technique = derive_technique(getattr(hyp_or_card, "technique", "") or "", combo)
    return (
        normalize_label(stored_technique or ""),
        "+".join(sorted(normalize_label(t) for t in combo)),
        str(getattr(hyp_or_card, "parent_hypothesis_id", "") or ""),
        " ".join(str(getattr(hyp_or_card, "prediction", "") or "").split()),
    )


def _covers(card: HypothesisRecommendation) -> Callable[[list[Any]], bool]:
    """Predicate for `create_unless_covered` (M16): same card already proposed?"""
    identity = _mint_identity(card)

    def covered(proposed: list[Any]) -> bool:
        return any(_mint_identity(hyp) == identity for hyp in proposed)

    return covered


def persist_recommendations(
    recommendations: list[HypothesisRecommendation],
    *,
    knowledge_dir: Path,
    competition: str,
) -> list[HypothesisRecommendation]:
    """Create Suggested (proposed) M2 hypotheses; fill hypothesis_id on cards.

    Returns only the cards that produced a row. A card another writer created
    in the meantime is dropped rather than returned with no hypothesis, because
    the caller reports `new_count = len(...)` of what comes back — counting a
    card that created nothing is how "23 new hypotheses" gets printed for a run
    that added three.
    """
    store = HypothesisStore(knowledge_dir, competition)
    updated: list[HypothesisRecommendation] = []
    for card in recommendations:
        hyp = store.create_unless_covered(
            covered_by=_covers(card),
            observation=card.observation or card.title,
            reason=card.reason or card.title,
            prediction=card.prediction,
            confidence=card.confidence,
            expected_impact=card.expected_impact_value,
            tags=card.tags,
            source="analyze",
            created_by=card.created_by,
            generator=card.generator,
            origin=card.origin,
            origins=card.origins,
            evidence=card.supporting_evidence,
            technique=card.technique or None,
            parent_hypothesis_id=card.parent_hypothesis_id,
            technique_stack=card.technique_stack,
            combo_techniques=card.combo_techniques,
        )
        if hyp is None:
            continue
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

    _meta = {
        "technique",
        "pipeline_diff",
        "transfer",
        "failure_fix",
        "stacked",
        "combination",
        "ablation",
        "improvement",
        "untried",
        "unused_belief",
        "unused_claim",
        "belief",
        "baseline",
    }
    open_tags: set[str] = set()
    for hyp in HypothesisStore(knowledge_dir, competition).list():
        if hyp.status not in _OPEN_HYPOTHESIS_STATUSES:
            continue
        for tag in hyp.tags:
            if tag.lower() in _meta or tag.lower().startswith("fork:"):
                continue
            open_tags.add(normalize_label(tag))
        if hyp.technique:
            open_tags.add(normalize_label(hyp.technique))
        if hyp.parent_hypothesis_id and hyp.technique:
            open_tags.add(
                normalize_label(f"{hyp.parent_hypothesis_id}+{hyp.technique}")
            )
        combo = [str(t).strip() for t in (hyp.combo_techniques or []) if str(t).strip()]
        if len(combo) >= 2:
            joined = "+".join(sorted(normalize_label(t) for t in combo))
            open_tags.add(joined)
            open_tags.add(
                normalize_label(f"{hyp.parent_hypothesis_id or 'root'}+{joined}")
            )
    return open_tags

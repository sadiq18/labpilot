"""SQL-backed Research Graph queries for planner / retrieve / hypothesize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label


def query_techniques(
    *,
    knowledge_dir: Path,
    competition: str,
    min_confidence: float | None = None,
    reusable_for: list[str] | None = None,
    max_train_time_s: float | None = None,
    min_cv_gain: float | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return techniques with local evidence matching filters.

    Joins beliefs + Evidence Cards (attribution / observed metrics). Prefer
    graph hits over paper-only recall when local evidence exists.
    """
    cards = EvidenceCardStore(knowledge_dir, competition).list()
    # Aggregate per technique from cards.
    stats: dict[str, dict[str, Any]] = {}
    for card in cards:
        domains = {d.lower() for d in card.reusable_for}
        for tech, credit in card.technique_attribution.items():
            label = normalize_label(tech) or tech
            row = stats.setdefault(
                label,
                {
                    "technique": tech,
                    "label": label,
                    "cv_gains": [],
                    "train_times": [],
                    "reusable_for": set(),
                    "n_cards": 0,
                    "accepted": 0,
                },
            )
            row["n_cards"] += 1
            if card.observed.cv_gain is not None:
                row["cv_gains"].append(float(credit if credit else card.observed.cv_gain))
            if card.observed.train_time_s is not None:
                row["train_times"].append(float(card.observed.train_time_s))
            row["reusable_for"] |= domains
            if card.decision.value == "accepted":
                row["accepted"] += 1

    beliefs: dict[str, float] = {}
    with KnowledgeStore(knowledge_dir, competition) as store:
        for belief in store.list_beliefs():
            tech = str(belief.get("technique") or "")
            if tech:
                beliefs[normalize_label(tech) or tech] = float(
                    belief.get("confidence") or 0.5
                )

    want_domains = {d.lower() for d in (reusable_for or []) if d}
    results: list[dict[str, Any]] = []
    for label, row in stats.items():
        conf = beliefs.get(label, 0.5)
        gains = row["cv_gains"]
        mean_gain = sum(gains) / len(gains) if gains else None
        times = row["train_times"]
        median_time = sorted(times)[len(times) // 2] if times else None
        domains = sorted(row["reusable_for"])

        if min_confidence is not None and conf < min_confidence:
            continue
        if want_domains and not (want_domains & set(domains)):
            continue
        if max_train_time_s is not None and median_time is not None:
            if median_time > max_train_time_s:
                continue
        if min_cv_gain is not None and mean_gain is not None:
            if mean_gain < min_cv_gain:
                continue
        if min_cv_gain is not None and mean_gain is None:
            continue

        results.append(
            {
                "technique": row["technique"],
                "confidence": conf,
                "mean_cv_gain": mean_gain,
                "median_train_time_s": median_time,
                "reusable_for": domains,
                "n_evidence_cards": row["n_cards"],
                "n_accepted": row["accepted"],
            }
        )

    results.sort(
        key=lambda r: (
            -(r["confidence"] or 0),
            -(r["mean_cv_gain"] or 0),
            r["technique"].lower(),
        )
    )
    return results[: max(0, limit)]

"""Merge multi-source ExperienceFacet hits.

Source priority is a **heuristic**, not empirically calibrated: prefer declared
competition metadata and concrete code/dataset artifacts over free-text rules
and narrative results. Revisit if seed/inspect shows systematic mis-ranking.

Confidence uses **max** across sources (recall-oriented Stage 2): if any source
is confident, surface the facet. Disagreement penalties (e.g. noisy-OR) wait
until histogram data justifies them.
"""

from __future__ import annotations

from labpilot.research_engine.memory.models import ExperienceFacet, FacetSource

# Higher = preferred when picking canonical ``source`` on ties / merge labels.
# Order: explicit metadata → durable artifacts → narrative → keyword rules.
_SOURCE_PRIORITY: dict[str, int] = {
    "metadata": 60,
    "code": 50,
    "dataset": 40,
    "paper": 30,
    "result": 20,
    "rules": 10,
    "legacy": 0,
}

_MAX_EVIDENCE = 8
# Cap per hit so one high-confidence source cannot crowd out others' evidence.
_MAX_EVIDENCE_PER_HIT = 3


def merge_facet_hits(hits: list[ExperienceFacet]) -> list[ExperienceFacet]:
    """Aggregate by facet name: max confidence, union evidence, best source."""
    by_name: dict[str, list[ExperienceFacet]] = {}
    for hit in hits:
        key = hit.facet.strip().lower()
        if not key:
            continue
        by_name.setdefault(key, []).append(hit)

    merged: list[ExperienceFacet] = []
    for canonical, group in by_name.items():
        merged.append(_merge_group(canonical, group))
    return sorted(merged, key=lambda f: (-f.confidence, f.facet))


def _merge_group(canonical: str, group: list[ExperienceFacet]) -> ExperienceFacet:
    evidence: list[str] = []
    seen: set[str] = set()
    # High confidence first, then source priority — fill remaining slots from
    # lower hits so critical low-conf evidence is not always dropped.
    ordered = sorted(
        group,
        key=lambda h: (-h.confidence, -_SOURCE_PRIORITY.get(h.source, 0)),
    )
    for hit in ordered:
        taken = 0
        for item in hit.evidence:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
            taken += 1
            if taken >= _MAX_EVIDENCE_PER_HIT or len(evidence) >= _MAX_EVIDENCE:
                break
        if len(evidence) >= _MAX_EVIDENCE:
            break

    best_source: FacetSource = max(
        group,
        key=lambda h: (_SOURCE_PRIORITY.get(h.source, 0), h.confidence),
    ).source
    return ExperienceFacet(
        facet=canonical,
        confidence=max(h.confidence for h in group),
        evidence=evidence,
        source=best_source,
    )

"""Merge multi-source ExperienceFacet hits."""

from __future__ import annotations

from labpilot.research_engine.memory.models import ExperienceFacet, FacetSource

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


def merge_facet_hits(hits: list[ExperienceFacet]) -> list[ExperienceFacet]:
    """Aggregate by facet name: max confidence, union evidence, best source."""
    by_name: dict[str, list[ExperienceFacet]] = {}
    for hit in hits:
        key = hit.facet.strip().lower()
        if not key:
            continue
        by_name.setdefault(key, []).append(hit)

    merged: list[ExperienceFacet] = []
    for group in by_name.values():
        merged.append(_merge_group(group))
    return sorted(merged, key=lambda f: (-f.confidence, f.facet))


def _merge_group(group: list[ExperienceFacet]) -> ExperienceFacet:
    evidence: list[str] = []
    seen: set[str] = set()
    ordered = sorted(
        group,
        key=lambda h: (-h.confidence, -_SOURCE_PRIORITY.get(h.source, 0)),
    )
    for hit in ordered:
        for item in hit.evidence:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
            if len(evidence) >= _MAX_EVIDENCE:
                break
        if len(evidence) >= _MAX_EVIDENCE:
            break

    best_source: FacetSource = max(
        group,
        key=lambda h: (_SOURCE_PRIORITY.get(h.source, 0), h.confidence),
    ).source
    return ExperienceFacet(
        facet=ordered[0].facet,
        confidence=max(h.confidence for h in group),
        evidence=evidence,
        source=best_source,
    )

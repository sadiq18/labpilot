"""Step 2: planner candidates respect vocabulary status (design §8.1)."""

from __future__ import annotations

from labpilot.research_engine.intelligence.hypothesis.candidates import (
    filter_by_technique_status,
    filter_incompatible_techniques,
    generate_candidates,
)
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryType,
    ResearchContext,
    RetrievalIntent,
)


class _Candidate:
    def __init__(self, technique: str) -> None:
        self.technique = technique
        self.title = technique
        self.key = technique


def test_junk_techniques_do_not_reach_generate_candidates() -> None:
    context = ResearchContext(
        techniques=[
            {"name": "the", "confidence": 0.9, "status": "dormant"},
            {"name": "Breath Focus practice", "confidence": 0.8, "status": "dormant"},
            {"name": "3D garment modeling", "confidence": 0.7, "status": "dormant"},
            {"name": "SWA", "confidence": 0.6, "status": "confirmed"},
            {"name": "rolling_features", "confidence": 0.5, "status": "rejected"},
        ],
        intent=RetrievalIntent(query_type=QueryType.HYPOTHESIS_GENERATION),
    )
    statuses = {
        normalize_label(row["name"]): row["status"] for row in context.techniques
    }
    candidates = generate_candidates(context, technique_statuses=statuses)
    offered = {c.technique for c in candidates if c.technique}
    assert "the" not in offered
    assert "Breath Focus practice" not in offered
    assert "3D garment modeling" not in offered
    assert "rolling_features" not in offered
    assert any(c.technique == "SWA" for c in candidates)


def test_swa_survives_status_filter() -> None:
    kept, dropped = filter_by_technique_status(
        [_Candidate("SWA"), _Candidate("the")],
        {normalize_label("SWA"): "confirmed", normalize_label("the"): "dormant"},
    )
    assert [c.technique for c in kept] == ["SWA"]
    assert dropped == ["the"]


def test_modality_filter_still_has_a_production_call_site() -> None:
    """Design fold: status is the vocab gate; modality stays wired for candidates."""
    context = ResearchContext(
        techniques=[
            {"name": "vit", "confidence": 0.7, "status": "candidate"},
            {"name": "rolling_features", "confidence": 0.6, "status": "candidate"},
        ],
        intent=RetrievalIntent(query_type=QueryType.HYPOTHESIS_GENERATION),
    )
    statuses = {
        normalize_label("vit"): "candidate",
        normalize_label("rolling_features"): "candidate",
    }
    candidates = generate_candidates(
        context,
        technique_statuses=statuses,
        problem_type="tabular_regression",
    )
    offered = {c.technique for c in candidates if c.technique}
    assert "vit" not in offered
    assert "rolling_features" in offered


def test_confirmed_is_not_hard_blocked_by_modality() -> None:
    """Measured techniques keep a justification note instead of disappearing."""
    context = ResearchContext(
        techniques=[{"name": "vit", "confidence": 0.9, "status": "confirmed"}],
        intent=RetrievalIntent(query_type=QueryType.HYPOTHESIS_GENERATION),
    )
    candidates = generate_candidates(
        context,
        technique_statuses={normalize_label("vit"): "confirmed"},
        problem_type="tabular_regression",
    )
    assert any(c.technique == "vit" for c in candidates)
    assert any("Applicability note" in (c.reason or "") for c in candidates)


def test_filter_incompatible_techniques_still_drops_vision_on_tabular() -> None:
    kept, dropped = filter_incompatible_techniques(
        [_Candidate("vit"), _Candidate("SWA")],
        "tabular_regression",
    )
    assert dropped == ["vit"]
    assert [c.technique for c in kept] == ["SWA"]

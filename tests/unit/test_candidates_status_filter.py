"""Step 2: planner candidates respect vocabulary status (design §8.1)."""

from __future__ import annotations

from labpilot.research_engine.intelligence.hypothesis.candidates import (
    generate_candidates,
    filter_by_technique_status,
)
from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidateKind
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext, RetrievalIntent
from labpilot.research_engine.intelligence.retrieval.models import QueryType


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
        normalize_label(row["name"]): row["status"]
        for row in context.techniques
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


def test_vit_is_not_dropped_on_tabular_by_modality_anymore() -> None:
    """Design §6: applicability is a candidate justification, not a hard block."""
    context = ResearchContext(
        techniques=[{"name": "vit", "confidence": 0.7, "status": "candidate"}],
        intent=RetrievalIntent(
            query_type=QueryType.HYPOTHESIS_GENERATION,
            current_pipeline=[],
        ),
    )
    candidates = generate_candidates(
        context,
        technique_statuses={normalize_label("vit"): "candidate"},
        problem_type="tabular_regression",
    )
    assert any(c.technique == "vit" for c in candidates)

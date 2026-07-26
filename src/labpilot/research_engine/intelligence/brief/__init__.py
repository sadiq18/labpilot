"""Research Brief — durable briefing document before experimentation."""

from __future__ import annotations

from labpilot.research_engine.intelligence.brief.context import render_competition_context
from labpilot.research_engine.intelligence.brief.models import (
    ResearchBrief,
    ResearchBriefNarrative,
)

__all__ = [
    "ResearchBrief",
    "ResearchBriefNarrative",
    "build_research_brief",
    "render_competition_context",
]


def build_research_brief(*args, **kwargs):
    from labpilot.research_engine.intelligence.brief.builder import (
        build_research_brief as _build,
    )

    return _build(*args, **kwargs)

"""Research Brief — durable briefing document before experimentation."""

from __future__ import annotations

from labpilot.research_engine.intelligence.brief.models import ResearchBrief

__all__ = ["ResearchBrief", "build_research_brief"]


def build_research_brief(*args, **kwargs):
    from labpilot.research_engine.intelligence.brief.builder import (
        build_research_brief as _build,
    )

    return _build(*args, **kwargs)

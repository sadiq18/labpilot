"""Handler package — in-process wrappers over existing libraries."""

from __future__ import annotations

from labpilot.research_engine.tools.handlers.analyze import analyze_competition
from labpilot.research_engine.tools.handlers.memory import query_memory
from labpilot.research_engine.tools.handlers.papers import search_papers
from labpilot.research_engine.tools.handlers.plan import generate_plan
from labpilot.research_engine.tools.handlers.reflect import reflect
from labpilot.research_engine.tools.handlers.run import run_plan
from labpilot.research_engine.tools.handlers.submit import submit, submit_learn

__all__ = [
    "analyze_competition",
    "generate_plan",
    "query_memory",
    "reflect",
    "run_plan",
    "search_papers",
    "submit",
    "submit_learn",
]

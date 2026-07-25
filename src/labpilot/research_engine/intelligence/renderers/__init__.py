"""Presentation layer — consume an ``AnalysisReport``; never called by analyzers.

Milestone 3 v1 ships terminal + JSON only (no HTML; design §12.5).
"""

from labpilot.research_engine.intelligence.renderers.json import (
    PUBLIC_TOP_LEVEL_KEYS,
    assert_public_contract,
    to_json,
    validate_json,
    write_report,
)
from labpilot.research_engine.intelligence.renderers.terminal import (
    render_terminal,
    render_terminal_text,
)

__all__ = [
    "PUBLIC_TOP_LEVEL_KEYS",
    "assert_public_contract",
    "render_terminal",
    "render_terminal_text",
    "to_json",
    "validate_json",
    "write_report",
]

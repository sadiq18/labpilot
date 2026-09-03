"""The competition corpus — M24.

Understanding is measured, not asserted. A captured competition, an expander
that rebuilds the shape it came from, and a scorer that runs the shipped path
against it and answers per criterion.

Plan: ``docs/research-os/autonomy-roadmap/19-competition-benchmark.md``
"""

from labpilot.accessor.benchmark.capture import capture_competition
from labpilot.accessor.benchmark.expand import expand_fixture
from labpilot.accessor.benchmark.fixture import (
    CapturedFile,
    CompetitionFixture,
    Expectations,
    load_fixture,
    save_fixture,
)
from labpilot.accessor.benchmark.score import CriterionResult, Scorecard, score_fixture

__all__ = [
    "CapturedFile",
    "CompetitionFixture",
    "CriterionResult",
    "Expectations",
    "Scorecard",
    "capture_competition",
    "expand_fixture",
    "load_fixture",
    "save_fixture",
    "score_fixture",
]

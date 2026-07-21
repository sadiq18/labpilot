"""Content-type analyzers (plugins).

Real analyzers (competition / papers / repositories / experiments / dataset /
discussions) land in Plans 4–7 and F. Plan 1 ships only the ``Analyzer``
Protocol and a ``BaseAnalyzer`` helper.
"""

from labpilot.research_engine.intelligence.analyzers.base import Analyzer, BaseAnalyzer

__all__ = ["Analyzer", "BaseAnalyzer"]

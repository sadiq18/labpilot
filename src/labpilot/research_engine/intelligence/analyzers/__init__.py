"""Content-type analyzers (plugins).

Plan 1 shipped the ``Analyzer`` Protocol + ``BaseAnalyzer`` helper. Plan 4 adds
the local-only ExperimentAnalyzer + DatasetAnalyzer; Plan 5 adds
CompetitionAnalyzer. Paper / repository / discussion analyzers land in Plans
6–7 and F.

The concrete analyzers are intentionally **not** imported here — they pull the
execution stack / pandas / Kaggle clients. Import them from their submodules
(as ``build_default_registry`` does) to keep ``import ...analyzers`` cheap.
"""

from labpilot.research_engine.intelligence.analyzers.base import Analyzer, BaseAnalyzer

__all__ = ["Analyzer", "BaseAnalyzer"]

"""Analyzer registry + selection (design §3.3).

The CLI is a thin view over this registry: bare ``research analyze <slug>``
runs every ``default_enabled`` analyzer; ``--include`` / ``--exclude`` and a
single positional analyzer name narrow the set.
"""

from __future__ import annotations

from labpilot.research_engine.intelligence.analyzers.base import Analyzer


class UnknownAnalyzerError(KeyError):
    """Raised when a requested analyzer name is not registered."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        listing = ", ".join(available) if available else "(none registered)"
        super().__init__(f"Unknown analyzer '{name}'. Available: {listing}")


class AnalyzerRegistry:
    """Ordered collection of analyzers keyed by their stable ``name``."""

    def __init__(self) -> None:
        self._analyzers: dict[str, Analyzer] = {}

    def register(self, analyzer: Analyzer) -> None:
        name = analyzer.name
        if not name:
            raise ValueError("Analyzer.name must be a non-empty stable id.")
        if name in self._analyzers:
            raise ValueError(f"Analyzer '{name}' is already registered.")
        self._analyzers[name] = analyzer

    def get(self, name: str) -> Analyzer:
        try:
            return self._analyzers[name]
        except KeyError as exc:
            raise UnknownAnalyzerError(name, self.names()) from exc

    def list(self) -> list[Analyzer]:
        return list(self._analyzers.values())

    def names(self) -> list[str]:
        return list(self._analyzers)

    def select(
        self,
        *,
        only: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[Analyzer]:
        """Resolve which analyzers to run.

        - ``only``: run exactly this one analyzer (regardless of default_enabled).
        - ``include``: restrict to this set (must all be registered).
        - ``exclude``: drop these from the default set.
        - default: every analyzer with ``default_enabled=True``.

        ``only`` is mutually exclusive with ``include`` / ``exclude``.
        """
        if only is not None:
            if include or exclude:
                raise ValueError(
                    "A single analyzer argument cannot be combined with "
                    "--include/--exclude."
                )
            return [self.get(only)]

        if include:
            unknown = include - set(self._analyzers)
            if unknown:
                raise UnknownAnalyzerError(sorted(unknown)[0], self.names())
            return [a for a in self._analyzers.values() if a.name in include]

        exclude = exclude or set()
        unknown = exclude - set(self._analyzers)
        if unknown:
            raise UnknownAnalyzerError(sorted(unknown)[0], self.names())
        return [
            a
            for a in self._analyzers.values()
            if a.default_enabled and a.name not in exclude
        ]


def build_default_registry() -> AnalyzerRegistry:
    """Registry wired with the built-in analyzers available so far.

    Plan 4: local ExperimentAnalyzer + DatasetAnalyzer.
    Plan 5: CompetitionAnalyzer (Kaggle-expert brief; Null winning solutions).
    Plan 6: PaperAnalyzer (LiteratureProvider + PaperKnowledge).
    Plan 7: RepositoryAnalyzer (GitHub collect + extract + local diff).
    Discussion analyzers append in Plan F.
    """
    # Imported here (not at module top) so importing the registry stays cheap
    # and does not pull pandas / the execution stack unless a registry is built.
    from labpilot.research_engine.intelligence.analyzers.competition import CompetitionAnalyzer
    from labpilot.research_engine.intelligence.analyzers.dataset import DatasetAnalyzer
    from labpilot.research_engine.intelligence.analyzers.experiments import ExperimentAnalyzer
    from labpilot.research_engine.intelligence.analyzers.papers import PaperAnalyzer
    from labpilot.research_engine.intelligence.analyzers.repositories import (
        RepositoryAnalyzer,
    )

    registry = AnalyzerRegistry()
    registry.register(CompetitionAnalyzer())
    registry.register(ExperimentAnalyzer())
    registry.register(DatasetAnalyzer())
    registry.register(PaperAnalyzer())
    registry.register(RepositoryAnalyzer())
    return registry

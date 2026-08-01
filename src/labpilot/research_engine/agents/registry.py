"""Specialist registry — capability routing for Conductor."""

from __future__ import annotations

from labpilot.research_engine.agents.models import SpecialistDescriptor
from labpilot.research_engine.context.models import ContextBundle


class SpecialistRegistry:
    """Register specialists and select candidates by capability (+ budget)."""

    def __init__(self) -> None:
        self._by_name: dict[str, SpecialistDescriptor] = {}

    def register(self, descriptor: SpecialistDescriptor) -> None:
        """Add or replace a specialist under ``descriptor.name``."""
        self._by_name[descriptor.name] = descriptor

    def get(self, name: str) -> SpecialistDescriptor | None:
        """Return a descriptor by name, or ``None``."""
        return self._by_name.get(name)

    def require(self, name: str) -> SpecialistDescriptor:
        """Return a descriptor or raise :class:`KeyError`."""
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"no specialist registered: {name}")
        return spec

    def list_specialists(self) -> list[SpecialistDescriptor]:
        """Return registered specialists sorted by name."""
        return [self._by_name[k] for k in sorted(self._by_name)]

    def names(self) -> list[str]:
        """Return sorted specialist names."""
        return sorted(self._by_name)

    def candidates(
        self,
        *,
        capability: str,
        budget: float | None = None,
        context: ContextBundle | None = None,
    ) -> list[SpecialistDescriptor]:
        """Return specialists advertising ``capability``, cheapest first.

        ``budget`` filters by ``cost_hint`` when both are set.
        ``context`` is accepted for Conductor routing hooks (unused for now).
        """
        del context  # reserved for context-aware scoring
        matches: list[SpecialistDescriptor] = []
        for spec in self._by_name.values():
            if capability not in spec.capabilities:
                continue
            if budget is not None and spec.cost_hint is not None and spec.cost_hint > budget:
                continue
            matches.append(spec)

        def _sort_key(s: SpecialistDescriptor) -> tuple[float, str]:
            cost = s.cost_hint if s.cost_hint is not None else float("inf")
            return (cost, s.name)

        return sorted(matches, key=_sort_key)

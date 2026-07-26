"""Legacy per-run reflection writer (quarantined).

Research Engineer Reporting capability + ``ReflectionGeneratorAgent`` are the
SoR for new executions. This module remains for inspecting historical
``runs/*/reflection.md`` artifacts and the ``research report`` HTML path.
"""

from labpilot.reflection.generator import ReflectionGenerator

__all__ = ["ReflectionGenerator"]

"""Techniques the templates can execute deterministically.

The recipe-backed subset only — see `registry` for why this is deliberately
*not* a vocabulary, and design §8.7 for the split between description (what
codegen consumes) and identity (what the ledger needs).
"""

from labpilot.research_engine.execution.technique.models import TechniqueSpec
from labpilot.research_engine.execution.technique.registry import (
    EXECUTABLE_TECHNIQUES,
    canonical_name,
    executable_names,
    get_technique,
)

__all__ = [
    "EXECUTABLE_TECHNIQUES",
    "TechniqueSpec",
    "canonical_name",
    "get_technique",
    "executable_names",
]

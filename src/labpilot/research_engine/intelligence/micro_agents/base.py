"""Micro Agent contract for the Research Intelligence platform.

The implementation lives in :mod:`labpilot.common.micro_agents` so the
Execution platform can share it without importing ``intelligence``. This module
re-exports it under the design §11 path (``intelligence/micro_agents/base.py``)
so callers can import from the package that owns the agents.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, MicroAgent, StructuredContext

__all__ = ["BaseMicroAgent", "MicroAgent", "StructuredContext"]

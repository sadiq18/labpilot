"""Micro Agent contract for the Execution platform.

Re-exports the shared contract from :mod:`labpilot.accessor.common.micro_agents` so
``execution`` never imports ``intelligence`` (import-hygiene rule) while still
sharing one ``MicroAgent`` Protocol / ``BaseMicroAgent`` implementation.
"""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, MicroAgent, StructuredContext

__all__ = ["BaseMicroAgent", "MicroAgent", "StructuredContext"]

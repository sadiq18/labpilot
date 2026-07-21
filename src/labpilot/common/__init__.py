"""Shared building blocks usable by both the Research Intelligence and
Execution platforms.

Import hygiene: ``common`` may import stdlib / third-party and other
``common`` modules only. It must never import ``research_engine`` or ``cli``,
so both platforms can depend on it without creating a cycle.
"""

"""Execution Platform slice of the research engine (M2 run/reflect, etc.).

Import hygiene: ``execution`` may import ``common`` but must never import
``research_engine.intelligence`` or ``cli``. Shared Micro Agent contracts
therefore live in :mod:`labpilot.common.micro_agents`.
"""

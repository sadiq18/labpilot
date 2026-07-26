"""Shared data-access layer.

``labpilot.accessor`` owns the low-level infrastructure every research pillar
needs — the SQLite client + unified ``schema.sql`` + migrator, Kaggle API client
(``accessor.kaggle``), dataset download (``accessor.data``), profiling
(``accessor.profiler``), and shared helpers (``accessor.common``) — so
``intelligence`` / ``planner`` / ``execution`` share infrastructure without
importing one another.

The LLM layer lives under ``labpilot.llm`` (not accessor).

Import rule: ``accessor`` never imports a pillar (no ``intelligence`` /
``planner`` / ``execution`` imports here). Raw API DTOs such as
``CompetitionMetadata`` live under ``accessor.kaggle``; richer
``CompetitionSpec`` lives under ``intelligence.competition``.
"""

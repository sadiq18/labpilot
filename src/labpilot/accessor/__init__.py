"""Shared data-access layer.

``labpilot.accessor`` owns the low-level infrastructure every research pillar
needs — the SQLite client + unified ``schema.sql`` + migrator, the LLM client,
Kaggle API client (``accessor.kaggle``), and small commons helpers — so
``intelligence`` / ``planner`` / ``execution`` share infrastructure without
importing one another.

Import rule: ``accessor`` never imports a pillar (no ``intelligence`` /
``planner`` / ``execution`` imports here). Raw API DTOs such as
``CompetitionMetadata`` live under ``accessor.kaggle``; richer
``CompetitionSpec`` lives under ``intelligence.competition``.
"""

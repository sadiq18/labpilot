"""Shared data-access layer.

``labpilot.accessor`` owns the low-level infrastructure every research pillar
needs — the SQLite client + unified ``schema.sql`` + migrator, the LLM client,
and small commons helpers — so ``intelligence`` / ``planner`` / ``execution``
share infrastructure without importing one another.

Import rule: ``accessor`` never imports a pillar (no ``intelligence`` /
``planner`` / ``execution`` imports here).
"""

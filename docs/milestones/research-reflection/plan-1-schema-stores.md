# Plan 1 — Schema + ReflectionStore

Back to [Research Reflection](README.md). Design: [schema.md](schema.md).

**Status:** Done. **Depends on:** Design Phase A. **Unlocks:** Plans 2–10.

---

## Goal

Add reflection DDL and a thin `ReflectionStore` for CRUD — before extractors or
critic logic.

## In scope

- Tables: `experiment_evidence`, `belief_updates`, `lessons`, `research_claims`,
  `claim_evidence` in `accessor/sqlite/schema.sql`
- Bump `SCHEMA_VERSION` (4 → 5)
- `research_engine/reflection/store.py` — create/get evidence, append belief_update,
  CRUD stubs for lessons/claims
- Unit tests: migrate; insert evidence; append belief_update; claim + claim_evidence

## Out of scope

- EvidenceExtractor / Critic / BeliefUpdater logic
- CLI
- Engineer Reporting wiring

## Acceptance criteria

- [x] Fresh and migrated DBs at SCHEMA_VERSION 5
- [x] Store round-trips evidence + belief_update
- [x] Existing belief/hypothesis/experiment tables unchanged in meaning

# Plan 1 — Schema, evidence, and execution records

Back to [Research Engineer](README.md). Design: [schema.md](schema.md) ·
[architecture.md](architecture.md) §6–8.

**Status:** Not started. **Depends on:** Design Phase A approved; Research Planner MVP.
**Unlocks:** Plans 2–11.

---

## Goal

Stand up durable execution persistence: `research_executions` (`E-xxx`), task timing/error
hooks on `research_tasks` (columns or metadata), evidence file convention, and mapping into
reused DB `experiments` — without building the Engineer controller yet.

## Why this matters

Resume, multi-attempt runs, and capstone evidence need a SoR before any capability runs.

## In scope

- DDL in `accessor/sqlite/schema.sql` + `SCHEMA_VERSION` bump
- `research_executions` table (see [schema.md](schema.md) §3)
- Additive `research_tasks` fields if needed (`started_at` / `completed_at` / `error` — prefer
  metadata first if adequate)
- `ExecutionStore` sketch under `research_engine/execution/store.py` (CRUD for executions;
  task status updates via PlanStore or shared client)
- On-disk layout: `…/executions/E-xxx/evidence/<task_id>.json` (or agreed path)
- Unit tests: migrate existing DB; create execution; cascade rules

## Out of scope

- Engineer controller / capability registry (Plan 2)
- Baseline plan compiler (Plan 3)
- Any capability implementation
- CLI `research run`

## Acceptance criteria

- Migrating `knowledge.db` adds `research_executions` without data loss
- Create/get/list executions by `plan_id`; status transitions persist
- Layer-3 `tasks` untouched

## Test plan

- Unit: migrate + round-trip execution row
- Unit: link execution → plan; reject unknown plan_id (FK)

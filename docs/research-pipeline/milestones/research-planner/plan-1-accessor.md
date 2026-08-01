# Plan 1 — Accessor layer (SQLite + commons)

Back to [Research Planner](README.md). Design: [package-layout.md](package-layout.md) §2–3 ·
[schema.md](schema.md) §2.

**Status:** Not started. **Depends on:** Research Intelligence Phase 1 shipped.
**Unlocks:** Plans 2–6 (planner must not import `intelligence` for infrastructure).

---

## Goal

Introduce `labpilot.accessor` as the shared data-access layer: SQLite client + unified
`schema.sql` + migrator and small commons helpers. Migrate existing ownership
out of `intelligence/knowledge/`. LLM remains under `labpilot.llm` so planner /
intelligence / execution share infrastructure without pillar-to-pillar imports.

## Why this matters

The planner needs SQLite and an LLM client but must not import `intelligence`. Today the
SoR schema and connection live under Knowledge Store; the LLM client lives under
`labpilot/llm/`. Without the accessor refactor, every planner PR either duplicates SQLite
infra or violates import hygiene. LLM stays in `labpilot.llm` (no `accessor.llm` facade).

## In scope

- Package tree:

```
src/labpilot/accessor/
  sqlite/     # client.py, schema.sql (moved), migrate.py
  commons/    # ids.py, json_utils (JSON-in-TEXT helpers) — not schema.sql
```

- Move current `intelligence/knowledge/schema.sql` → `accessor/sqlite/schema.sql`
- Extract connection/PRAGMA/row-factory into `SqliteClient`; KnowledgeStore uses it
- LLM stays under `labpilot.llm` (do **not** add an `accessor.llm` facade — callers import
  `labpilot.llm` directly)
- Idempotent `migrate.py` + `SCHEMA_VERSION` (no new tables yet — Plan 2 adds `research_*`)
- Docs: update [package-layout.md](package-layout.md) status note; ARCHITECTURE pointer

## Out of scope

- `research_plans` / `research_tasks` tables (Plan 2)
- `research_engine/planner/` package (Plan 2+)
- CLI `research plan` (Plan 5)
- Behavior changes to analyze / hypothesize

## Design summary

- **One** `schema.sql`, **one** migrator, under `accessor/sqlite/` (not `commons/`).
- `commons/` = id allocators + JSON helpers only.
- Domain stores stay pillar-owned; they take/use `SqliteClient`.

## Implementation checklist

| Path | Work |
|------|------|
| `src/labpilot/accessor/` | Package skeleton |
| `accessor/sqlite/{client,migrate,schema}.sql` | Client + moved DDL |
| `accessor/common/` | Shared helpers |
| `intelligence/knowledge/store.py` | Use `SqliteClient` + accessor migrate |
| `labpilot.llm/` | Sole LLM SoR (providers, router, cache) — not under accessor |
| Tests | Migrate on temp DB; store still opens; LLM create_client unchanged |

## Acceptance criteria

- Existing KnowledgeStore tests / analyze smoke still pass.
- No second SQLite schema file left as SoR under `intelligence/knowledge/`.
- Import graph: `intelligence` → `accessor`; `accessor` ✖ `intelligence`.
- `SCHEMA_VERSION` still recorded in `schema_meta`.

## Test plan

- Unit: `SqliteClient` opens DB, runs migrate, foreign_keys on.
- Unit: KnowledgeStore round-trip artifact/hypothesis after migration.
- Unit: LLM client import path used by Micro Agents still resolves.

## Review notes

- Prefer a dedicated PR for this plan alone — high blast radius, no planner features yet.
- Confirm `schema.sql` stays under `sqlite/`, not `commons/`.

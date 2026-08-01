# Plan 2 — Planner schemas, DDL, and PlanStore

Back to [Research Planner](README.md). Design: [schema.md](schema.md) ·
[package-layout.md](package-layout.md) §4.

**Status:** Not started. **Depends on:** Plan 1. **Unlocks:** Plans 3–5.

---

## Goal

Stand up `research_engine/planner/` with Pydantic schemas (`ResearchPlan`, `ResearchTask`,
`TaskType`, statuses, verification/retry), add `research_plans` / `research_tasks` /
`research_task_deps` to the unified schema, and ship a `PlanStore` that persists and
reassembles plans via the accessor SQLite client.

## Why this matters

The structured DAG is the product. Without first-class DB entities and typed models, the
compiler and CLI have nothing durable to emit or show.

## In scope

- Package skeleton:

```
src/labpilot/research_engine/planner/
  __init__.py
  schemas/{models.py,task_types.py}
  store.py              # PlanStore
  validator.py          # DAG validate + topo levels (usable without full compiler)
```

- DDL in `accessor/sqlite/schema.sql` (bump `SCHEMA_VERSION`) — see [schema.md](schema.md)
- IDs: `P-001`, `P-001-T01`
- `PlanStore`: `upsert_plan`, `get_plan`, `list_plans`, `update_plan_status`,
  `update_task_status`
- `ResearchPaths.plans_dir` + `ensure()` includes `plans/`
- Keep Layer-3 `tasks` table untouched (collision documented in schema.md)

## Out of scope

- Compiler stages / templates / LLM (Plans 3–4)
- CLI (Plan 5)
- Capability executors / executing tasks
- Filling `estimated_cost` / `runtime_target` (columns exist; leave null)

## Design summary

- DB is SoR; JSON/MD projections come in Plan 3 serializer.
- `dependencies` on the model ↔ rows in `research_task_deps`.
- Validator: unique ids, deps resolve, acyclic; `topological_levels()`.

## Implementation checklist

| Path | Work |
|------|------|
| `planner/schemas/` | Enums + Pydantic models |
| `accessor/sqlite/schema.sql` | New tables + indexes |
| `planner/store.py` | PlanStore |
| `planner/validator.py` | DAG checks |
| `intelligence/paths.py` | `plans_dir` helpers (or planner-local path helper if preferred) |
| Tests | Schema migrate; upsert/get round-trip; cycle rejection |

## Acceptance criteria

- Migrating an existing `knowledge.db` adds the three tables without data loss.
- `upsert_plan` then `get_plan` reconstructs tasks + dependency edges.
- Validator rejects cycles and missing deps.
- Import hygiene: `planner` → `accessor`; `planner` ✖ `intelligence` for infra
  (reading HypothesisStore APIs for later plans is OK at API level).

## Test plan

- Unit: model validation (status enums, TaskType).
- Unit: PlanStore CRUD + cascade delete on plan.
- Unit: validator fixtures (valid DAG, cycle, dangling dep).

## Review notes

- Do not store plan nodes in Layer-3 `tasks`.
- Status transitions for executor (`running` / `in_progress`) are schema-ready only.

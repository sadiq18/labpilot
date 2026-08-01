# Plan 1 — Experience schema + ExperienceStore

Back to [README.md](README.md).

**Status:** Done.

## Goal

Define the M6 system of record: structured **Experience Records** and a durable
cross-competition **ExperienceStore** (SQLite).

Fields (minimum):

| Field | Notes |
|-------|--------|
| `id` | Stable experience id |
| `source_competition` | Origin slug |
| `goal` | Text |
| `hypothesis` | Text / optional id link |
| `action` | Text |
| `result` | Metrics summary / delta text or structured metric payload |
| `outcome` | Coarse label: `success` \| `fail` (extend later if needed) |
| `artifacts` | Refs: experiment id, metrics path/id, reflection id, `git_commit` when present |
| `tags` | String list for filter/retrieve |
| `idempotency_key` | From experiment/execution id for upserts |
| `created_at` / `updated_at` | Timestamps |

No category tables for prompts, HPs, architectures, or papers. Artifact links + tags
are enough for Context Engine retrieval.

## Acceptance

- [x] Pydantic (or equivalent) Experience Record model documented and importable when implemented
- [x] SQLite DDL + store API: create, get, upsert-by-idempotency-key, list/filter by competition/tags/outcome
- [x] Cross-competition queries supported (not scoped to a single slug only)
- [x] Unit tests for upsert idempotency and basic filters
- [x] No wiki/category satellite tables in this plan

## Implementation notes

- Package: `labpilot.research_engine.memory`
- Shared DB via `resolve_experience_db_path` (env → yaml → parent research root → `~/.labpilot`)
- Unified schema v8 + `experience_records`; ids `XR-xxx`; upserts on `idempotency_key`
- Client knowledge layout is flat (`knowledge/research/…`); do not nest under competition workspace

## Out of scope

- Extractor logic (plan 2)
- Context provider (plan 3)
- CLI (plan 4)
- Event/write hooks (plan 5)
- DuckDB, Qdrant, Kuzu

# Plan 1 — Experience schema + ExperienceStore

Back to [README.md](README.md).

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

- [ ] Pydantic (or equivalent) Experience Record model documented and importable when implemented
- [ ] SQLite DDL + store API: create, get, upsert-by-idempotency-key, list/filter by competition/tags/outcome
- [ ] Cross-competition queries supported (not scoped to a single slug only)
- [ ] Unit tests for upsert idempotency and basic filters
- [ ] No wiki/category satellite tables in this plan

## Out of scope

- Extractor logic (plan 2)
- Context provider (plan 3)
- CLI (plan 4)
- Event/write hooks (plan 5)
- DuckDB, Qdrant, Kuzu

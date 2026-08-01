# Backlog — Shared multi-tenant campaign store

**Status:** Backlog (post-M3). Today Conductor tables live in a **per-competition**
`knowledge.db` (`ConductorStore(knowledge_dir, competition)`), so sessions,
tasks, decisions, metrics, and suggestions do not span competitions or tenants.

## Problem

Operators and teams run many competitions. Useful campaign state is often
**shared**:

- Org/team budgets and autonomy defaults
- Cross-competition suggestion themes / capability gaps
- User-level “latest active session” across comps
- Shared metrics dashboards (ties to
  [telemetry-suggestions-export.md](telemetry-suggestions-export.md))

A competition-scoped SQLite file cannot express user / team / org tenancy.

## Proposed later work

- Introduce a **shared store** (Postgres or equivalent) with tenancy columns:
  `org_id`, `team_id`, `user_id`, plus `competition` as a dimension — not the
  sole partition key
- Shared tables (conceptual):
  - sessions / tasks / decisions (or references into per-comp DBs during
    migration)
  - campaign metrics rollups
  - suggestion index (blobs may live in S3)
  - org/team policy: default autonomy, budgets, gated tools
- Migration path: keep per-competition SQLite as local cache; sync or dual-write
  to shared store; eventually read shared for `status` / resume across comps
- AuthZ: row-level scope by org → team → user; competitions as resources under
  a team

## Signals to watch

- Multiple competitions under one operator with fragmented sessions
- Desire for org-wide budget / pause controls
- Cross-comp transfer (M6) needing shared experience tables

## Out of scope here

Implementing multi-tenant schema in M3. Per-competition SQLite remains correct
for the Campaign Engine milestone.

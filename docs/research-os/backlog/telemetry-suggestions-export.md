# Backlog — Campaign telemetry & suggestion storage

**Status:** Backlog (post-M3). M3 keeps counters in SQLite (`os_campaign_metrics`)
and suggestions in SQLite (`os_suggestions`) for durable local campaigns.

## Problem

SQLite is fine for single-machine Conductor loops, but campaign **metrics** and
**capability-gap suggestions** will outgrow a per-competition DB:

- Metrics need dashboards, traces, and correlation with LLM/tool spans
- Suggestions are append-heavy and useful across machines / teams (not just
  local `knowledge.db`)

## Proposed later work

### Metrics → OpenTelemetry + Phoenix / Langfuse

- Emit campaign counters and stop events as OTel metrics / spans / events
  (failed/blocked tasks, unmet goal, human interventions, `no_capability`,
  submissions, LLM cost, budget stops)
- Export to **Phoenix** and/or **Langfuse** for LLM-aware observability
- Keep a thin local rollup only if offline / resume still needs it; prefer
  remote as source of truth for ops dashboards
- Map session / competition / autonomy / stop_reason as resource or span attrs

### Suggestions → object storage (S3)

- Move `os_suggestions` payloads off SQLite into **S3** (or S3-compatible)
  objects keyed by org/team/competition/session
- Retain a small index (id, kind, created_at, object URI, session_id) if query
  UX still needs it — index may live in shared store (see
  [shared-multi-tenant-store.md](shared-multi-tenant-store.md))
- `conduct status` reads recent suggestions via index + optional hydrate from S3

## Signals to watch (while still on SQLite)

- `os_campaign_metrics` / `os_suggestions` row growth per competition
- Need for cross-machine campaign dashboards
- Duplicate suggestion themes that warrant shared analytics

## Out of scope here

Changing M3 schema or CLI. Local SQLite remains the M3 implementation.

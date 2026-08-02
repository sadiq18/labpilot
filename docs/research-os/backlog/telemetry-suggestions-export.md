# Backlog — Campaign telemetry & suggestion storage

**Status:** Backlog (post-M3) — **prerequisite** for unparking
[capability-registration](capability-registration.md).  
M3 keeps counters in SQLite (`os_campaign_metrics`) and suggestions in SQLite
(`os_suggestions`) for durable local campaigns.

**Unblocks:** [capability-registration](capability-registration.md) /
[design/11](../design/11-capability-registration.md) — maintainers cannot read
user-local `os_suggestions`; client telemetry (or opt-in export) is the product
gap feed.

## Problem

SQLite is fine for single-machine Conductor loops, but campaign **metrics** and
**capability-gap suggestions** will outgrow a per-competition DB:

- Metrics need dashboards, traces, and correlation with LLM/tool spans
- Suggestions are append-heavy and useful across machines / teams (not just
  local `knowledge.db`)
- LabPilot maintainers need aggregated gaps **without** access to user DBs

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

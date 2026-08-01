# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end Context Engine + Conductor online smoke; M5 handoff checklist.

## Acceptance

- [x] Integration tests green (`tests/unit/test_context_m4_capstone.py`)
- [x] M5 handoff checklist written
- [x] Backlog entries linked from M4 README

## M5 handoff checklist

M5 consumes M4 bundles; it does **not** re-own retrieve/rank/compress.

| Item | Notes |
|------|--------|
| [ ] Agent `execute(..., context: ContextBundle)` consumes M4 bundles | Pass compressed bundle (or `summary()` + refs) into specialist agents; do not re-query RI alone |
| [ ] Parallel workers reuse Context Engine sync facade or async API | Prefer `build_context` / `build_context_async`; Conductor loop may stay sync and call the facade |
| [ ] Event bus subscribers may rebuild context; do not bypass Conductor strategy | Rebuild via `build_context(ContextRequest(...))`; policy/tool choice still goes through Conductor |
| [ ] Keep submit gated; no ungated live Kaggle in M5 first cut | Same gate matrix as M3/M4; autonomy 0/1 only until level-2 design lands |

### Import / API contracts to preserve

- `CLI / Conductor / agents → context` allowed
- `intelligence` must **not** import `labpilot.research_engine.context`
- Durable shape: `ContextBundle` (+ `to_json()`); observe fields `context_summary` / `context_refs`
- Offline / `prefer_offline` must never *require* Context Engine success

### Deferred (not M5 blockers)

See [backlog](../../backlog/README.md): memory-hierarchy ports, hybrid/Qdrant, Kuzu,
capability registration, telemetry/S3, shared multi-tenant store.

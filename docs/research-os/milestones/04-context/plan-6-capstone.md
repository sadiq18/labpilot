# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end Context Engine + Conductor online smoke; M5 handoff checklist.

## Acceptance

- [ ] Integration tests green
- [ ] M5 handoff checklist written
- [ ] Backlog entries linked from M4 README

## M5 handoff checklist

- [ ] Agent `execute(..., context: ContextBundle)` consumes M4 bundles
- [ ] Parallel workers (asyncio/AnyIO) reuse Context Engine sync facade or async API
- [ ] Event bus subscribers may rebuild context; do not bypass Conductor strategy
- [ ] Keep submit gated; no ungated live Kaggle in M5 first cut

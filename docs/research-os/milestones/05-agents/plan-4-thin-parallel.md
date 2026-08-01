# Plan 4 — Thin parallel workers

Back to [README.md](README.md).

## Goal

Limited concurrency for **independent experiment tasks** under a sync Conductor:

```text
Conductor (sync)
   |
   +-- Experiment Task 1
   +-- Experiment Task 2
   +-- Experiment Task 3
        → asyncio.gather / AnyIO workers
```

Rules for first cut:

- Max workers (configurable)
- Shared budget accounting across concurrent tasks
- Collect results; surface failures without aborting siblings by default
- Workers reuse Context Engine sync facade or async API (M4 handoff)

**Not in this plan:** autonomous branch creation, merge trees, competing research
campaigns, conflict resolution — see
[parallel research branches backlog](../../backlog/parallel-research-branches.md).

## Acceptance

- [ ] Conductor remains sync; workers run under AnyIO/asyncio from a sync facade
- [ ] Concurrent experiment tasks respect max-workers and shared budget
- [ ] Results collected; per-task failure does not silently drop siblings’ outcomes
- [ ] No branch-merge / campaign-tree APIs
- [ ] Unit or integration test covers ≥2 concurrent fake experiment tasks

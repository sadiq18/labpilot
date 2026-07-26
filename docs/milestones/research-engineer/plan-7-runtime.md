# Plan 7 — Runtime capability (local / Kaggle / cloud)

Back to [Research Engineer](README.md). Design: [runtime-and-recovery.md](runtime-and-recovery.md) ·
[capabilities.md](capabilities.md).

**Status:** Not started. **Depends on:** Plan 6 (prefer smoke before remote). **Unlocks:** Plan 8.

---

## Goal

Implement **Runtime** as the only place that chooses and drives environments: local,
Kaggle Notebook/API patterns already in repo, and cloud dispatch/poll/pull. Absorb P2
“remote job dispatch” into this capability. Training/eval call Runtime; they do not open
SSH or Kaggle sessions themselves.

## In scope

- `capabilities/runtime/` wrapping `labpilot/runtimes/` (and related)
- Dispatch → poll → artifact pull; timeouts; cancel hooks if present
- Evidence: env id, job id, cost/time estimates if available, artifact URIs
- Idempotent resume: don’t double-submit if job already running (store job id on task/execution)

## Out of scope

- Model training logic (Plan 8)
- Submission upload (Plan 9)
- New cloud providers beyond what repo already supports (extend later)

## Acceptance criteria

- Local path works end-to-end for a no-op/runtime-ping task
- Remote path: mock or dry-run mode for CI; real path documented for local/dev
- LLM never selects environment

## Test plan

- Unit: local runner
- Unit: resume with existing job id skips re-dispatch
- Integration optional behind env flag

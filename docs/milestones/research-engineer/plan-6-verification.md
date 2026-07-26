# Plan 6 — Verification (unit + smoke gate)

Back to [Research Engineer](README.md). Design: [capabilities.md](capabilities.md) ·
[runtime-and-recovery.md](runtime-and-recovery.md).

**Status:** Not started. **Depends on:** Plan 5. **Unlocks:** Plans 7–8 (expensive work gated).

---

## Goal

Implement **Verification** capability: unit tests and **smoke** as the hard gate before
training/remote spend. Engineer calls verify after relevant tasks; capability owns running
checks and producing `TaskEvidence`.

## In scope

- `capabilities/verification/` + `execution/verification.py` integration
- Task types: `verify.unit`, `verify.smoke` (names per [capabilities.md](capabilities.md))
- Deterministic pass/fail; no LLM
- On fail: typed recovery (retry codegen once / fail) via Plan 2 hooks
- Evidence: command, exit code, log path, duration

## Out of scope

- Full training (Plan 8)
- Remote dispatch (Plan 7) — smoke may still be local-only for MVP

## Acceptance criteria

- Smoke fail prevents train tasks from starting (deps + Engineer gate)
- Smoke pass allows train queue progression
- ★ Gate documented in CLI/help or skill notes

## Test plan

- Unit: pass/fail evidence shapes
- Integration: code → unit → smoke fail → execution stopped before train stub

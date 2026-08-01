# Plan 4 — Conductor wiring

Back to [README.md](README.md).

## Goal

Online Conductor observe/policy consumes `ContextBundle` via sync
`build_context`. Offline Conductor path unchanged (no forced retrieve).

## Acceptance

- [x] `build_observe_bundle` includes context summary / refs
- [x] LLM policy prompt sees ranked evidence
- [x] `--offline` / `prefer_offline` does not require Context Engine success

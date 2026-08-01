# Plan 4 — Conductor wiring

Back to [README.md](README.md).

## Goal

Online Conductor observe/policy consumes `ContextBundle` via sync
`build_context`. Offline Conductor path unchanged (no forced retrieve).

## Acceptance

- [ ] `build_observe_bundle` includes context summary / refs
- [ ] LLM policy prompt sees ranked evidence
- [ ] `--offline` / `prefer_offline` does not require Context Engine success

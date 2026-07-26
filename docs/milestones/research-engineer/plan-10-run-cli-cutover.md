# Plan 10 — Plan-driven `run` / `resume`; retire obsolete orchestration

Back to [Research Engineer](README.md). Design: [architecture.md](architecture.md) ·
[package-layout.md](package-layout.md).

**Status:** Not started. **Depends on:** Plans 2–9 (capabilities registered). **Unlocks:** Plan 11.

---

## Goal

Wire CLI so `research run` is **plan-driven** (`--plan P-xxx` required for the Engineer
path). Implement `research resume`. Remove linear `Pipeline` / obsolete orchestration as
system of record after cutover. Keep thin shims only if needed for one release.

## In scope

- Enhance `cli/run.py` (and related): `--plan`, competition resolution from plan, approval
  gate, create `E-xxx`, call Engineer
- `research resume <execution_id|plan_id>` (exact UX in CLI.md)
- Delete or quarantine `orchestrator/pipeline.py` as SoR; update imports/tests
- Docs: CLI.md, ARCHITECTURE.md, SOP.md — Analyze → plan → run
- Deprecate `research run --competition` without `--plan` (error with migration message)

## Out of scope

- Capstone proof on a live competition (Plan 11)
- New capabilities

## Acceptance criteria

- `research run --plan P-001` drives Engineer end-to-end with registered capabilities
- Run without `--plan` fails with clear “create a plan first” message
- No production path depends on linear Pipeline
- `resume` continues interrupted `E-xxx`

## Test plan

- CLI: missing plan rejected
- CLI: approved plan starts execution
- Unit/integration: Pipeline module absent or unused by run path

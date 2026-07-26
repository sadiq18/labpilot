# Plan 4 — Workspace + Dependency capabilities

Back to [Research Engineer](README.md). Design: [capabilities.md](capabilities.md) ·
[package-layout.md](package-layout.md).

**Status:** Not started. **Depends on:** Plan 2 (registry). **Unlocks:** Plans 5–6.

---

## Goal

Implement **Workspace** and **Dependency** capability executors and register them for
`workspace.*` and `deps.*` task types. Prefer wrapping existing workspace / path helpers;
deterministic only.

## In scope

- `capabilities/workspace/` — ensure dirs, paths, git/init if already used
- `capabilities/dependency/` — install from lockfile/requirements; pin recording in evidence
- TaskContext: competition root, run/execution paths
- Evidence: paths created, packages installed / hashes

## Out of scope

- Code generation / LLM (Plan 5)
- Verification (Plan 6)
- Remote environments (Plan 7)

## Acceptance criteria

- Running a plan with only workspace+deps tasks succeeds on a fixture competition root
- Re-run is idempotent (dirs exist; deps already satisfied → skip or no-op)

## Test plan

- Unit: workspace creates expected tree
- Unit: dependency no-op when satisfied
- Integration: Engineer + these two capabilities on mini DAG

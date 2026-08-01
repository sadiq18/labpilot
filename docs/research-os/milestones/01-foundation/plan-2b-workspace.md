# Plan 2b — Workspace facade

Back to [Foundation](README.md). Design: [workspace.md](../../design/03-workspace.md).

## Goal

Provide a single `Workspace` handle over competition + knowledge layouts.

## Depends on

- Plan 1 (artifact path resolution may use workspace roots)

## Parallel with

- Plan 2a (tools)

## Work (when implementing)

- `Workspace.from_competition(...)` / from CWD `labpilot.yaml`
- Accessors: data, pipeline, knowledge, artifacts roots
- Support client-owned and legacy layouts already documented in pipeline design
- Unit tests for path resolution

## Acceptance

- [x] Workspace constructs for fixture competition roots
- [x] At least one tool/handler can take Workspace (even if CLI not switched yet)

## Notes (implemented)

- Module: `labpilot.research_engine.workspace_facade.Workspace`
- Composes existing `CompetitionWorkspace` / `competition_workspace_path`
- Tools in plan 2a accept `Workspace` (e.g. `submit`)

## Non-goals

- Goal field driven by Conductor (stub optional — `goal` field present)
- Git/terminal multiplexing

# Plan 1 — Agent runtime + registry + CodingTool

Back to [README.md](README.md).

## Goal

First-class specialist runtime package with:

- `Agent` port: `name`, `capabilities`, `async execute(task, workspace, context) -> ArtifactRefs`
- `SpecialistRegistry`: advertise capabilities, required tools, I/O artifacts, cost/duration hints
- Routing helper: Conductor asks registry for candidates by capability + budget + context
- `CodingTool` interface boundary wrapping **existing V1 Code Engineering** (no Claude Code / Aider / OpenHands yet)
- `execute(..., context: ContextBundle)` consumes M4 bundles

No Impl/Experiment specialist behavior, event bus, or parallel workers yet.

## Acceptance

- [x] Package importable; unit tests for registry register/lookup + CodingTool V1 smoke
- [x] Agents do not import peer agents; only Conductor / registry / tools / workspace
- [x] CodingTool is a swappable port; V1 is the only backend in this plan
- [x] Sync Conductor callers can invoke agent execute via AnyIO facade (no event loop required at call site)
- [x] ContextBundle is required input type for execute (M4 handoff)

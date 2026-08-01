# Plan 2a — Tool registry

Back to [Foundation](README.md). Design: [tools.md](../../design/02-tools.md).

## Goal

Register V1 capabilities as named tools with typed I/O over artifact adapters.

## Depends on

- Plan 1 (artifact contracts)

## Parallel with

- Plan 2b (workspace) — integrate in plan 3

## Work (when implementing)

- `ToolRegistry` + tool descriptors
- Register: `analyze_competition`, `generate_plan`, `run_plan`, `reflect`,
  `submit` / `submit_learn`, `query_memory` (minimal)
- Handlers call existing libraries in-process
- Unit tests: lookup, invoke fake tool, artifact out

## Acceptance

- [ ] Registry lists stage tools
- [ ] Handlers do not chain into the next stage
- [ ] Tests green

## Non-goals

- LLM function-calling protocol
- Full Claude-Code tool surface (`run_shell`, etc.)

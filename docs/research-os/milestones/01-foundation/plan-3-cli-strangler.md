# Plan 3 — CLI Strangler Phase A

Back to [Foundation](README.md).

## Goal

Stage CLIs (`analyze`, `plan`, `run`, `reflect`, `submit`) call **tools** and
persist **artifacts** via adapters — operator UX unchanged.

## Depends on

- Plan 2a and Plan 2b

## Work (when implementing)

- Thin CLI → tool invocations with Workspace
- Preserve flags and exit codes
- Update pipeline SOP/CLI only if behavior docs need wording (“tools underneath”)
- Regression: existing unit/CLI golden tests

## Acceptance

- [x] Documented stage commands still work
- [x] No direct stage→stage execute from CLI handlers
- [x] Relevant tests pass

## Notes (implemented)

- CLI resolves `Workspace` via `resolve_os_workspace` and invokes
  `build_default_tool_registry()` tools:
  `analyze_competition`, `generate_plan`, `run_plan`, `reflect`, `submit_learn`
- Inspection commands (`plan show` / `list`, journal, claims) stay store-backed
- `research resume` still uses Engineer directly (no resume tool yet)
- Guard: `tests/unit/test_cli_strangler.py`

## Non-goals

- Goal string CLI (M3)
- Conductor behind CLI (M2 Phase B)

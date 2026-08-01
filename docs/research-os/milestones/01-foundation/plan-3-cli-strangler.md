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

- [ ] Documented stage commands still work
- [ ] No direct stage→stage execute from CLI handlers
- [ ] Relevant tests pass

## Non-goals

- Goal string CLI (M3)
- Conductor behind CLI (M2 Phase B)

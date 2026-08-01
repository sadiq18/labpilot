# Plan 1 — Context Engine skeleton

Back to [README.md](README.md).

## Goal

First-class `context/` package with:

- `ContextBundle` / request models
- Provider port (`ContextProvider`)
- Sync `build_context()` facade over async AnyIO gather
- Thin RI retrieval provider wrapping existing Plan 9 `ContextBuilder`
- Abstract `GraphPort` (SQL-backed stub OK)

No BM25, rank, or Conductor wiring yet.

## Acceptance

- [x] Package importable; unit tests for facade + RI provider smoke
- [x] Sync callers never need an event loop
- [x] RI retrieval is reused, not copied/rewritten
- [x] Intelligence packages do not import `context/`

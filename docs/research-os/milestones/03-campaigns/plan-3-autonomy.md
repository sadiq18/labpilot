# Plan 3 — Autonomy 0 / 1

Back to [README.md](README.md).

## Goal

`--autonomy 0` (default): gate `generate_plan` + submit family.  
`--autonomy 1`: gate submit family only.  
Submit/submit_learn always gated (S2).

## Acceptance

- [x] Gate matrix matches autonomy level
- [x] Submit gated even when other tools auto-run

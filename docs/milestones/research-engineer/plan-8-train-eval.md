# Plan 8 — Training + Inference / Evaluation / Compare

Back to [Research Engineer](README.md). Design: [capabilities.md](capabilities.md).

**Status:** Done. **Depends on:** Plans 6–7. **Unlocks:** Plan 9.

---

## Goal

Wrap existing **training**, **inference**, **evaluation**, and **compare** flows as
capabilities invoked only after smoke. Metrics and artifact paths are written by the
platform (Runtime + capability), never invented by an LLM. Reuse DB `experiments` rows.

## In scope

- `capabilities/training/` — wrap `research_engine/training/`
- `capabilities/evaluation/` — infer + score + compare-to-baseline hooks
- Task types per capabilities map (`train.*`, `eval.*`, `compare.*`)
- Create/update `experiments` linked from execution/task metadata
- Evidence: metrics JSON path, checkpoint path, duration, env

## Out of scope

- Kaggle submission upload (Plan 9)
- LLM choosing hyperparameters mid-flight (suggest-only if ever; apply is deterministic/config)

## Acceptance criteria

- Train/eval run only when smoke ancestor is `done`
- Metrics land on disk + experiment row; Engineer does not fabricate scores
- Compare task can reference P-001 / best experiment id from context

## Test plan

- Unit: capability builds correct Runtime request
- Integration: stub Runtime + fake metrics → experiment row

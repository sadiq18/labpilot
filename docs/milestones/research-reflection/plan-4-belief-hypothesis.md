# Plan 4 — BeliefUpdater + HypothesisEvaluator

Back to [Research Reflection](README.md). Design: [beliefs-and-claims.md](beliefs-and-claims.md).

**Status:** Done. **Depends on:** Plans 1–3. **Unlocks:** Plan 5.

---

## Goal

Durable belief and hypothesis mutation from critic output, with audit trail and
status+why.

## In scope

- `BeliefUpdater`: confidence/status rules; append `belief_updates`
- `HypothesisEvaluator`: confirm/reject/partial/inconclusive + why; align
  `suggested`↔`proposed`
- Hook sketch: `mark_testing` when plan execution starts (wire fully in Plan 5)
- Prefer reflection as write path for post-run updates; keep file hypothesis SoR
  in sync

## Out of scope

- Research Claims promotion (Plan 7)
- Full Engineer cutover (Plan 5)

## Acceptance criteria

- [x] Critic “supports” increases belief confidence and writes audit row
- [x] Linked hypothesis gets status + why without manual CLI
- [x] Unit tests for arithmetic rules and status transitions

# Plan 5 — Engineer cutover

Back to [Research Reflection](README.md). Design: [architecture.md](architecture.md) §4.

**Status:** Done. **Depends on:** Plans 2–4. **Unlocks:** Plans 6–8, 10.

---

## Goal

Replace Reporting TaskType stubs (`REFLECT`, `UPDATE_BELIEF`, `CREATE_HYPOTHESIS`)
with reflection library calls; auto-run on execution success/fail when tasks present.

## In scope

- Update `execution/capabilities/reporting/capability.py`
- Call EvidenceExtractor → Critic → BeliefUpdater → HypothesisEvaluator
- `mark_testing` on plan start when hypothesis linked
- Workspace JSON may remain as **projection** of durable SoR (optional)

## Out of scope

- Journal CLI (Plan 8)
- Claims (Plan 7)
- Deleting top-level `reflection/` (Plan 9)

## Acceptance criteria

- [x] Dry-run plan with Reporting tasks writes evidence + belief_updates in DB
- [x] No durable mutation required for tasks not in the plan DAG
- [x] Existing Engineer unit/integration tests still green

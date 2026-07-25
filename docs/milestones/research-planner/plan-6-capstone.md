# Plan 6 — Capstone, docs polish, and acceptance

Back to [Research Planner](README.md). Design: [README.md](README.md) §10–11.

**Status:** Not started. **Depends on:** Plan 5. **Unlocks:** Research Planner MVP "done"
for plan-only scope.

---

## Goal

Prove the end-to-end plan-only loop on a seeded fixture (hypothesis → DAG → DB → projections),
close documentation gaps, and lock non-goals so the milestone does not slide into execution.

## Why this matters

Same role as Research Intelligence Plan 11: a reviewable gate that the product slice works
offline and matches the design.

## In scope

- Fixture competition/hypothesis (or reuse existing knowledge fixtures) that exercises
  template + optional mocked LLM path
- Capstone script or test: `plan create` → assert task types, deps acyclic, files on disk
- Docs sweep: `CLI.md`, `SOP.md` (analyze → hypothesize → **plan** → human → improve),
  `ARCHITECTURE.md` (three pillars + accessor), README status → Phase B implemented
- Optional: empty stubs for helper micro-agents (`risk_checker`, …) with skill.md noting
  "not wired" — only if cheap; otherwise document as Future
- Update [IN-PROGRESS.md](../IN-PROGRESS.md) / [MILESTONES.md](../../MILESTONES.md) status

## Out of scope

- Capability executors / `WRITE_CODE` runners
- Auto-calling `research improve` from a plan
- Cost optimizer / experiment budget allocator
- Accessor follow-ups unrelated to planner

## Design summary

Capstone answers: *Given hypothesis H, can LabPilot emit an inspectable, durable,
non-executing ResearchPlan DAG without an LLM?*

## Implementation checklist

| Path | Work |
|------|------|
| `tests/` or `tests/helpers/` | Fixture + capstone test |
| Docs | CLI / SOP / ARCHITECTURE / README status |
| Milestone indexes | Mark plans complete / MVP shipped |

## Acceptance criteria

- Offline capstone test green in CI (no network, no LLM key).
- README non-goals still accurate; no `--execute` anywhere.
- Operator can follow SOP: analyze → hypothesize → plan → (human) improve.

## Test plan

- Capstone integration-style unit on temp dirs.
- Regression: KnowledgeStore / analyze still pass (accessor migration).

## Review notes

- Resist sneaking executor work into this plan.
- Future work: helper micro-agents, competing planners, budget — backlog only.

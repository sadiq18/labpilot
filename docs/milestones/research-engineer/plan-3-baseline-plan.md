# Plan 3 — P-001 baseline plan compiler

Back to [Research Engineer](README.md). Design: [baseline-plan.md](baseline-plan.md) ·
Planner [../research-planner/](../research-planner/).

**Status:** Not started. **Depends on:** Planner MVP; can parallelize with Plan 2 after Plan 1.
**Unlocks:** Capstone (Plan 11); plan-driven baseline path.

---

## Goal

Add `research plan create <competition> --baseline` that allocates **P-001**, requires Analyze
context, refuses if any plan exists, sets `metadata.plan_kind = baseline`, and emits the
full baseline DAG (workspace → code → review → deps → unit → smoke → train → infer/eval →
submit → report → reflect).

## Why this matters

Baseline varies by problem type; it must be a first-class plan the Engineer can run — not a
hidden linear Pipeline.

## In scope

- Planner template(s) for baseline (problem-type aware via registry/profile/Analyze artifacts)
- CLI flag `--baseline` (mutually exclusive with `--hypothesis` for MVP)
- Optional Planning Engine revision (existing Option B soft-fail)
- Docs: CLI.md / plan help text

## Out of scope

- Running the plan (Plans 2, 10)
- Hypothesis-plan compare wiring beyond documenting compare-to-P-001 intent
- Removing legacy `research run --competition`

## Acceptance criteria

- `--baseline` → `P-001`, `plan_kind=baseline`, valid acyclic DAG
- Second `--baseline` on same competition fails clearly
- Missing Analyze context fails clearly
- Offline / `llm_client=None` works

## Test plan

- Unit: tabular-style fixture → expected task types present
- CLI: create baseline → show/list
- CLI: duplicate baseline rejected

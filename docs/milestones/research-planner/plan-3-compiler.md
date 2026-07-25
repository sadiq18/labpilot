# Plan 3 — Deterministic compiler (rule_engine path)

Back to [Research Planner](README.md). Design: [README.md](README.md) §3, §6 ·
[package-layout.md](package-layout.md) §4–5.

**Status:** Not started. **Depends on:** Plan 2. **Unlocks:** Plans 4–5.

---

## Goal

Implement the planning **compiler** as a deterministic pipeline that turns a `Hypothesis`
into a validated `ResearchPlan` DAG using rule_engine templates only (no LLM required).
Wire retrieval → context → template → lowering → validate/optimize/schedule → PlanStore →
derived JSON/MD.

## Why this matters

CI and offline use must produce a valid plan without an API key. The LLM (Plan 4) is an
optional upgrade on top of this path — same soft-fail posture as Research Brief.

## In scope

- Modules:

```
planner.py            # compile_research_plan() driver
retrieval.py          # bounded load: hypothesis, beliefs, brief snippets
context_builder.py    # PlanningContext / StructuredContext assembly
templates.py          # SpecAugment-style and generic technique templates
optimizer.py          # merge type-default verification/retry; topo helpers
scheduler.py          # order_index + parallel levels
serializer.py         # DB rows already via PlanStore; write .json + .md projections
```

- At least one keyword/tag-driven template (e.g. augmentation / SpecAugment) and a
  generic fallback template covering the ~15 `TaskType` instruction set where relevant
- Soft-fail: never mutates source, configs, or `runs/`
- Emit plan status `draft` or `ready`

## Out of scope

- Planning Engine LLM Micro Agent (Plan 4)
- CLI (Plan 5)
- Micro-helper agents (risk/dependency/evidence) — stubs OK, not required
- Cost/runtime formulas

## Design summary

```
hypothesis → retrieval → context_builder → templates (rule_engine)
  → lower (ids, verification defaults) → validator → optimizer → scheduler
  → PlanStore.upsert_plan → serializer JSON/MD
```

Markdown is **derived** from structure, never primary.

## Implementation checklist

| Path | Work |
|------|------|
| `planner/{planner,retrieval,context_builder,templates}.py` | Core stages |
| `planner/{optimizer,scheduler,serializer}.py` | Back-end stages |
| `knowledge/.../plans/` | Derived projections via paths |
| Tests | Template → valid DAG; no runs/ created; MD/JSON from model |

## Acceptance criteria

- `compile_research_plan(hypothesis)` with `llm_client=None` returns a valid `ResearchPlan`.
- Plan persisted; `<plan_id>.json` and `.md` written under `plans/`.
- No files under `runs/` or competition templates modified.
- Topological levels match dependency edges.

## Test plan

- Unit: SpecAugment-tagged hypothesis → expected task types present.
- Unit: generic hypothesis → fallback template still validates.
- Unit: serializer MD contains goal + task list from model fields.

## Review notes

- Conditional "if better, continue training" = `success_criteria` + gated `RUN_TRAINING`
  node — not executed.
- Keep retrieval bounded (L1–L3 budgets from README).

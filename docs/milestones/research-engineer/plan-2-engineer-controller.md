# Plan 2 — Research Engineer controller, queue, registry

Back to [Research Engineer](README.md). Design: [architecture.md](architecture.md) ·
[package-layout.md](package-layout.md).

**Status:** Done. **Depends on:** Plan 1. **Unlocks:** Plans 4–10 (capabilities plug in).

---

## Goal

Implement the deterministic **Research Engineer** orchestrator: load approved plan → create
`E-xxx` → topo queue → dispatch via `CapabilityRegistry` → verify hook → recovery stub →
resume/idempotency. Capabilities may be no-op / stub handlers that record evidence.

## Why this matters

One controller owns control flow. Without it, every capability reinvents queueing and resume.

## In scope

```
execution/
  engineer.py       # run_plan / resume
  context.py        # TaskContext assembly (bounded)
  registry.py       # CapabilityRegistry
  verification.py   # pass/fail over TaskEvidence (minimal)
  recovery.py       # stub typed policies (retry once / fail)
  evidence.py
  capabilities/base.py   # CapabilityExecutor protocol
```

- Walk `research_task_deps` / planner `topological_levels`
- Update plan + task + execution statuses
- Stub capability that marks tasks done with empty evidence (for wiring tests)
- No LLM in the controller

## Out of scope

- Real Workspace/Code/Train/… implementations (Plans 4–9)
- Baseline compiler (Plan 3)
- Deleting legacy `Pipeline` (Plan 10)
- Kaggle upload

## Acceptance criteria

- `run_plan(plan_id)` with all-stub registry completes a tiny fixture DAG end-to-end
- `resume(execution_id)` continues from first non-terminal task
- LLM never called from `engineer.py`

## Test plan

- Unit: topo dispatch order matches deps
- Unit: resume skips `done` tasks
- Unit: failed verify → recovery stub → fail execution cleanly

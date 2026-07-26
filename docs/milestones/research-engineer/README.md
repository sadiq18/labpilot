# Research Engineer

Back to [MILESTONES.md](../../MILESTONES.md).

**Status:** Phase B Plans 1–4 implemented; Plans 5–11 remaining.
**Depends on:** Research Planner MVP ([../research-planner/](../research-planner/)),
Research Intelligence Phase 1, Experiment Scientist.
**Branch:** `research-execution` (product name is Research Engineer).
**CLI target (after Phase B):** `research run --plan P-001` / `research resume --execution E-xxx`.
**CLI now:** `research plan create <competition> --baseline` → P-001.

This directory is the architecture/design workspace for the **Autonomous Research
Engineer** — Milestone 5.

| Doc | Role |
|-----|------|
| **This README** | Mission, Engineer vs Executor, CLI, success criteria, non-goals |
| [architecture.md](architecture.md) | One orchestrator, queue, capabilities, verification, recovery |
| [schema.md](schema.md) | Execution records; extend `research_tasks`; reuse `experiments` |
| [package-layout.md](package-layout.md) | `research_engine/execution/` tree + migration map |
| [baseline-plan.md](baseline-plan.md) | Analyze-derived **P-001** baseline DAG |
| [capabilities.md](capabilities.md) | Stable capability inventory + TaskType routing |
| [runtime-and-recovery.md](runtime-and-recovery.md) | Runtime dispatch, smoke gate, retry/resume |

### Phase B implementation plans

| # | Plan | Focus |
|---|------|--------|
| 1 | [plan-1-schema-executions.md](plan-1-schema-executions.md) | `research_executions`, evidence, task timing |
| 2 | [plan-2-engineer-controller.md](plan-2-engineer-controller.md) | Engineer, queue, registry, resume stubs |
| 3 | [plan-3-baseline-plan.md](plan-3-baseline-plan.md) | `plan create --baseline` → P-001 |
| 4 | [plan-4-workspace-deps.md](plan-4-workspace-deps.md) | Workspace + Dependency |
| 5 | [plan-5-code-review.md](plan-5-code-review.md) | Code Engineering + Research Review |
| 6 | [plan-6-verification.md](plan-6-verification.md) | Unit + smoke gate |
| 7 | [plan-7-runtime.md](plan-7-runtime.md) | Local / Kaggle / cloud Runtime |
| 8 | [plan-8-train-eval.md](plan-8-train-eval.md) | Training + Inference / Eval / Compare |
| 9 | [plan-9-submit-report.md](plan-9-submit-report.md) | Submission + Reporting / Memory |
| 10 | [plan-10-run-cli-cutover.md](plan-10-run-cli-cutover.md) | Plan-driven `run`/`resume`; retire Pipeline |
| 11 | [plan-11-capstone.md](plan-11-capstone.md) | Unattended Analyze → P-001 → evidence |

Ship-and-review one plan at a time (same style as Research Planner).

---

## 1. Mission

> **Turn research intent into experimental evidence.**

Given an approved Research Plan, autonomously implement, review, verify, train, evaluate,
submit, and reflect while producing durable evidence for every task and decision.

### Engineer vs Executor (locked)

| | Research Executor (rejected identity) | Research Engineer (this milestone) |
|--|---------------------------------------|------------------------------------|
| Role | Walk a DAG: dispatch → wait → mark done | Own **implementation** of the plan |
| Analogy | Job runner | Junior ML engineer on a research team |
| Scope | Thin dispatch | Workspace, code, review, verify, runtime, train, eval, submit, recover, evidence |

Capability executors are the **tools** the Engineer uses. They are not the milestone’s
identity.

---

## 2. Where this sits in the Research OS

```
analyze → knowledge / beliefs / hypotheses
              ↓
           plan  (Research Planner — shipped)
              ↓
         P-001 baseline (or hypothesis plan)
              ↓
    research run  ← Research Engineer (this milestone)
              ↓
    verified experiment + submission + artifacts
              ↓
         reflect → updated beliefs → next plan
```

Planner answers *how*. The Research Engineer owns *making it real*.

Roles are **not** agents. There is no Planner Agent → Coder Agent → Executor Agent chain.
There is one deterministic **Research Engineer** orchestrator that dispatches to stable
**capabilities**.

---

## 3. Success criteria (MVP gate)

An operator should be able to leave the laptop after:

```bash
research analyze <competition>
research plan create <competition> --baseline   # → P-001
research run --plan P-001
```

Hours later, evidence shows:

- ✓ Workspace created  
- ✓ Code implemented (and research-reviewed)  
- ✓ Unit tests passed  
- ✓ Smoke training passed (production-shaped gate)  
- ✓ Full training completed  
- ✓ Validation evaluated  
- ✓ Submission generated  
- ✓ Kaggle submission uploaded  
- ✓ Experiment report generated  

---

## 4. Locked product decisions

1. **Design docs first**, then Phase B plans, then code (same as Planner).
2. **Plan-driven only.** `research run --plan P-001`. Resume by durable execution id
   (`research resume --execution E-xxx`). No parallel legacy “slug → linear 14-stage
   pipeline” as the research path once the P-001 capstone lands.
3. **Baseline = P-001.** `research plan create <competition> --baseline` creates the first
   plan from Analyze outputs (problem type, profile, metric, rules, runtime constraints).
   No experiment-improvement hypothesis required. Later plans are hypothesis-driven and
   compare against P-001 / best experiment. Details: [baseline-plan.md](baseline-plan.md).
4. **One top-level orchestrator** — deterministic controller (queue, state, retries, resume,
   evidence, capability dispatch). Not multi-agent; not ReAct.
5. **Capabilities, not task agents** — ~8–12 stable capabilities; many `TaskType`s map to one
   capability. Details: [capabilities.md](capabilities.md).
6. **Micro-agents** — stateless reasoning slices **inside** capabilities that need judgement.
   Platform owns memory. Bounded `TaskContext` in → typed out → verify → recovery → forget.
7. **Full autonomous path in scope** — workspace through submit/upload, report, reflect,
   belief update; local + configured remote runtimes. Details:
   [runtime-and-recovery.md](runtime-and-recovery.md).

---

## 5. CLI sketch (Phase B)

```bash
# Baseline plan (first plan for a competition)
research plan create <competition> --baseline
# → P-001, metadata.plan_kind = baseline

# Hypothesis plan (existing)
research plan create <competition> --hypothesis H-xxx

# Implement the plan (Research Engineer)
research run --plan P-001
research resume --execution E-001

# Inspect
research plan show <competition> P-001
```

`research run` requires an approved `--plan`. Competition comes from the plan. One plan may
have multiple execution attempts (durable `E-xxx` ids).

Legacy `research run --competition <slug>` (linear Pipeline) is **deprecated** and removed
after the P-001 capstone proves equivalent baseline behavior. Stage implementations migrate
into capabilities; obsolete orchestration is deleted.

---

## 6. High-level architecture

```
Research Plan
      │
      ▼
Research Engineer          ← one orchestrator
      │
 ┌────┼────┐
 │    │    │
Code Runtime Validation    ← capabilities (tools)
 │    │    │
 └────┼────┘
      ▼
Experiment → Submission → Artifacts → Reflect
```

See [architecture.md](architecture.md).

---

## 7. Explicit non-goals

| Non-goal | Why |
|----------|-----|
| Multi-agent / ReAct “team of agents” | One Engineer owns control flow |
| Thin executor without implementation ownership | Wrong milestone identity |
| Unbounded repo edits | Controlled ops + Research Review only |
| LLM choosing next task / training / metrics / GPU / upload | Deterministic platform owns these |
| Unapproved Analyze → Run auto-loop | Human (or explicit approval) still gates `research run` |
| Planner helper micro-agents | Separate Planner backlog |
| Renaming Layer-3 `tasks` or overloading it | Planner collision already resolved |

---

## 8. Design Phase A checklist

- [x] README (this file) — mission, Engineer vs Executor, CLI, success, non-goals
- [x] [architecture.md](architecture.md)
- [x] [schema.md](schema.md)
- [x] [package-layout.md](package-layout.md)
- [x] [baseline-plan.md](baseline-plan.md)
- [x] [capabilities.md](capabilities.md)
- [x] [runtime-and-recovery.md](runtime-and-recovery.md)
- [x] Milestone indexes wired (`MILESTONES.md`, `IN-PROGRESS.md`, `ARCHITECTURE.md`, `SOP.md`)

Phase B plan docs are listed in the table above. Do **not** start application code until
Plan 1 is approved for implementation.

---

## 9. Phase B dependency graph

```text
Plan 1 (schema/executions)
    ├── Plan 2 (Engineer controller) ──┬── Plans 4–9 (capabilities, in order)
    │                                  └── Plan 10 (CLI cutover) → Plan 11 (capstone)
    └── Plan 3 (baseline P-001) ───────┘
         (Plan 3 may parallelize with Plan 2 after Plan 1)
```

| Phase | Plans | Outcome |
|-------|-------|---------|
| Foundation | 1–2 | Durable executions + deterministic Engineer |
| Baseline product | 3 | Analyze → P-001 without hypothesis |
| Capabilities | 4→9 | Workspace … Report (smoke before train/remote) |
| Cutover | 10 | `research run --plan` SoR; Pipeline retired |
| Proof | 11 | Unattended baseline experiment |

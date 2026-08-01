# Design — Artifact contracts

Back to [../README.md](../README.md) · Milestone: [../milestones/01-foundation/](../milestones/01-foundation/).

**Milestone:** M1 · **Impl branch:** `research-os-m1-foundation`

---

## Goal

Every pipeline stage **reads and writes typed artifacts**. No module depends on
another module’s concrete implementation to “call the next step.”

```text
CompetitionAnalyzer → CompetitionAnalysis artifact
Planner → ResearchPlan (+ tasks)
Engineer consumes plan/tasks → Experiment / Submission artifacts
Reflection → Reflection / memory updates
```

---

## Core artifact set (v1 contracts)

| Artifact | Producer (today) | Consumer |
|----------|------------------|----------|
| `CompetitionAnalysis` | Analyze / intelligence | Planner, Conductor observe |
| `ResearchPlan` + tasks | Planner | Engineer, Conductor |
| `Task` (OS-era) | Conductor (M2+) | Engineer / tools |
| `Experiment` | Eval / compare | Reflection, ranking, Conductor |
| `EvidenceCard` | COMPARE / submit-learn | Beliefs, graph, Conductor |
| `Reflection` / journal | Reflection pipeline | Conductor, humans |
| `Submission` | Submit capability | LB track, learning |
| `Workspace` snapshot refs | Workspace facade | All tools |

Reuse existing Pydantic / DB shapes where they already exist (`ResearchPlan`,
executions, Evidence Card). M1’s job is **stabilize boundaries and adapters**, not
rename everything.

---

## Observability (artifact trail)

Artifact writes should be attributable: which task/tool produced them, when, and
with what decision id (once M2 exists). Prefer appending to the decision/task log
over silent side effects — supports `explain` / `replay` later.

---

## Rules

1. Stages produce artifacts; they do not invoke the next stage’s `Execute()`.
2. Cross-package imports for “side-effect next step” are forbidden; orchestration
   owns sequencing (human CLI today; Conductor from M2).
3. Artifacts are durable (DB and/or workspace paths) and inspectable.
4. Version or schema_id fields where formats will evolve.

---

## Non-goals

- Replacing SQLite / file layout in M1
- Full OS `Task` model before M2 (may stub IDs that wrap plan tasks)
- Breaking existing CLI behavior

---

## Acceptance (when implementing)

- Documented schema or adapter module per core artifact
- Analyze / plan / run / reflect write through adapters
- Unit tests: round-trip serialize + “no direct stage→stage execute” lint or review checklist

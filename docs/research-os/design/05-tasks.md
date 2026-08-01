# Design — Tasks

Back to [../README.md](../README.md) · Milestone: [../milestones/02-conductor/](../milestones/02-conductor/).

**Milestone:** M2 · **Impl branch:** `research-os-m2-conductor`

---

## Goal

Replace hidden sequential stage calls with a durable **task queue** the Engineer
(and tools) **consume**.

---

## Task fields (minimum)

| Field | Purpose |
|-------|---------|
| `id` | Stable id (e.g. `T-401`) |
| `type` / tool name | What to run |
| `status` | Lifecycle |
| `priority` | Scheduling |
| `owner` | Conductor / agent id (later) |
| `dependencies` | Task ids that must complete first |
| `retry` | Count / policy |
| `artifacts` | Inputs/outputs refs |
| `context_ref` | Optional bundle id (richer in M4) |

---

## Lifecycle

```text
Pending → Running → Completed
                 ↘ Failed → Retry → Pending | Cancelled
```

Also track campaign-level job states: Paused | Sleeping | Waiting (approvals /
external kernels). Everything is resumable.

---

## Scheduler ≠ executor

Separate **scheduling** from **execution**:

```text
Priority queue → dependency resolution → resource/budget checks → worker assignment
```

Workers are tools / Engineer / (later) agents. **Never** let specialists call each
other directly — they complete tasks; the Conductor + queue decide what’s next.

### Queue implementation

| Phase | Choice |
|-------|--------|
| M2–M3 | In-process scheduler + durable rows in SQLite |
| Later | **Temporal** (or similar) only if we need distributed, multi-machine durability |

Do not introduce Temporal for a single-laptop research loop.

---

## Checkpointing

Treat tasks like resumable jobs, not chat sessions. Periodically persist:

- Workspace refs / dirty paths
- Current plan / objective
- Task queue + open tasks
- Compact context / last decision ids

`research continue` / `resume` restores from the last checkpoint
([06-campaigns](06-campaigns.md)).

---

## Relation to ResearchPlan

V1 `ResearchPlan` tasks remain valid. M2 may:

- Wrap a whole plan as one high-level task (`run_plan`), or
- Project plan tasks into the OS queue one-for-one

Prefer wrapping first (less churn); project later if Conductor needs finer control
(M3).

---

## Non-goals

- Parallel fan-out (M5)
- Per-task LLM context assembly (M4)
- Distributed workflow engine in M2

# Design — Parallel execution and events

Back to [../README.md](../README.md) · Milestone: [../milestones/05-agents/](../milestones/05-agents/).

**Milestone:** M5 · **Impl branch:** `research-os-m5-agents`

---

## Why one design doc

Parallel fan-out and a full event bus both exist to **decouple** work and reduce
wall-clock. They share M5 with specialist agents.

---

## Parallel tree

```text
Conductor
   ├─ Paper branch
   ├─ Feature branch
   └─ Model branch
         → join / compare → reflect → next iteration
```

Requirements: independent tasks, merge policy, budget accounting across branches.
Scheduler assigns workers; agents still do not call each other.

---

## Event log vs event bus

| Layer | When | Role |
|-------|------|------|
| Decision/task **event log** | M2 | Append-only; resume/explain |
| **Pub/sub bus** | M5 | Decoupled reactions |

Publish examples: `ExperimentCompleted`, `ReflectionGenerated`, `PaperAdded`,
`DatasetChanged`, `ModelFailed`, `SubmissionAccepted`, `LeaderboardUpdated`.

Reflection (and others) **subscribe** — they are not invoked via hard-coded
downstream calls from Experiment.

### Bus implementation

| Phase | Choice |
|-------|--------|
| M5 first cut | In-process pub/sub (e.g. **Blinker**) on top of the durable log |
| Multi-process / multi-host | **NATS** or **Redis Streams** |

Do not start with a distributed bus for a single Conductor process.

---

## Non-goals

- Distributed multi-machine orchestration in the first M5 cut
- Replacing the M2 log (bus builds on it)
- Agent→agent RPC as control flow

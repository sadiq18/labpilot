# M2 — Research Conductor

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Implemented (kernel).  
**Branch:** `research-os-m2-conductor`  
**Depends on:** M1  
**Design:** [04-conductor](../../design/04-conductor.md) · [05-tasks](../../design/05-tasks.md)

## Mission

Pipeline becomes orchestration: **“What should happen next?”** Decide-only
Conductor, durable task queue, scheduler. Constrained LLM picks the next tool
from the fixed catalog (not a rigid Analyze→Plan→Run→Reflect pipeline; not
free-form planning).

## Usable outcome

- Product entry: `research conduct "<goal>"`
- Power-user stage CLIs remain: `analyze` / `plan` / `run` / `reflect`
- Durable decisions + queue; tools execute; approvals with operator comments

## CLI hierarchy

```text
User Layer        research conduct "goal"
Power User Layer  research analyze | plan | run | reflect | submit
Runtime Layer     Conductor → Task Queue → Tools / Engineer → Memory
```

## Tech that ships with M2

| Area | Technology |
|------|------------|
| Orchestration | Custom Conductor |
| Queue / scheduler | Custom; SQLite persistence |
| LLM policy | Structured NextAction via existing router (LiteLLM cutover OK) |
| Observability | Structured decision/tool/approval logs |

Not yet: Blinker bus, Temporal, Ray, specialist agents, dynamic task creation (M3).

## Phase plans

| # | Plan |
|---|------|
| 1 | [plan-1-queue.md](plan-1-queue.md) |
| 2 | [plan-2-policy.md](plan-2-policy.md) |
| 3 | [plan-3-loop-cli.md](plan-3-loop-cli.md) |
| 4 | [plan-4-capstone.md](plan-4-capstone.md) |

## Non-goals

- Dynamic research task invention (M3 Campaign Engine)
- Full pub/sub (M5)
- Parallel fan-out (M5)
- LangGraph/CrewAI

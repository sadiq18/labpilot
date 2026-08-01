# M3 — Campaign Engine

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Implemented (phase plans 1–6).  
**Branch:** `research-os-m3-campaigns`  
**Depends on:** M2  
**Design:** [06-campaigns](../../design/06-campaigns.md)

## Mission

Turn the M2 Conductor kernel into the **Campaign Engine**: dynamic research
actions that map onto **existing** tools, budgets, stop conditions, autonomy 0/1,
and checkpoint restore. Completes the **Orchestrator** product stage.

## Usable outcome

```text
research conduct run "Win Rogii" --autonomy 0
research conduct continue | pause | resume | status
```

Operator drives by goal; workflow is not limited to single-tool picks; resume
after leaving the machine.

## Phase plans

| # | Plan | Focus |
|---|------|--------|
| 1 | [plan-1-checkpoint-cli.md](plan-1-checkpoint-cli.md) | Checkpoint + continue/pause/resume/status |
| 2 | [plan-2-budgets-stops.md](plan-2-budgets-stops.md) | Budgets + automatic stop conditions |
| 3 | [plan-3-autonomy.md](plan-3-autonomy.md) | Autonomy 0/1; submit always gated |
| 4 | [plan-4-actions-compose.md](plan-4-actions-compose.md) | ResearchAction → existing tools |
| 5 | [plan-5-metrics-suggestions.md](plan-5-metrics-suggestions.md) | Gap metrics + suggestions |
| 6 | [plan-6-capstone.md](plan-6-capstone.md) | Integration + M4 handoff |

## Tech

| Area | Technology |
|------|------------|
| Loop | Sync Conductor (extend M2) |
| Checkpoint | SQLite + session metadata |
| Runtime | Sync only — asyncio deferred to M4/M5 |

## Non-goals

- New tools / capability registration (see [backlog](../../backlog/capability-registration.md))
- Remote telemetry / S3 suggestions (see [telemetry backlog](../../backlog/telemetry-suggestions-export.md))
- Shared multi-tenant store across competitions (see [tenancy backlog](../../backlog/shared-multi-tenant-store.md))
- Agent runtime, UI, ungated live submit
- Context engine (M4), multi-agent/parallel (M5), transfer (M6)
- Autonomy levels 2–3

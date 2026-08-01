# M3 — Long-running campaigns

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub.  
**Branch:** `research-os-m3-campaigns`  
**Depends on:** M2  
**Design:** [06-campaigns](../../design/06-campaigns.md)

## Mission

Autonomous loop under budgets: dynamic tasks, approval ladder, checkpointing,
`research "<goal>"` / continue / pause / resume. Completes the **Orchestrator**
product stage.

## Usable outcome

Operator drives by goal; workflow no longer fixed; resume from checkpoint.

## Tech that ships with M3

| Area | Technology |
|------|------------|
| Runtime | asyncio (+ AnyIO) |
| Checkpoint | SQLite + workspace refs |
| Distributed jobs | Deferred (Temporal only if multi-machine later) |

## Non-goals

- Context engine (M4)
- Agent zoo / parallel trees (M5)
- Cross-comp transfer (M6)

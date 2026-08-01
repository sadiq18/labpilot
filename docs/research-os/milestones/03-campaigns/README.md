# M3 — Campaign Engine

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub.  
**Branch:** `research-os-m3-campaigns`  
**Depends on:** M2  
**Design:** [06-campaigns](../../design/06-campaigns.md)

## Mission

Turn the M2 Conductor kernel into the **Campaign Engine**: dynamic research
tasks beyond the fixed tool catalog, budgets, autonomy ladder, and checkpoint
restore. Completes the **Orchestrator** product stage.

Extends `research conduct` (from M2) with `continue` / `pause` / `resume` /
`status` — not a second product entrypoint.

## Usable outcome

Operator drives by goal; workflow is no longer limited to the fixed catalog;
resume from checkpoint after leaving the machine.

## Tech that ships with M3

| Area | Technology |
|------|------------|
| Runtime | asyncio (+ AnyIO) |
| Checkpoint | SQLite + workspace refs |
| Distributed jobs | Deferred (Temporal only if multi-machine later) |

## Non-goals

- Context engine (M4) — improves decide quality
- Agent zoo / parallel trees / event bus (M5) — delegate + concurrent branches
- Cross-comp transfer (M6) — learn across campaigns

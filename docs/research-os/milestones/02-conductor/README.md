# M2 — Research Conductor

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub — deep Phase B when pulled forward.  
**Branch:** `research-os-m2-conductor`  
**Depends on:** M1  
**Design:** [04-conductor](../../design/04-conductor.md) · [05-tasks](../../design/05-tasks.md)

## Mission

Pipeline becomes orchestration: **“What should happen next?”** Decide-only Conductor,
task queue, scheduler. Strangler B = fixed Analyze→Plan→Run→Reflect.

## Usable outcome

Same research loop via Conductor; durable decisions + queue; Engineer consumes tasks.

## Tech that ships with M2

| Area | Technology |
|------|------------|
| Orchestration | Custom Conductor |
| Queue / scheduler | Custom + asyncio; SQLite persistence |
| LLM policy router | LiteLLM (or thin router → LiteLLM) |
| Observability | Structured decision/tool logs |

Not yet: Blinker bus, Temporal, Ray, specialist agents.

## Non-goals

- Dynamic extra tasks / goal CLI (M3)
- Full pub/sub (M5)
- LangGraph/CrewAI

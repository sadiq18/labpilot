# M5 — Agents, events, parallel

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub.  
**Branch:** `research-os-m5-agents`  
**Depends on:** M4  
**Design:** [08-agents](../../design/08-agents.md) · [09-parallel-and-events](../../design/09-parallel-and-events.md)

## Mission

Specialist agents + registry routing, pub/sub event bus, parallel branches with
Conductor merge.

## Usable outcome

Dynamic “who can solve this?”; decoupled subscribers; faster independent work.

## Tech that ships with M5

| Area | Technology |
|------|------------|
| Agent runtime | Custom |
| Implementation coding | Adapter to Claude Code / Aider / OpenHands (do not rebuild) |
| LLM / structured | LiteLLM + PydanticAI (or Instructor) |
| Event bus | Blinker → NATS/Redis if multi-process |
| Parallel / execute | **asyncio + AnyIO** workers + Docker sandboxes → Ray if needed |

M3 kept a **sync** Conductor loop on purpose. M5 is where concurrent experiments,
background jobs, and remote/GPU workers enter scope.

## Non-goals

- LangGraph/CrewAI core
- Kubernetes as M5 gate
- Peer agents overriding Conductor

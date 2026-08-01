# M5 — Agents, events, parallel

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Implementing (plans 1–2 done; next plan 3).  
**Branch:** `research-os-m5-agents`  
**Depends on:** M4  
**Design:** [08-agents](../../design/08-agents.md) · [09-parallel-and-events](../../design/09-parallel-and-events.md)

## Mission

Turn capabilities into **controlled specialists**: registry routing, Implementation +
Experiment first cut, in-process event bus, thin parallel workers. Conductor stays
the sync decision layer; workers execute underneath. Context Builder (M4) feeds
specialists — M5 does not rebuild intelligence layers.

Handoff from M4: [../04-context/plan-6-capstone.md](../04-context/plan-6-capstone.md).

## Usable outcome

```text
Conductor → Specialist Registry → Implementation | Experiment
Event bus (Blinker) decouples reactions
Thin asyncio workers run independent experiment tasks
Git commits track code evolution per experiment
```

Dynamic “who can solve this?” for Impl/Experiment; wall-clock drops on parallel
experiment tasks; code changes are revertible via commit hash.

## Phase plans

| # | Plan | Focus |
|---|------|--------|
| 1 | [plan-1-agent-runtime.md](plan-1-agent-runtime.md) | Agent port, registry, CodingTool interface (V1 wrap) |
| 2 | [plan-2-impl-experiment.md](plan-2-impl-experiment.md) | Implementation + Experiment specialists; Conductor routing |
| 3 | [plan-3-event-bus.md](plan-3-event-bus.md) | Blinker pub/sub on M2 decision/task log |
| 4 | [plan-4-thin-parallel.md](plan-4-thin-parallel.md) | asyncio workers: max concurrency, shared budget, collect results |
| 5 | [plan-5-git-code-evolution.md](plan-5-git-code-evolution.md) | Branch/commit per experiment; hash on artifact; `research revert` |
| 6 | [plan-6-capstone.md](plan-6-capstone.md) | Integration tests + M6 handoff |

**Order:** plan-1 → … → plan-6.

## First-cut specialists

```text
Specialist Registry
        |
        +-- Implementation Specialist
        |       - write/update code, tests, fixes
        |       - EDA / features as code tasks (not separate agents)
        |       - via CodingTool → V1 Code Engineering
        |
        +-- Experiment Specialist
                - run experiments, collect metrics, compare results
                - track evidence; emit events
```

Defer Paper, Reflection/Critic, Submit/Eval, Literature, EDA, Feature Engineering
as separate agents — see [future specialists backlog](../../backlog/future-specialists.md).

## Tech that ships with M5

| Area | Technology |
|------|------------|
| Agent runtime | Custom (not LangGraph/CrewAI) |
| Coding backend | **CodingTool** port → existing V1 Code Engineering |
| LLM / structured | LiteLLM + PydanticAI (or Instructor) via Conductor LLM layer |
| Event bus | **Blinker** in-process on M2 log → NATS/Redis later |
| Parallel | **asyncio** / AnyIO workers from sync Conductor facade |
| Sandbox | Existing Docker / execute_python paths |
| Code memory | **Git** branch + commit per experiment (code only) |

## Non-goals

- LangGraph/CrewAI as core runtime
- Kubernetes / Ray as M5 gates
- Peer agents overriding Conductor strategy
- Full research branch-merge tree ([backlog](../../backlog/parallel-research-branches.md))
- Claude Code / Aider / OpenHands adapters ([backlog](../../backlog/coding-tool-adapters.md))
- Async Conductor / distributed scheduler ([backlog](../../backlog/async-conductor.md))
- Separate EDA / Feature Engineering / Paper / Critic / Submit agents ([backlog](../../backlog/future-specialists.md))
- Ungated live Kaggle submit
- Committing knowledge artifacts / Research Graph into git (git = code evolution only)

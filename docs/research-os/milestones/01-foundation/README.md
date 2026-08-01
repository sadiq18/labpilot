# M1 — Platform Foundation

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Plans 1–2b implemented on branch; plan 3 (CLI strangler) next.  
**Branch:** `research-os-m1-foundation`  
**Design:** [01-artifacts](../../design/01-artifacts.md) · [02-tools](../../design/02-tools.md) · [03-workspace](../../design/03-workspace.md)

## Mission

Stop thinking in pipeline stages. Ship **artifacts + workspace + tools + service
interfaces** under unchanged CLIs (Strangler A). Highest-priority OS milestone —
do not skip.

## Usable outcome

Operators keep `research analyze|plan|run|reflect`. Internals are OS-shaped.

## Architecture that ships

| Slice | Outcome |
|-------|---------|
| Artifact system | Typed objects; no stage→stage `Execute()` |
| Workspace | Persistent facade; tools get Workspace (+ Task later) |
| Tool runtime | Capabilities as `Tool.run` |
| Service layer | Planner creates tasks/artifacts; Engineer consumes |

## Tech that ships with M1

| Area | Technology |
|------|------------|
| Tool system | Custom registry |
| Models | Pydantic |
| Config | Pydantic Settings |
| Storage | SQLite |
| Git helper | GitPython if needed |
| Logging | structlog (or structured logs) |

Do **not** pull in Conductor, Kuzu, Qdrant, Temporal, or agent frameworks here.

## Phase B plans

| # | Plan | Parallel |
|---|------|----------|
| 1 | [plan-1-artifacts.md](plan-1-artifacts.md) | Serial |
| 2a | [plan-2a-tools.md](plan-2a-tools.md) | ∥ 2b after 1 |
| 2b | [plan-2b-workspace.md](plan-2b-workspace.md) | ∥ 2a after 1 |
| 3 | [plan-3-cli-strangler.md](plan-3-cli-strangler.md) | After 2a+2b |
| 4 | [plan-4-capstone.md](plan-4-capstone.md) | After 3 |

## Non-goals

- Conductor / goal CLI / agents
- Changing Engineer’s in-plan deterministic walk

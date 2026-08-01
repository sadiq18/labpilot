# Design — Specialist agents

Back to [../README.md](../README.md) · Milestone: [../milestones/05-agents/](../milestones/05-agents/).

**Milestone:** M5 · **Impl branch:** `research-os-m5-agents`

---

## Goal

Promote tool bundles into **Conductor-scheduled specialists** with a registry and
capability routing (“who can solve this?”) — not peer autonomous managers.

**Build a lightweight custom agent runtime.** Do not use LangGraph or CrewAI as the
core architecture (fine for throwaway prototypes; wrong for an OS).

Enough concepts:

```text
Goal → Task → Agent → Tool → Artifact → Event
```

Sketch:

```python
class Agent:
    name: str
    capabilities: list[str]

    async def execute(self, task, workspace, context) -> ArtifactRefs:
        ...
```

Agents **must not** know about each other. Communication path:

```text
Conductor → Event Bus / Task Queue → Workers
```

not `PaperAgent → ExperimentAgent`.

---

## Illustrative specialties

| Agent | Domain |
|-------|--------|
| Research / Paper | Literature and technique intake |
| Data | Dataset prep / profiling |
| Implementation | Code, tests, fixes — **via CodingTool adapter**, not a from-scratch coding agent |
| Experiment | Train / eval loops — sandbox executor (Docker/…); strategy from Conductor |
| Evaluation | Compare / Evidence Card |
| Submission | Checkpoint → upload → LB track |
| Reflection / Critic | Lessons and memory updates |

**Implementation agent** schedules work; **coding backends** are swappable data-plane
tools ([02-tools](02-tools.md), [architecture §3](../architecture.md#3-control-plane-vs-data-plane-build-vs-reuse)).

The V1 Research Engineer maps to **Implementation + Experiment execution** under
deterministic plan/task consumption — strategy stays with the Conductor. Existing
Code Engineering can remain until the adapter replaces it.

---

## Registry

Agents advertise: capabilities, required tools, input/output artifacts, cost,
estimated duration.

---

## Routing

Conductor asks the registry for candidates; selects by capability + budget +
context — similar to tool selection, at agent granularity.

---

## Promotion rule

Start as tools (M1–M4). Promote to an agent only when a stable skill loop +
context profile exists. Agents never override Conductor strategy or approval
policy. LLM calls go through the isolated LLM layer
([04-conductor](04-conductor.md)), not raw SDKs inside each agent.

---

## Non-goals

- Multi-agent debate replacing Conductor
- LangGraph/CrewAI as the runtime
- Shipping seven agents on day one of M5 — land registry + 1–2 specialists first
- Rebuilding Claude Code / Cursor inside LabPilot as the differentiator

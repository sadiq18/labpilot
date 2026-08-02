# Autonomous Research Engineer → Research Operating System

Back to [docs index](../README.md). V1 pipeline docs:
[../research-pipeline/](../research-pipeline/).

**Status:** M1 Platform Foundation implemented on `research-os-m1-foundation`
(artifacts + workspace + tools + CLI strangler). Next: M2 Conductor.  
**Branch (impl):** `research-os-m1-foundation` · **Design:** historically `research-os-design`  
**Ops / parallel schedule:** [execution-plan.md](execution-plan.md)  
**Principles:** [architecture.md](architecture.md)

This is the **single product roadmap**. Architecture and technology **co-evolve per
milestone**. We do **not** pick Neo4j / Qdrant / Temporal / Ray up front — each
milestone ships a usable system and only the stack it needs, laying foundation for
the next.

---

## Vision

```text
Research Pipeline                    (M0 — done)
        ↓
Autonomous Research Orchestrator     (M1–M3)
        ↓
Research Operating System            (M4–M6 + goal UX)
        ↓
General Autonomous Research Engineer (beyond Kaggle — later)
```

**Principle:** The Conductor decides. Tools (then agents) perform. The Engineer
executes approved work correctly. Memory remembers.

**Tech principle:** Build the OS yourself. Use libraries for **primitives** (LLM
access, schemas, storage engines, telemetry) — never for the execution loop.
No LangGraph/CrewAI as core architecture.

**Build vs reuse:** Build the **control plane** (Conductor, memory, context,
strategy). Reuse the **data plane** (coding agents, sandboxes, browsers, git).
See [architecture.md §3](architecture.md#3-control-plane-vs-data-plane-build-vs-reuse).
Details: [architecture.md](architecture.md).

---

## Roadmap at a glance

| Stage | Product name | Usable outcome | Track | Branch |
|-------|--------------|----------------|-------|--------|
| **M0** | Research Pipeline | Analyze→Plan→Run→Reflect loop | [research-pipeline](../research-pipeline/) | — |
| **M1** | Platform Foundation | Artifacts + Workspace + Tools (CLI unchanged) | [milestones/01-foundation/](milestones/01-foundation/) | `research-os-m1-foundation` |
| **M2** | Research Conductor | Constrained LLM + task queue + `research conduct` | [milestones/02-conductor/](milestones/02-conductor/) | `research-os-m2-conductor` |
| **M3** | Campaign Engine | Dynamic tasks, budgets, continue/pause/resume | [milestones/03-campaigns/](milestones/03-campaigns/) | `research-os-m3-campaigns` |
| **M4** | Memory & context | Retrieve→rank→compress; hierarchy ports | [milestones/04-context/](milestones/04-context/) | `research-os-m4-context` |
| **M5** | Agents, events, parallel | Specialists + bus + concurrent branches | [milestones/05-agents/](milestones/05-agents/) | `research-os-m5-agents` |
| **M6** | Self-improving memory | Experience records + retrieve/seed CLI | [milestones/06-transfer-memory/](milestones/06-transfer-memory/) | `research-os-m6-transfer-memory` |

**Critical path:** M1 → M2 → M3 (Orchestrator). M4–M6 deepen the OS into a research
**manager** (better decisions → delegate/parallel → learn across campaigns).  
**Order note:** Context (**M4**) before Agents (**M5**) — specialists without retrieval
regress quality. Full event bus ships with M5; M2 only needs an append-only decision
log. Long-running **Campaign Engine** autonomy is **M3** (M2 ships the Conductor kernel
+ `research conduct`).

Design satellites (`design/01`…`11`) hold depth; **this README is the roadmap**.

---

## M0 — Foundations (current / done)

**Goal:** First autonomous research loop on Kaggle.

**Architecture:** `Analyze → Plan → Research Engineer → Reflect`

**Delivered:** Competition analyzer, knowledge collector, planner, Research Engineer,
reflection, Evidence Card / Research Graph seeds, SQLite knowledge DB, Typer CLI.

**Stack in production today** (baseline — change only when a later milestone needs it):

| Area | Technology |
|------|------------|
| Language | Python 3.11+ (target 3.12+) |
| Package | uv |
| CLI | Typer |
| Validation / config | Pydantic / Pydantic Settings |
| Database | SQLite |
| LLM | Existing provider router (LiteLLM cutover when Conductor policy lands) |
| Async | Sync Engineer OK; asyncio at boundaries as needed |
| Testing | pytest |

Docs: [../research-pipeline/](../research-pipeline/).

---

## M1 — Platform Foundation *(do not skip)*

**Goal:** Stop thinking in pipeline stages. Think **artifacts, tasks, workspace, tools**.

**Architecture that ships:**

```text
Stages → produce Artifacts
Planner → Create Tasks (not call Executor)
Every capability → Tool
Every tool → Workspace + Task
```

| Slice | Outcome |
|-------|---------|
| 1.1 Artifact system | Typed `CompetitionAnalysis`, `ResearchPlan`, `Task`, `Experiment`, `Reflection`, `Submission`, … — no module invokes the next’s `Execute()` |
| 1.2 Workspace | Persistent facade: goal, files, artifacts, experiments, memory handles, git, datasets, models, logs |
| 1.3 Tool runtime | `analyze`, `generate_plan`, `run_plan`, `evaluate`, `submit`, `reflect`, `query_memory`, … — data-plane tools (coding/sandbox/browser) arrive as **adapters**, not rebuilds ([architecture §3](architecture.md#3-control-plane-vs-data-plane-build-vs-reuse)) |
| 1.4 Service layer | Planner creates tasks/artifacts; Engineer **consumes** — Strangler Phase A under existing CLIs |

**Usable system:** Same `research analyze|plan|run|reflect` UX; internals are OS-shaped.

**Tech that ships with M1** (only):

| Area | Technology |
|------|------------|
| Tool system | **Custom** registry + **adapters** to external coding/sandbox/browser tools when needed |
| Workspace / artifacts | **Pydantic** models |
| Config | Pydantic Settings (existing) |
| Storage | SQLite (existing) |
| Git (optional helper) | GitPython when needed |
| Logging | structlog (or structured stdlib → structlog) |
| Coding agent | **Do not build** — optional adapter later; V1 code eng OK for now |

Depth: [design/01–03](design/01-artifacts.md) · Plans: [01-foundation](milestones/01-foundation/).

---

## M2 — Research Conductor

**Goal:** Pipeline becomes orchestration. System asks **“What should happen next?”**
within a **fixed tool catalog** (constrained LLM — not a rigid stage pipeline).

**Architecture that ships:**

```text
Goal → Conductor (observe → think → schedule → stop)
         → Task Queue → Research Engineer / tools
research conduct "<goal>"   # product entry
analyze | plan | run | …    # power-user / debug
```

| Slice | Outcome |
|-------|---------|
| 2.1 Conductor | Decide-only brain; LLM picks/skips among registered tools |
| 2.2 Task queue | Pending / Running / Completed / Retry / Failed / Blocked |
| 2.3 Scheduler | Priorities, retries, dependencies — **separate from execution** |
| 2.4 Approvals | Plan + submit gates; approve/reject + comments → future observe |

**Usable system:** `research conduct` drives the loop; decisions + queue are durable;
stage CLIs remain stable.

**Tech that ships with M2:**

| Area | Technology |
|------|------------|
| Orchestration | **Custom** Conductor |
| Scheduler / queue | **Custom**; durable state in **SQLite** |
| State machine | Explicit enums |
| LLM (policy) | Structured NextAction via router; **LiteLLM** when cutting over |
| Observability | Decision/task/tool/approval structured logs (JSONL/DB) |

Depth: [design/04–05](design/04-conductor.md) · [02-conductor](milestones/02-conductor/).

---

## M3 — Campaign Engine (Orchestrator complete)

**Goal:** True autonomous research loop under budgets — dynamic tasks beyond the
fixed catalog, without waiting for the full agent zoo.

**Architecture that ships:**

```text
while not (goal | budget | time):
  observe → think → plan → execute → evaluate → reflect
checkpoint → research continue / pause / resume
```

Dynamic tasks (Strangler C) extend M2 `research conduct`:

```text
research conduct "<goal>" | continue | status | pause | resume
```

**Usable system:** Operator can leave a **goal** and return; workflow is no longer
limited to the fixed tool catalog.

**North star after M3:** M4 improves decide quality (context); M5 adds
delegation/parallel/events; M6 learns across campaigns — the Research OS **manager**.

**Tech that ships with M3:**

| Area | Technology |
|------|------------|
| Runtime | asyncio (+ AnyIO at boundaries) |
| Checkpoint | SQLite (+ workspace refs) |
| Distributed durability | **Not yet** — Temporal only if multi-machine appears later |

Depth: [design/06-campaigns.md](design/06-campaigns.md) · [03-campaigns](milestones/03-campaigns/).

---

## M4 — Memory & Context Engine

**Goal:** Intelligence layer — never dump full history into the model.

**Architecture that ships:**

```text
Goal + Task → retrieve → rank → compress → ContextBundle → LLM
```

Memory hierarchy ports: short-term, working, episodic, semantic (graph), long-term.
Semantic layer **keeps** Evidence Card / Research Graph — extend, don’t replace.

**Usable system:** Better Conductor/tool prompts; `explain` can cite retrieved context.

**Tech that ships with M4** (introduce only when retrieval needs it):

| Area | Technology |
|------|------------|
| Metadata | SQLite (Postgres if multi-user later) |
| Retrieval | Hybrid: filters + **BM25** + graph walk first |
| Graph engine | Stay on logical SQLite graph until it hurts → then **Kuzu** |
| Vectors | **Defer** → **Qdrant** when ANN is clearly needed |
| Embeddings | Provider via LLM router / Voyage / OpenAI / local — pick at impl time |

Depth: [design/07](design/07-context-engine.md), [design/10](design/10-memory-os.md) · [04-context](milestones/04-context/).

---

## M5 — Agent ecosystem + events + parallel

**Goal:** Break the monolith into specialists; decouple with events; run independent work concurrently.

**Architecture that ships:**

```text
Conductor → Registry (“who can solve this?”)
         → Paper / Plan / Impl / Experiment / Eval / Submit / Reflection agents
         → Tool runtime
Event bus: ExperimentCompleted, PaperAdded, … (subscribers, not hard calls)
Parallel branches → merge → reflect
```

Agents are thin:

```text
Capabilities → Task → Workspace → Tools → Artifacts → Events
```

**Usable system:** New specialty = register agent + tools; wall-clock drops on parallelizable work.

**Tech that ships with M5:**

| Area | Technology |
|------|------------|
| Agent runtime | **Custom** (not LangGraph/CrewAI) |
| Implementation coding | **Reuse** via CodingTool adapter (Claude Code / Aider / OpenHands / …) |
| LLM / structured out | LiteLLM + Pydantic / **PydanticAI** (or Instructor) |
| Event bus | **Blinker** (in-process) on M2 log → **NATS**/Redis Streams if multi-process |
| Parallel / sandbox | asyncio workers; Docker for execute_python; **Ray** only if needed |
| Multi-machine | Kubernetes later — not a M5 gate |

Depth: [design/08–09](design/08-agents.md) · [05-agents](milestones/05-agents/).

---

## M6 — Self-improving system

**Goal:** Future competitions start with prior experience — research memory, not a wiki.

**Architecture that ships:** Structured **Experience Records** (goal / hypothesis /
action / result / outcome / artifact links / tags). Context Engine retrieves similar
experiences into `ContextBundle`. Optional `research memory seed|inspect` for
human-visible warm-start. Memory influences Conductor; never silently controls it.

**Usable system:** Second competition can retrieve cross-comp experience; operators
can seed/inspect explicitly. Pattern libraries and auto-transfer with confidence are
post-M6 backlog.

**Tech that ships with M6:**

| Area | Technology |
|------|------------|
| Store | SQLite ExperienceStore (cross-competition) |
| Retrieve | M4 Context Engine + experience provider (BM25/filters/graph as available) |
| Analytics / vectors | **Defer** (DuckDB, Qdrant — backlog) |

Depth: [design/10-memory-os.md](design/10-memory-os.md) · [06-transfer-memory](milestones/06-transfer-memory/).

---

## Research Operating System (product surface)

When M3–M6 land, users think in **goals**, not stages:

```text
research "Win BirdCLEF"
research continue | pause | resume | inspect | explain
research benchmark | memory | replay
```

Legacy stage CLIs remain power-user/debug tools. CLI / IDE / API are **clients** of
one Research Runtime.

### Target shape (end state — not day-one stack)

```text
CLI / IDE / API
        → Research Conductor
        → Planner • Scheduler • Router
        → Event Bus & Task Queue
        → Specialist agents → Tool Runtime
        → Context Builder → Memory Manager (hierarchy)
        → SQLite/Postgres • (Kuzu) • (Qdrant) • Object store
```

**Custom forever:** Conductor, agent runtime, scheduler, tool *semantics*, context
engine, memory manager, workspace, event integration.  
**Reuse forever (data plane):** coding agents, browsers, sandboxes, git, LLM
gateways, vector engines.  
**Libraries when needed:** LiteLLM, Pydantic(AI), storage engines, asyncio, later
Ray/Temporal only when scale demands.

---

## Remapping (nothing wasted)

| V1 | Becomes |
|----|---------|
| Analyzer | Tool(s) |
| Planner | `generate_plan` / Planning agent |
| Research Engineer | Deterministic execution consumer / Impl+Experiment agents |
| Reflection | `reflect` / Critic agent |
| Knowledge + Evidence + Graph | Memory hierarchy (semantic + experiment) |

---

## Git

| When | Branch |
|------|--------|
| This design | `research-os-design` |
| Each milestone | `research-os-mN-…` (table above) |

One implementation branch per milestone. See [execution-plan.md](execution-plan.md).

---

## Strangler Fig

| Phase | Milestone | Visible | Internal |
|-------|-----------|---------|----------|
| A | M1 | Stage CLIs | Artifacts + tools |
| B | M2 | Familiar | Conductor fixed sequence |
| C | M3 | Familiar | Dynamic tasks |
| D | M3 | Goal CLI | Legacy CLIs remain |

---

## Design doc index

| Doc | Milestone |
|-----|-----------|
| [architecture.md](architecture.md) | Principles + target diagram |
| [execution-plan.md](execution-plan.md) | Sequence / parallel ops |
| [design/01-artifacts.md](design/01-artifacts.md) … [03-workspace.md](design/03-workspace.md) | M1 |
| [design/04-conductor.md](design/04-conductor.md) … [05-tasks.md](design/05-tasks.md) | M2 |
| [design/06-campaigns.md](design/06-campaigns.md) | M3 |
| [design/07-context-engine.md](design/07-context-engine.md) | M4 |
| [design/08-agents.md](design/08-agents.md) … [09-parallel-and-events.md](design/09-parallel-and-events.md) | M5 |
| [design/10-memory-os.md](design/10-memory-os.md) | M4 ports + M6 transfer |
| [design/11-capability-registration.md](design/11-capability-registration.md) | Post-M6 — grow tool catalog from `no_capability` gaps |

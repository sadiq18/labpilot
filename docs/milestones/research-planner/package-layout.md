# Research Planner — Package Layout

Back to [README.md](README.md) · [schema.md](schema.md) · [MILESTONES.md](../../MILESTONES.md).

**Status:** Design locked; Phase B **implemented** (Plans 1–6). Package lives at
`src/labpilot/research_engine/planner/` with shared infra in `src/labpilot/accessor/`.

---

## 1. Placement — sibling pillar under `research_engine`

```
src/labpilot/research_engine/
  intelligence/     # analyze → knowledge → brief → hypothesize  (shipped)
  planner/          # Hypothesis → ResearchPlan DAG              (this milestone)
  execution/        # future: capability executors consume DAG
```

| Option | Verdict |
|--------|---------|
| Inside `intelligence/` | **Reject** — intelligence understands & recommends; planner **compiles** a different artifact |
| Inside `execution/` | **Reject** — planner never executes; would blur plan vs run |
| Sibling `planner/` | **Choose** — matches Research OS: analyze → knowledge → **planner** → executor |

There is no separate `execution_engine` package; the service boundary is
`labpilot.research_engine` ([`__init__.py`](../../../src/labpilot/research_engine/__init__.py)).
When code lands, update that docstring to name three pillars: intelligence, planner, execution.

---

## 2. Accessor layer (shared data access) — design decision

The three pillars must not reach into each other for infrastructure. Introduce a dedicated
**accessor** layer that owns the low-level clients, with a **commons** inside it for the
SQLite schema and shared DB logic.

```
src/labpilot/accessor/
  __init__.py
  sqlite/
    __init__.py
    client.py             # SqliteClient: connect, row factory, PRAGMA, transaction ctx
    schema.sql            # UNIFIED SQLite schema — single source of record
    migrate.py            # run schema.sql idempotently; SCHEMA_VERSION
  llm/
    __init__.py
    client.py             # LLM client (provider-agnostic) — moved from labpilot/llm/
    json_utils.py         # parse_json_object for typed micro-agent output
  commons/
    __init__.py
    ids.py                # deterministic id allocators (H-/P-/… helpers)
    json_utils.py         # JSON-in-TEXT coerce/round-trip helpers
```

Why:

- **Planner needs the SQLite client + LLM client but must not import `intelligence`.** Today
  the SQLite client + `schema.sql` live under `intelligence/knowledge/` and the LLM client
  under `labpilot/llm/`. An accessor layer lets planner, intelligence, and execution share
  them without pillar-to-pillar imports.
- **Schema lives with the SQLite client.** `accessor/sqlite/schema.sql` is the single unified
  DDL (all tables: artifacts, beliefs, hypotheses, `research_plans`, `research_tasks`, …),
  next to `client.py` / `migrate.py`. No pillar owns a second database.
- **Commons holds shared logic that is not client-specific** (id allocators, JSON-in-TEXT
  helpers). Domain stores (`KnowledgeStore`, `PlanStore`) build on `SqliteClient`; they hold
  table-specific read/write logic, not connection or DDL plumbing.

Dependency direction (acyclic):

```
accessor        ← owns sqlite/ (client + schema.sql + migrate) · llm/ · commons/
   ▲   ▲   ▲
   │   │   │
intelligence  planner  execution     # pillars depend on accessor, never on each other
   ▲   ▲   ▲
   └───┴───┴──  cli / common
```

> **Migration note (not Phase A):** moving `schema.sql` + SQLite client out of
> `intelligence/knowledge/` and the LLM client out of `labpilot/llm/` is an **existing-code
> refactor** that touches Research Intelligence. It is documented here as the **target**
> architecture; the actual move is sequenced as an early Phase B implementation plan (or a
> standalone refactor) so this design phase stays docs-only. Until it lands, planner code may
> temporarily consume `intelligence/knowledge` + `labpilot/llm`, then migrate to `accessor`.

---

## 3. Import hygiene

```
accessor      ──owns──▶  SqliteClient · schema.sql · migrate · LLM client · commons helpers
intelligence  ──uses──▶  accessor (KnowledgeStore built on SqliteClient)
planner       ──uses──▶  accessor (PlanStore + LLM client)
planner       ──reads─▶  hypotheses/beliefs via KnowledgeStore/HypothesisStore APIs
planner       ──writes▶  research_plans / research_tasks via PlanStore
execution     ──reads─▶  planner schemas / PlanStore only (future)
intelligence  ✖ does not import planner
execution     ✖ does not import intelligence
planner       ✖ does not import intelligence for infrastructure (uses accessor)
```

- DB DDL lives in `accessor/sqlite/schema.sql` (single SoR) once the refactor lands.
- Planner does **not** own a second database.
- CLI (`labpilot.cli`) may import `planner`, `intelligence`, and `accessor`; no cycles.

---

## 4. Target directory tree

```
src/labpilot/research_engine/planner/
  __init__.py
  planner.py                 # compiler driver: compile_research_plan()
  context_builder.py         # Structured State → PlanningContext
  retrieval.py               # bounded load: hypothesis, beliefs, brief, experiment snippets
  validator.py               # DAG validate (unique ids, deps resolve, acyclic)
  optimizer.py               # light rewrites / topo helpers (MVP: merge defaults + levels)
  scheduler.py               # order_index + parallel levels from DAG (deterministic)
  serializer.py              # ResearchPlan ↔ DB rows / JSON / markdown projections
  templates.py               # rule_engine DAG templates (SpecAugment-style, …)
  schemas/
    __init__.py
    models.py                # ResearchPlan, ResearchTask, verification, retry, statuses
    task_types.py            # TaskType, RuntimeTarget enums
  prompts/
    planning_engine_system.md
    planning_engine_user.md  # (or .txt / .jinja — pick one in Phase B)
  micro_agents/
    planning_engine/
      __init__.py
      agent.py               # ResearchPlannerAgent — ONE LLM | rule_engine
      skill.md
    # later (optional helpers — stubs ok in early plans):
    # risk_checker/
    # dependency_checker/
    # evidence_summary/
```

Map to the compiler stages in [README.md](README.md):

| Stage | Module(s) |
|-------|-----------|
| Structured State + Knowledge Retrieval | `retrieval.py`, `context_builder.py` |
| Optional micro-helpers | `micro_agents/*` (stubs / later) |
| Planning Engine (ONE LLM) | `micro_agents/planning_engine/` |
| Lowering (ids, verification defaults) | `planner.py` + `templates.py` defaults |
| Validation / Optimizer / Scheduler | `validator.py`, `optimizer.py`, `scheduler.py` |
| Emit | `serializer.py` + `PlanStore` (on `accessor` SqliteClient) |

---

## 5. Compiler entry point

```python
def compile_research_plan(
    hypothesis: Hypothesis,
    *,
    store: KnowledgeStore | None = None,
    llm_client: object | None = None,
) -> ResearchPlan:
    """Hypothesis → validated ResearchPlan (DAG). Never executes tasks."""
    ...
```

CLI `research plan create` loads the hypothesis, calls `compile_research_plan`, then
`upsert_plan` + serializer projections. Soft-fail: invalid LLM draft → `rule_engine`
template path (same posture as `ResearchBriefAgent`).

Micro Agents use the shared contract in `labpilot.accessor.common.micro_agents`
(`StructuredContext` in → typed Pydantic out; no memory, no loops, no side effects). The
Planning Engine's LLM client comes from `labpilot.llm`.

---

## 6. What stays outside this package

| Concern | Where |
|---------|--------|
| SQLite client · unified `schema.sql` · migrate · shared helpers | `accessor/` (`sqlite/`, `common/`) |
| Provider-agnostic LLM (OpenAI / Gemini / Ollama + router/cache) | `labpilot.llm` |
| Analyzers, Knowledge Hub, Brief, Hypothesis Assistant | `intelligence/` |
| `PlanStore` table-specific read/write (built on `accessor` SqliteClient) | `planner/` (this package) |
| Capability executors (`WRITE_CODE` runner, train, …) | `execution/` (future) |
| `ImprovementPlan` / `research improve` | existing `improvement/` |
| Pipeline stages / runs | `orchestrator/`, `runs/` |

The planner package must not import `improvement`, must not start pipeline runs, and must not
import `intelligence` for infrastructure — it reaches SQLite through `accessor` and LLM
through `labpilot.llm`.

---

## 7. Tests (Phase B preview)

When implementation starts:

- Unit: validator (cycle / missing dep / topo levels)
- Unit: rule_engine templates produce expected task types + verification
- Unit: PlanStore round-trip (upsert → get reconstructs deps)
- Unit/CLI: `plan create` writes DB + derived files; **no** `runs/` dirs created

Place tests under `tests/unit/` (e.g. `test_research_planner.py`) following existing
intelligence test style.

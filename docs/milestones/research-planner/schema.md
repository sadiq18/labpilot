# Research Planner — Schema

Back to [README.md](README.md) · [MILESTONES.md](../../MILESTONES.md).

**Status:** Design Phase A. No `schema.sql` changes until Phase B implementation plans are
approved.

This doc specifies the durable plan/task model: SQLite tables in `knowledge.db`, Pydantic
shapes, PlanStore API, and derived file projections.

---

## 1. Naming collision — keep Layer-3 `tasks`

`knowledge.db` already has a **`tasks`** table: Layer-3 merged knowledge objects (ML problem
types such as "Audio Classification"), via `_ENTITY_TABLES["task"]` in the Knowledge Store.

That is **not** a planner execution node.

| Table | Meaning | Action |
|-------|---------|--------|
| `tasks` | Knowledge ontology: named task/problem types | **Keep as-is; do not overload** |
| `research_plans` | One compiled plan (usually from one hypothesis) | **New** |
| `research_tasks` | DAG nodes (`WRITE_CODE`, `RUN_TRAINING`, …) | **New** |
| `research_task_deps` | DAG edges (`task_id` depends on `depends_on`) | **New** |

Do not rename `tasks` (breaks Knowledge Hub / retrieval / fixtures). Do not store plan nodes
in `tasks`. The `research_*` prefix mirrors `research_artifacts` and keeps the planner
namespace obvious.

---

## 2. SQLite DDL (additive; future `SCHEMA_VERSION` bump)

Tables live in the existing per-competition DB:
`knowledge/<slug>/research/knowledge.db` (same file as artifacts / beliefs / hypotheses).
Migration stays `CREATE TABLE IF NOT EXISTS`, run idempotently by the shared migrator.

**DDL home:** these tables belong in the **unified** schema. Target location is
`accessor/sqlite/schema.sql` (single source of record, next to the SQLite client + migrator)
once the accessor-layer refactor lands (see [package-layout.md §2](package-layout.md)).
Until then they can be added to today's `intelligence/knowledge/schema.sql` and migrate with
the rest. Either way there is **one** `schema.sql` and one migrator — the planner never owns
a second database.

```sql
CREATE TABLE IF NOT EXISTS research_plans (
    id                 TEXT PRIMARY KEY,
    competition_slug   TEXT,
    hypothesis_id      TEXT,
    goal               TEXT NOT NULL DEFAULT '',
    current_state      TEXT NOT NULL DEFAULT '',
    expected_outcome   TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'draft',   -- draft|ready|in_progress|done|abandoned
    priority           INTEGER NOT NULL DEFAULT 0,
    estimated_gain     REAL NOT NULL DEFAULT 0.0,
    risk               TEXT NOT NULL DEFAULT '',
    estimated_cost     REAL,                            -- hook; null in early MVP
    estimated_duration TEXT,                            -- hook
    runtime_target     TEXT,                            -- hook: local|docker|kaggle|cpu|p100|a100
    success_criteria   TEXT NOT NULL DEFAULT '[]',      -- JSON list
    rollback           TEXT NOT NULL DEFAULT '',
    artifacts          TEXT NOT NULL DEFAULT '[]',      -- JSON list
    generated_by       TEXT NOT NULL DEFAULT 'rule_engine',  -- llm|rule_engine
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_comp ON research_plans(competition_slug);
CREATE INDEX IF NOT EXISTS idx_plans_hyp  ON research_plans(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_plans_status ON research_plans(status);

CREATE TABLE IF NOT EXISTS research_tasks (
    id             TEXT PRIMARY KEY,
    plan_id        TEXT NOT NULL REFERENCES research_plans(id) ON DELETE CASCADE,
    parent_task_id TEXT REFERENCES research_tasks(id) ON DELETE SET NULL,
    task_type      TEXT NOT NULL,                       -- TaskType enum value
    description    TEXT NOT NULL DEFAULT '',
    inputs         TEXT NOT NULL DEFAULT '[]',          -- JSON list
    outputs        TEXT NOT NULL DEFAULT '[]',          -- JSON list
    status         TEXT NOT NULL DEFAULT 'pending',     -- pending|running|done|failed|skipped
    verification   TEXT NOT NULL DEFAULT '{}',          -- JSON TaskVerification
    retry_policy   TEXT NOT NULL DEFAULT '{}',          -- JSON RetryPolicy
    order_index    INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL,
    estimated_time TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_tasks_plan ON research_tasks(plan_id);
CREATE INDEX IF NOT EXISTS idx_research_tasks_status ON research_tasks(status);

-- DAG edges: task_id depends on depends_on (must complete first).
CREATE TABLE IF NOT EXISTS research_task_deps (
    task_id    TEXT NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    depends_on TEXT NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, depends_on),
    CHECK (task_id != depends_on)
);
CREATE INDEX IF NOT EXISTS idx_research_task_deps_on ON research_task_deps(depends_on);
```

### ID allocation

| Entity | Pattern | Example |
|--------|---------|---------|
| Plan | `P-<n>` zero-padded | `P-001` |
| Task | `<plan_id>-T<n>` | `P-001-T01` |

Allocator mirrors `HypothesisStore` (`H-001` style): scan existing ids, next integer.

### Status enums

**PlanStatus:** `draft` | `ready` | `in_progress` | `done` | `abandoned`

**TaskStatus:** `pending` | `running` | `done` | `failed` | `skipped`

Compiler v1 typically writes plans as `draft` or `ready`. Transitions to `in_progress` /
task `running` belong to a future executor — schema supports them now.

---

## 3. Task types (instruction set)

~15 typed instructions. The Planning Engine chooses among these; the future executor
dispatches on them. The planner **never** performs the side effect.

| TaskType | Value | Intent |
|----------|-------|--------|
| `READ_CODE` | `read_code` | Inspect existing code/paths |
| `WRITE_CODE` | `write_code` | Request code change (node only) |
| `MODIFY_CONFIG` | `modify_config` | Request config change |
| `INSTALL_PACKAGE` | `install_package` | Dependency install |
| `RUN_UNIT_TEST` | `run_unit_test` | Unit tests |
| `RUN_SMOKE_TEST` | `run_smoke_test` | Short smoke / integration check |
| `RUN_TRAINING` | `run_training` | Train (smoke or full — described in inputs) |
| `RUN_INFERENCE` | `run_inference` | Inference / predict |
| `BUILD_SUBMISSION` | `build_submission` | Submission artifact |
| `EVALUATE` | `evaluate` | Metrics / CV eval |
| `COMPARE` | `compare` | Diff vs baseline / parent run |
| `GENERATE_REPORT` | `generate_report` | Human-readable report |
| `UPDATE_BELIEF` | `update_belief` | Belief store update |
| `CREATE_HYPOTHESIS` | `create_hypothesis` | New hypothesis from findings |
| `REFLECT` | `reflect` | Structured reflection |

Per-type **verification / retry defaults** live in compiler templates (deterministic merge
after the Planning Engine draft).

---

## 4. Pydantic shapes (design)

Sketch — exact module paths in [package-layout.md](package-layout.md).

```python
class TaskVerification(BaseModel):
    expected_output: str = ""
    check: str = ""              # what "success" means
    failure_recovery: str = ""   # e.g. restore config; abort after N failures


class RetryPolicy(BaseModel):
    max_retries: int = 0
    abort_on_failure: bool = True


class ResearchTask(BaseModel):
    id: str
    plan_id: str
    parent_task_id: str | None = None
    type: TaskType
    description: str = ""
    inputs: list[str] = []
    outputs: list[str] = []
    dependencies: list[str] = []   # task ids; persisted to research_task_deps
    status: TaskStatus = TaskStatus.PENDING
    order: int = 0
    verification: TaskVerification = TaskVerification()
    retry_policy: RetryPolicy = RetryPolicy()
    estimated_cost: float | None = None
    estimated_time: str | None = None


class ResearchPlan(BaseModel):
    id: str
    competition: str
    hypothesis_id: str
    goal: str = ""
    current_state: str = ""
    expected_outcome: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    priority: int = 0
    estimated_gain: float = 0.0
    risk: str = ""
    estimated_cost: float | None = None
    estimated_duration: str | None = None
    runtime_target: RuntimeTarget | None = None
    tasks: list[ResearchTask] = []
    success_criteria: list[str] = []
    artifacts: list[str] = []
    rollback: str = ""
    generated_by: Literal["llm", "rule_engine"] = "rule_engine"
    created_at: datetime
    updated_at: datetime
    notes: list[str] = []
```

`parent_task_id` groups/sequences; **`dependencies` / `research_task_deps`** carry the true
DAG (parallel branches). Validator rejects missing deps and cycles.

---

## 5. PlanStore API

DB is the **source of truth**. `PlanStore` holds the plan/task read/write logic and is built
on the shared `accessor` SQLite client (same JSON-in-TEXT + commit conventions as
`upsert_hypothesis` / belief APIs). It lives in the `planner/` package, not in `intelligence/`
— the planner reaches SQLite through `accessor`, not by importing the Knowledge Store's
pillar. (Pre-refactor, it may temporarily extend `KnowledgeStore`; post-refactor it uses the
accessor client directly.)

| Method | Behavior |
|--------|----------|
| `upsert_plan(plan: ResearchPlan) -> str` | One transaction: plan row + replace tasks + replace deps |
| `get_plan(plan_id) -> ResearchPlan \| None` | Reassemble plan + tasks + dep edges |
| `list_plans(*, status=None, hypothesis_id=None)` | Filter + order |
| `update_plan_status(plan_id, status)` | Lifecycle for future executor |
| `update_task_status(task_id, status)` | Lifecycle for future executor |

Hypothesis file store (`HypothesisStore`) remains authoritative for hypothesis JSON files;
plans are **DB-first** (with optional JSON/MD projections — below). No second SQLite.

---

## 6. Derived projections (not primary)

Under `knowledge/<slug>/research/plans/` (add `plans_dir` on `ResearchPaths`; include in
`ensure()`):

| File | Role |
|------|------|
| `<plan_id>.json` | Structured projection for diff/inspect |
| `<plan_id>.md` | Human-readable, **generated from** the DAG |

Serializer always derives these **from** the `ResearchPlan` model / DB rows. Never treat
markdown as the SoR. Never invent plan content only in prose.

---

## 7. Verification examples (per task)

| Task | Verification (check) | Failure recovery |
|------|----------------------|------------------|
| `MODIFY_CONFIG` | Config loads successfully | Restore previous version |
| `RUN_UNIT_TEST` | Exit 0; required tests pass | Fix via new WRITE_CODE task / abort |
| `RUN_TRAINING` | Loss decreases (or finite metrics) | Abort after N failures |
| `COMPARE` | Metric delta recorded vs baseline | Mark inconclusive; skip gated train |

Plan-level `success_criteria` and `rollback` summarize the whole experiment intent.

---

## 8. Hooks for later modules (columns exist; unused by early compiler)

- `estimated_cost`, `estimated_duration`, `runtime_target` on plans
- `estimated_cost`, `estimated_time` on tasks
- `priority` for ranking multiple plans under a future budget allocator

Formulas and schedulers that fill these are **out of Phase A / early MVP**; the schema is
ready so later modules do not reshape tables.

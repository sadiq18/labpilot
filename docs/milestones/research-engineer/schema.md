# Research Engineer — Schema

Back to [README.md](README.md) · Planner schema: [../research-planner/schema.md](../research-planner/schema.md).

**Status:** Design Phase A. DDL changes land in Phase B via
`accessor/sqlite/schema.sql` (single SoR) + `SCHEMA_VERSION` bump.

---

## 1. Principles

- **Reuse** existing planner tables (`research_plans`, `research_tasks`,
  `research_task_deps`) — extend, do not fork.
- **Reuse** DB `experiments` for experiment result mirrors where possible — do **not**
  invent a parallel `experiment_runs` table unless Design A review finds a hard gap.
- **Do not** store execution nodes in Layer-3 knowledge `tasks`.
- Disk workspace (`runs/` or a successor under the execution) remains the rich artifact
  home; DB holds joins, status, and evidence indexes.

---

## 2. Existing planner tables (keep)

### `research_plans`

Already has lifecycle statuses suitable for the Engineer:
`draft | ready | in_progress | done | abandoned`.

Engineer transitions: `ready` → `in_progress` → `done` / `abandoned`.

Baseline plans set `metadata.plan_kind = "baseline"` (and usually no / empty
`hypothesis_id`).

### `research_tasks`

Already: `status` (`pending|running|done|failed|skipped`), `verification`,
`retry_policy`, `inputs`/`outputs`, `order_index`.

### Proposed additive columns (Phase B — only if needed)

Prefer **metadata JSON** first; promote to columns when queried heavily.

| Field | Where | Intent |
|-------|-------|--------|
| `started_at` / `completed_at` | column or metadata | Task attempt timing |
| `error` | column or metadata | Last failure summary |
| `evidence` | metadata / side table | Pointers to evidence blobs |
| `attempt` | metadata | Retry count for this task |
| `capability` | metadata | Which capability handled it |

If columns are added, keep them nullable and migrate with `CREATE TABLE IF NOT EXISTS`
style additive scripts / version bump — same migrator as today.

### `research_task_deps`

Unchanged. Engineer schedules from topological levels.

---

## 3. Execution attempts (new durable entity)

One plan may run multiple times. Introduce an **execution** record (name TBD in
implementation: `research_executions` or `plan_executions`).

Sketch:

```sql
CREATE TABLE IF NOT EXISTS research_executions (
    id                 TEXT PRIMARY KEY,          -- E-001
    plan_id            TEXT NOT NULL REFERENCES research_plans(id),
    competition_slug   TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    -- pending|running|succeeded|failed|cancelled
    workspace_path     TEXT,                      -- disk root for this attempt
    runtime_target     TEXT,
    experiment_id      TEXT,                      -- link to experiments.id when done
    error              TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    started_at         TEXT,
    completed_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_exec_plan ON research_executions(plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_status ON research_executions(status);
```

CLI: `research resume --execution E-001`.

Task-level status remains on `research_tasks` (or per-execution task state in metadata /
join table if concurrent executions must not clobber — **default MVP:** one active
execution per plan; task statuses update in place; historical snapshots in evidence /
execution metadata).

---

## 4. Reuse `experiments`

Existing table (knowledge mirror):

| Column | Use for Engineer results |
|--------|---------------------------|
| `id` | Experiment / result id |
| `competition_slug` | From plan |
| `summary` | Short outcome narrative |
| `outcome` | e.g. `improved` / `regressed` / `baseline` / `failed` |
| `metrics` | JSON metrics |
| `techniques` | JSON list |
| `metadata` | `plan_id`, `execution_id`, `hypothesis_id`, artifact paths, deltas |

Do **not** confuse with the Pydantic filesystem `Experiment` graph model
(`experiments/models.py`) — that remains the run-tree view. Engineer writes:

1. Disk workspace artifacts (checkpoints, logs, submission, report)
2. A row in DB `experiments` as the durable join for intelligence / ranking
3. Optional dual-write into the filesystem experiment graph conventions if still needed
   for Scientist dashboards (Phase B decides; prefer one clear SoR)

---

## 5. Evidence

Every task attempt produces structured evidence (pass/fail, checks, logs paths, metrics
snippets). Storage options (pick in Phase B, prefer simplest):

1. JSON files under `…/executions/E-xxx/evidence/<task_id>.json` + path in task metadata  
2. Optional `research_task_evidence` table if SQL queries become first-class  

Evidence feeds Verification Engine, Recovery, and the final Experiment Artifact.

---

## 6. ID allocation

| Entity | Pattern | Example |
|--------|---------|---------|
| Plan | `P-<n>` (existing) | `P-001` |
| Task | `<plan_id>-T<n>` (existing) | `P-001-T01` |
| Execution | `E-<n>` | `E-001` |
| Experiment row | reuse existing conventions or `exp_<slug>_<n>` | TBD Phase B |

---

## 7. Status machine (summary)

**Plan:** `ready` → `in_progress` → `done` | `abandoned`  

**Execution:** `pending` → `running` → `succeeded` | `failed` | `cancelled`  

**Task:** `pending` → `running` → `done` | `failed` | `skipped`  

Skipped: gated tasks whose success_criteria / upstream compare failed (e.g. full train
after smoke compare fails).

---

## 8. Non-goals for schema

- Second SQLite database
- Overloading Layer-3 `tasks`
- Replacing `research_*` planner tables with a new parallel DAG store
- Storing full training logs in SQLite TEXT columns

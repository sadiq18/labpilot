# Plan 1 — Experiment Graph

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** nothing (foundational). **Unlocks:** Plans 2, 3, 6, 7, 8.

---

## Goal

Turn "a folder of runs" into a real graph: every experiment knows its parent, its children,
and enough of its own state (config, metrics, git commit, runtime) to be compared, ranked, or
displayed — without inventing a new persisted "database." This is Task 1 in the brief and the
foundation everything else in Milestone 2 reads from.

## Why this matters

Today, `research improve` sets `parent_run_id` in `manifest.json`, but nothing ever asks "what
are the children of run X?" — you'd have to grep `runs/*/manifest.json` by hand. And nothing
stops a second `research improve --run-id X` from producing a *second* child of X today (the
fork logic in `improvement/fork.py` has no uniqueness constraint on `parent_run_id`), but
nothing renders that branching structure either. This plan makes the branching tree the brief
asks for (`Baseline → {Mixup, Focal Loss} → ... → Ensemble`) a first-class, queryable thing.

## Current state

- `orchestrator/manifest.py` — `RunManifest.metadata` is a free-form `dict[str, Any]`. Only
  `improvement/fork.py` ever writes `parent_run_id`, `iteration`, `improvement_strategy` into
  it. No `git_commit` field exists anywhere in the codebase today (confirmed: no
  `git rev-parse` / `git_commit` usage in `src/labpilot/`).
- `tracking/index.py:scan_runs()` returns a flat `list[RunIndexEntry]` per competition — no
  parent → children reverse index, no traversal helpers, no rendering.
- `training/runner.py` writes stage timing implicitly via `StageRecord.started_at` /
  `finished_at` on the `train_model` stage in `manifest.json` — runtime is derivable today, it
  is just never surfaced as an "experiment" field.

## Design

### 1. `Experiment` — assembled, not stored

`Experiment` is **not** a new file written to `runs/<id>/`. It's a read-side model assembled
from artifacts that already exist, so there is exactly one writer for every field (avoids the
two-sources-of-truth bug class). Only the fields that genuinely don't exist yet need a small
addition to an existing writer:

```python
class Experiment(BaseModel):
    id: str                              # = run_id
    competition: str
    status: str                          # from manifest.status
    parent_id: str | None                # manifest.metadata["parent_run_id"]
    children_ids: list[str]              # COMPUTED: other runs whose parent_id == self.id
    iteration: int                       # manifest.metadata.get("iteration", 0)
    hypothesis_id: str | None            # manifest.metadata.get("hypothesis_id")  (Plan 2)
    git_commit: str | None               # NEW — see below
    template_name: str | None            # baseline_choice.json
    problem_type: str | None             # baseline_choice.json
    model_params: dict[str, Any]         # training_overrides.json (child) or {} (root)
    feature_recipes: list[str]           # training_overrides.json
    metrics: dict[str, float]            # experiment/record.json (or metrics.json fallback)
    public_score: float | None           # submission_result.json
    runtime_seconds: float | None        # COMPUTED from train_model stage started/finished_at
    artifacts: list[str]                 # experiment/record.json
    reflection_path: str | None          # runs/<id>/reflection.md if present
    created_at: datetime                 # manifest.created_at
```

`git_commit`: capture `git rev-parse HEAD` (best-effort, `None` outside a git checkout or if
`git` isn't on `PATH`) once, in `orchestrator/pipeline.py` at manifest creation for `run` /
`init`, and in `improvement/fork.py` for child runs. This is the only new *write* in this
plan — everything else is read-side aggregation. Non-fatal by design: wrap in try/except like
the existing LLM-client-unavailable fallbacks elsewhere in the codebase.

### 2. `ExperimentGraph`

```python
class ExperimentGraph:
    competition: str
    nodes: dict[str, Experiment]

    @property
    def roots(self) -> list[Experiment]: ...        # parent_id is None
    def children(self, run_id: str) -> list[Experiment]: ...
    def ancestors(self, run_id: str) -> list[Experiment]: ...   # root-ward walk
    def descendants(self, run_id: str) -> list[Experiment]: ... # leaf-ward walk
    def best_path(self, metric_key: str) -> list[Experiment]:   # "current best pipeline"
        """Root-to-leaf path maximizing (or minimizing) `metric_key` at each branch point."""
    def to_tree_text(self, metric_key: str | None = None) -> str:
        """ASCII tree, mirroring the brief's `Baseline → {Mixup, Focal Loss} → ...` diagrams."""
```

`build_graph(runs_dir: Path, competition: str) -> ExperimentGraph` scans `runs_dir` exactly
like `tracking/index.py:scan_runs()` does today (same directory-walk, same manifest load), so
it can reuse or directly replace `scan_runs`'s traversal loop — decide at implementation time
whether `scan_runs` becomes a thin wrapper that returns `RunIndexEntry` built from `Experiment`
(keeps `research runs diff` unchanged) or stays fully independent. Recommendation: make
`RunIndexEntry` derivable from `Experiment` to avoid maintaining two scanners.

`best_path()` needs the competition's metric direction (maximize vs minimize) — reuse
`competition.json`'s `MetricSpec` (already has this concept per `evaluation/metrics.py`).

### 3. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/__init__.py` | new package |
| `src/labpilot/experiments/models.py` | new — `Experiment` (this plan); other models added by later plans |
| `src/labpilot/experiments/graph.py` | new — `build_graph`, `ExperimentGraph` |
| `src/labpilot/orchestrator/pipeline.py` | +git commit capture at manifest creation |
| `src/labpilot/improvement/fork.py` | +git commit capture for child runs |
| `src/labpilot/tracking/index.py` | `scan_runs()` internals reuse `graph.py` (optional, low-risk refactor) |
| `src/labpilot/cli/main.py` | + `experiments_app` with `graph` and `show` subcommands |

### 4. CLI

```
research experiments graph --competition <slug> [--metric cv_macro_f1]
research experiments show <run_id> [--format json|table]
```

`graph` renders `ExperimentGraph.to_tree_text()`; `--metric` annotates each node with its score
and highlights `best_path()`. `show` prints one `Experiment` (rich table, or `--format json`
for scripting / for the other plans' tests).

## Non-goals

- No new way to *create* branches — `research improve` remains the only fork trigger. Multiple
  children of one parent already works structurally today; this plan only makes it visible.
- No cross-competition graph. One `ExperimentGraph` = one competition slug.
- No persistence of the graph itself — it's rebuilt from disk on every call. If this becomes
  slow at "142 experiments" scale, an on-disk cache is a follow-up, not a Milestone-2
  requirement (142 manifest reads is milliseconds; revisit at 10k+ runs).

## Open questions

1. Do we backfill `git_commit: null` semantics for pre-Milestone-2 runs, or leave old runs
   without the field and have `Experiment.git_commit` be optional everywhere? → **Optional
   everywhere**; no migration script for historical runs.
2. Should `scan_runs()` be refactored to sit on top of `graph.py` in this same PR, or deferred?
   → Prefer doing it in this PR since it's small and removes a duplicate directory walk, but
   it's not required for Plan 1's acceptance criteria below.

## Acceptance criteria

- Given a competition with a root run and two direct children (simulating
  `research improve` run twice against the same parent) plus one grandchild, `build_graph()`
  produces correct `roots`, `children()`, `ancestors()`, `descendants()` for every node.
- `research experiments graph --competition <slug>` prints a tree matching the actual
  parent/child structure, annotated with status and (if `--metric` given) score.
- `research experiments show <run_id>` includes `git_commit` for any run created after this
  plan ships, and `None` (not a crash) for runs created before it.
- `ExperimentGraph.best_path("cv_accuracy")` picks the higher-scoring child at each branch
  point on a small fixture graph with a known-best path.

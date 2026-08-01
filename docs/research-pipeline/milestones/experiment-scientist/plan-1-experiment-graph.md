# Plan 1 — Experiment Graph

Back to [Milestone 2](README.md).

**Status:** Shipped. **Depends on:** nothing (foundational). **Unlocks:** Plans 2, 3, 6, 7, 8.

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

- `experiments/manifest.py` — `RunManifest.metadata` is a free-form `dict[str, Any]`. Only
  `improvement/fork.py` ever writes `parent_run_id`, `iteration`, `improvement_strategy` into
  it. No `git_commit` field exists anywhere in the codebase today (confirmed: no
  `git rev-parse` / `git_commit` usage in `src/labpilot/`).
- `tracking/index.py:scan_runs()` returns a flat `list[RunIndexEntry]` per competition — no
  parent → children reverse index, no traversal helpers, no rendering.
- `training/runner.py` writes stage timing implicitly via `StageRecord.started_at` /
  `finished_at` on the `train_model` stage in `manifest.json` — runtime is derivable today, it
  is just never surfaced as an "experiment" field.
- **Config is never snapshotted per run.** `orchestrator/pipeline.py` reads `config: AppConfig`
  at every stage but never writes it to `runs/<id>/`. If `configs/default.yaml` changes later
  (e.g. `training.cv_folds` 5→10), there is no way to recover what a historical run actually
  used. `AppConfig` already excludes secrets from serialization today (`llm.api_key`,
  `kaggle.api_token`/`username`/`key` all use `Field(exclude=True)`, confirmed in
  `config.py`), so `config.model_dump_json()` is already safe to write to disk as-is — this is
  a missing *write*, not a missing safeguard.
- **`Experiment.artifacts` would under-report if sourced from `experiment/record.json`.**
  `Pipeline._log_experiment` hardcodes `artifacts=[submission.csv, oof.csv]` — nothing else.
  Worse, `log_experiment` runs *before* `write_reflection` and `write_report` in the default
  stage order (see `ARCHITECTURE.md`'s Stage Sequence), so even fixing that hardcoded list
  in-place could never include `reflection.md` or `report.html` for a normal run. `artifacts`
  needs to be computed at read time (see below), not trusted from that file.

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
    progress: str                        # NEW — see below, e.g. "8/14 stages"
    description: str                     # NEW — see below, one-line human summary
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
    config_snapshot: dict[str, Any]      # NEW — runs/<id>/config.json if present, else {}
    artifacts: list[str]                 # NEW SOURCE — COMPUTED, see below (not experiment/record.json)
    reflection_path: str | None          # runs/<id>/reflection.md if present
    report_path: str | None              # NEW — runs/<id>/report.html if present (Milestone 1)
    # reflection: StructuredReflection | None  — added by Plan 4 once reflection.json exists
    created_at: datetime                 # manifest.created_at
```

`git_commit`: capture `git rev-parse HEAD` (best-effort, `None` outside a git checkout or if
`git` isn't on `PATH`) once, in `orchestrator/pipeline.py` at manifest creation for `run` /
`init`, and in `improvement/fork.py` for child runs. This is the only new *write* in this
plan — everything else is read-side aggregation. Non-fatal by design: wrap in try/except like
the existing LLM-client-unavailable fallbacks elsewhere in the codebase.

`progress`: computed from `manifest.stages`, not a new write — `f"{completed}/{total} stages"`
where `completed` counts `StageStatus.COMPLETED` + `StageStatus.SKIPPED` records and `total` is
`len(manifest.stages)`. This is deliberately a plain string, not a percentage or progress bar —
it's meant to answer "how far did this experiment get", including mid-run/failed experiments,
not to power a live progress UI (see Plan 8's non-goals: no live-updating dashboard). Pair with
`status` (`running` / `completed` / `failed` / `partial`) for the full picture — `progress`
answers "how much", `status` answers "in what state".

`description`: a one-line human-readable summary, assembled from data that already exists —
**no LLM call, no new field to author by hand**:

- **Root runs** (`parent_id is None`): `f"{template_name} baseline for {competition}"` from
  `baseline_choice.json` (e.g. `"tabular_classification baseline for titanic"`).
- **Child runs**: `improvement_plan.json.rationale` if non-empty (this field **already exists**
  on `ImprovementPlan`, populated by the LLM planner or a deterministic strategy blurb — see
  `improvement/planner.py`) — otherwise fall back to
  `f"{improvement_strategy} iteration on {parent_id}"`.
- Once Plan 2 (`Hypothesis`) ships, a run with `hypothesis_id` set should prefer
  `hypothesis.prediction` as the description — this plan ships the fallback chain above so
  `description` is never blank even before Plan 2 exists.

`config_snapshot`: a **new artifact**, `runs/<id>/config.json`, written once — same timing as
`git_commit` — at manifest creation in `orchestrator/pipeline.py` (`run`/`init`) and in
`improvement/fork.py` for child runs (deliberately **not** copied from the parent: a child run
snapshots whatever config was actually resolved for the `improve` invocation that created it,
which may legitimately differ from the parent's if `configs/default.yaml` changed meanwhile —
that's the point of tracking it per-run). Content is `config.model_dump_json(indent=2)` —
`AppConfig`'s secret fields (`llm.api_key`, `kaggle.api_token`/`username`/`key`) already use
`Field(exclude=True)` today, so no new redaction logic is needed, only the write call itself.
Non-fatal by design, same as `git_commit`: if writing fails for any reason, log and continue
rather than fail the run. Read side: `Experiment.config_snapshot` is `json.loads(config.json)`
if the file exists, else `{}` — `{}`, not `None`, so downstream dotted-path lookups (Plan 7)
don't need a null check.

`artifacts`: **computed at read time**, not read from `experiment/record.json.artifacts`. Scan
`run_dir` for the fixed set of well-known relative paths already documented in
`ARCHITECTURE.md`'s Run Artifact Layout (`competition.json`, `profile.json`, `profile.md`,
`brief.md`, `baseline_choice.json`, `training_overrides.json`, `improvement_plan.json`,
`config.json`, `pipeline/train.py`, `pipeline/config.yaml`, `models/`, `oof.csv`,
`metrics.json`, `submission.csv`, `kernel/`, `submission_result.json`, `training.log`,
`experiment/record.json`, `reflection.md`, `report.html`) and include only the ones that
actually exist for that run. This fixes the `log_experiment`-runs-before-`write_report`
ordering problem above for free — the scan happens whenever `Experiment` is assembled (i.e.
whenever `research experiments show`/`graph`/`report` is invoked), by which point a completed
run has every later-stage artifact on disk too. `experiment/record.json`'s own `artifacts`
field is left as-is (out of scope for this plan) — it's a pre-existing, narrower field used
elsewhere and not something Plan 1 needs to change to get a correct `Experiment.artifacts`.

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
| `src/labpilot/experiments/graph.py` | new — `build_graph`, `ExperimentGraph`, artifact-path scan for `Experiment.artifacts` |
| `src/labpilot/orchestrator/pipeline.py` | +git commit capture at manifest creation; +write `config.json` snapshot |
| `src/labpilot/improvement/fork.py` | +git commit capture for child runs; +write `config.json` snapshot for the fork's own config |
| `src/labpilot/tracking/index.py` | `scan_runs()` internals reuse `graph.py` (optional, low-risk refactor) |
| `src/labpilot/cli/main.py` | + `experiments_app` with `graph` and `show` subcommands |

### 4. CLI

```
research experiments graph --competition <slug> [--metric cv_macro_f1]
research experiments show <run_id> [--format json|table]
```

`graph` renders `ExperimentGraph.to_tree_text()`; `--metric` annotates each node with its score
and highlights `best_path()`. `show` prints one `Experiment` (rich table, or `--format json`
for scripting / for the other plans' tests) — including `progress`, `description`, and
`report_path` so a single experiment's state and a link to its full per-run HTML report
(Milestone 1's `report.html`) are visible without opening `manifest.json` by hand. Plan 8's
dashboard is the flat, all-experiments view of the same fields; this command is the
single-experiment view.

## Non-goals

- No new way to *create* branches — `research improve` remains the only fork trigger. Multiple
  children of one parent already works structurally today; this plan only makes it visible.
- No cross-competition graph. One `ExperimentGraph` = one competition slug.
- No persistence of the graph itself — it's rebuilt from disk on every call. If this becomes
  slow at "142 experiments" scale, an on-disk cache is a follow-up, not a Milestone-2
  requirement (142 manifest reads is milliseconds; revisit at 10k+ runs).

## Open questions (resolved)

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
- Every `Experiment.description` is non-empty for both a root run and a `research improve`
  child run on the standard test fixtures, without calling an LLM.
- `Experiment.progress` matches `f"{n}/{total}"` for a manually constructed manifest with a
  mix of `completed`, `skipped`, and `pending` stages (including a still-`running` experiment,
  where `progress` reports partial completion instead of erroring).
- `Experiment.config_snapshot` round-trips real `AppConfig` values (e.g. `training.cv_folds`,
  `llm.provider`) for a run created after this plan ships, and is `{}` (not a crash) for a run
  created before it (no `config.json` on disk).
- `runs/<id>/config.json` on disk never contains `api_key`, `api_token`, `username`, or `key`
  values, for a fixture run created with real-looking (fake) credentials set.
- A child run created via `research improve` gets its **own** `config.json`, distinct from its
  parent's, when the two are created under different `--config` files in the test fixture.
- `Experiment.artifacts` for a fully completed run (through `write_report`) includes
  `reflection.md` and `report.html`, even though `log_experiment` — the stage that used to be
  the (incomplete) source of this field — runs before either of those stages.

Formalizing the two resolved open questions above:

- **No backfill (open question 1):** `assemble_experiment()`/`build_graph()` are strictly
  read-only with respect to `manifest.json` — given a fixture run directory whose
  `manifest.json` has no `git_commit` key (simulating a pre-Milestone-2 run), reading it via
  either function returns `Experiment.git_commit is None` *and* leaves `manifest.json`'s bytes
  on disk unchanged (no silent write-back/migration is ever attempted).
- **`scan_runs()` refactor done in this PR, not deferred (open question 2):** `scan_runs()`
  calls `experiments.graph.assemble_experiment()` per run instead of its own manifest walk
  (verified by reading `tracking/index.py`, not just behaviorally), and a regression test on
  the pre-refactor fixture confirms every `RunIndexEntry` field (`run_id`, `competition`,
  `status`, `parent_run_id`, `iteration`, `improvement_strategy`, `metrics`, `params`) is
  unchanged for existing callers.

# Plan 2 — Structured Hypothesis

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** Plan 1 (`Experiment.hypothesis_id` field). **Unlocks:**
Plans 4, 6.

---

## Goal

Give experiments a "why" *before* they run. Instead of "Experiment 41", store:

```
Hypothesis H-021
  Observation:  Rare classes perform poorly.
  Reason:       Dataset imbalance.
  Prediction:   Focal Loss will improve Macro F1.
  Confidence:   0.74
  Status:       testing
```

This is Task 2 in the brief. The system starts *thinking* before it trains, not just recording
after.

## Why this matters

`improvement/planner.py` already decides *what to change* for one specific child run (tune
params, apply a recipe, or an LLM-authored plan). What it doesn't do is give that decision a
name that outlives the run — if you tune hyperparameters twice for two different underlying
reasons, there's currently no way to tell those two attempts apart except by reading
`improvement_plan.json.rationale` free text on each. A `Hypothesis` is the durable, reusable
unit; an `ImprovementPlan`/training run is one *test* of it (and a hypothesis can be tested more
than once, e.g. re-tested with a different seed or a refined version).

## Design

### 1. Storage: per-competition, not per-run

Hypotheses outlive any single run (a hypothesis can be proposed before any run tests it, and
can be tested by more than one run). They live under the new top-level `knowledge/` directory
introduced in the [Milestone 2 overview](README.md#2-repository-shape-after-milestone-2),
sibling to `runs/`:

```
knowledge/<competition-slug>/hypotheses/H-001.json
knowledge/<competition-slug>/hypotheses/H-002.json
...
```

One file per hypothesis (human-diffable, matches the existing "one artifact per concern"
convention used for every other JSON file in this codebase) rather than a single growing
JSONL/array file that every write has to rewrite.

### 2. Data model

```python
class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"

class Hypothesis(BaseModel):
    id: str                        # "H-001", auto-incremented per competition
    competition: str
    observation: str
    reason: str
    prediction: str
    confidence: float              # 0.0-1.0, author's prior belief
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    tags: list[str] = []           # e.g. ["augmentation", "class-imbalance"]
    source: Literal["manual", "reflection", "llm"] = "manual"
    evidence_for: list[str] = []       # run_ids where the prediction held
    evidence_against: list[str] = []   # run_ids where it didn't
    created_at: datetime
    updated_at: datetime
```

**Linked experiments are derived, not stored on the hypothesis.** `Hypothesis` does not keep a
`linked_experiment_ids` list that has to be kept in sync from two places. Instead:

- The *experiment → hypothesis* link is the source of truth, written once at run creation
  (`manifest.metadata["hypothesis_id"]`), exactly like `parent_run_id` is written once at fork
  time today.
- `evidence_for` / `evidence_against` on the `Hypothesis` *are* stored (they're a verdict, not
  a mechanical index) and are appended by whoever concludes the test — in v1 that's a human via
  `research hypothesis update`, and once Plan 4 ships, automatically by the reflection engine.
- `experiments/hypothesis.py` exposes `linked_experiments(hypothesis_id, graph) -> list[Experiment]`
  as a computed query (filter the `ExperimentGraph` from Plan 1 by `hypothesis_id`), so the
  "derived" data is always fresh and never drifts from the manifests.

### 3. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `Hypothesis`, `HypothesisStatus` |
| `src/labpilot/experiments/hypothesis.py` | new — `HypothesisStore` (CRUD, id allocation, status transitions), `linked_experiments()` |
| `src/labpilot/improvement/planner.py` / `improvement/fork.py` | accept optional `hypothesis_id`, write it into child manifest metadata |
| `src/labpilot/orchestrator/pipeline.py` | `run()`/`build()` accept optional `hypothesis_id` for root runs too (a hypothesis doesn't have to start from an existing parent) |
| `src/labpilot/cli/main.py` | + `hypothesis_app` (`add`, `list`, `show`, `update`) |

`HypothesisStore`:

```python
class HypothesisStore:
    def __init__(self, knowledge_dir: Path, competition: str) -> None: ...
    def create(self, *, observation, reason, prediction, confidence, tags=(), source="manual") -> Hypothesis: ...
    def get(self, hypothesis_id: str) -> Hypothesis | None: ...
    def list(self, *, status: HypothesisStatus | None = None) -> list[Hypothesis]: ...
    def update_status(self, hypothesis_id: str, status: HypothesisStatus, *, evidence_run_id: str | None = None) -> Hypothesis: ...
```

ID allocation: scan existing `H-*.json` files in the competition's `hypotheses/` dir, take
`max + 1`, zero-padded to 3 digits (`H-001`, ... `H-999`, then widen if it ever matters).

### 4. CLI

```
research hypothesis add --competition <slug> \
    --observation "Rare classes perform poorly" \
    --reason "Dataset imbalance" \
    --prediction "Focal Loss will improve Macro F1" \
    --confidence 0.74 \
    --tags loss,class-imbalance

research hypothesis list --competition <slug> [--status testing]
research hypothesis show H-021
research hypothesis update H-021 --status confirmed --evidence-run 20260712-...-exp41
```

And the two existing run-creation commands gain one optional flag each:

```
research run --competition <slug> --hypothesis H-021       # tests a hypothesis from a fresh root run
research improve --run-id <parent> --hypothesis H-021      # tests a hypothesis via a child run
```

## Non-goals

- No automatic hypothesis generation in this plan — that's Plan 4 (reflection engine can
  *propose* new hypotheses with `source="reflection"`/`"llm"`, but the store itself is
  source-agnostic and ships first with only manual authoring).
- No enforcement that a hypothesis can only be `testing` via exactly one run — a hypothesis
  can be tested multiple times; `evidence_for`/`evidence_against` simply accumulates run ids.
- No hypothesis "graph" (hypotheses referencing other hypotheses) — flat list per competition
  in v1.

## Open questions

1. Should `--hypothesis` on a *root* `research run` (no parent) be allowed, or should
   hypotheses only attach to `research improve` children? → Allow both; a first baseline run
   can validly test "a LightGBM baseline will get above X" as a hypothesis.
2. What happens if a user deletes/never resolves a `testing` hypothesis — do we need a
   "stale" status or a TTL? → Not for v1; `research hypothesis list --status testing` surfaces
   these for human cleanup, no automatic expiry.

## Acceptance criteria

- `research hypothesis add` creates `knowledge/<slug>/hypotheses/H-NNN.json` with a
  correctly incremented id.
- `research improve --run-id <parent> --hypothesis H-001` writes `hypothesis_id: "H-001"` into
  the child's `manifest.json` metadata.
- `experiments/hypothesis.py:linked_experiments("H-001", graph)` returns exactly the runs whose
  manifest metadata references it, using the Plan-1 `ExperimentGraph`.
- `research hypothesis update H-001 --status confirmed --evidence-run <id>` persists the new
  status and appends `<id>` to `evidence_for`.

# Plan 3 — Automatic Comparator

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** Plan 1 (`Experiment` model). **Unlocks:** Plans 4, 5, 8
(Plan 7 optionally filters on its output).

---

## Goal

Given experiment A and experiment B, deterministically produce:

```
Changes
  + Mixup
  + EMA

Validation        +0.009
Training Time      +40%
Inference          No Change

Conclusion         Worth Keeping
```

**No LLM required** — this is Task 3 in the brief, and the highest-ROI, lowest-risk plan in
the milestone because it's pure engineering on data the system already has.

## Current state

`tracking/index.py:diff_runs(runs_dir, base_run_id, compare_run_id) -> RunDiff` already computes:

- `metric_deltas` (compare − base, for every shared numeric metric key)
- `param_changes` (base vs compare, for every key in `experiment/record.json.params`)
- `lineage` (iteration, parent, strategy)
- `submission_notes` (free text status string)

This is real, working logic — reused, not thrown away. What's missing, from the brief's Task 3:

1. **Categorization.** `param_changes` is a flat dict of key → {base, compare}; there's no
   grouping into "this was an augmentation change" vs "this was a model change."
2. **A verdict.** Today a human reads the deltas and decides. Task 3 wants a
   `Worth Keeping` / `Not Worth Keeping` / `Inconclusive` conclusion computed from thresholds.
3. **Cost, not just accuracy.** `runtime_seconds` (derivable per Plan 1) isn't in `RunDiff` at
   all today. `Training Time +40%` in the brief's example needs it.
4. **Persistence.** `diff_runs` is compute-on-demand only (`research runs diff`). Task 3
   implies every child experiment gets compared to its parent automatically, so the
   comparison is a fact recorded once, not recomputed by whoever happens to ask later.

## Design

### 1. `ExperimentComparison`

```python
class ChangeCategory(StrEnum):
    MODEL = "model"
    AUGMENTATION = "augmentation"
    TRAINING_STRATEGY = "training_strategy"
    SCHEDULER = "scheduler"
    FEATURE_ENGINEERING = "feature_engineering"
    OTHER = "other"

class ConfigChange(BaseModel):
    category: ChangeCategory
    field: str              # e.g. "model_params.learning_rate", "feature_recipes"
    base_value: Any
    compare_value: Any
    label: str               # human-readable, e.g. "+ Mixup", "learning_rate: 0.05 → 0.03"

class Verdict(StrEnum):
    WORTH_KEEPING = "worth_keeping"
    NOT_WORTH_KEEPING = "not_worth_keeping"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"

class ExperimentComparison(BaseModel):
    base_id: str
    compare_id: str
    primary_metric_key: str | None       # from competition.json's MetricSpec
    metric_deltas: dict[str, float]
    changes: list[ConfigChange]
    runtime_delta_seconds: float | None
    runtime_delta_pct: float | None
    verdict: Verdict
    verdict_reason: str                  # one-line deterministic explanation
```

### 2. Categorization — a static lookup, not magic

`field → category` is a small, explicit dict maintained in `comparator.py`, keyed on the
prefix/name of the field as it appears in `model_params` / `feature_recipes` /
`baseline_choice.json`:

```python
_CATEGORY_RULES: dict[str, ChangeCategory] = {
    "model_params.learning_rate": ChangeCategory.TRAINING_STRATEGY,
    "model_params.num_leaves": ChangeCategory.MODEL,
    "feature_recipes": ChangeCategory.FEATURE_ENGINEERING,
    "template_name": ChangeCategory.MODEL,
    # ... extended as new template/param names are introduced
}
```

Anything not matched falls into `OTHER` rather than raising — this list is expected to grow
over time as new templates/params are added; it fails safe.

### 3. Verdict logic — thresholds, explicit and configurable

```python
def _verdict(metric_delta: float | None, runtime_delta_pct: float | None, *, noise_epsilon: float, max_acceptable_runtime_increase_pct: float) -> tuple[Verdict, str]:
    if metric_delta is None:
        return Verdict.INCONCLUSIVE, "No shared primary metric to compare."
    if abs(metric_delta) <= noise_epsilon:
        return Verdict.INCONCLUSIVE, f"Metric delta ({metric_delta:+.4f}) within noise band (±{noise_epsilon})."
    if metric_delta < 0:
        return Verdict.REGRESSION, f"Primary metric regressed by {metric_delta:+.4f}."
    if runtime_delta_pct is not None and runtime_delta_pct > max_acceptable_runtime_increase_pct:
        return Verdict.NOT_WORTH_KEEPING, f"Gain ({metric_delta:+.4f}) not worth +{runtime_delta_pct:.0f}% runtime."
    return Verdict.WORTH_KEEPING, f"Primary metric improved by {metric_delta:+.4f}."
```

`noise_epsilon` and `max_acceptable_runtime_increase_pct` are config, not hardcoded —
`configs/default.yaml: experiments.comparator.noise_epsilon` (default e.g. `0.001`) and
`.max_runtime_increase_pct` (default e.g. `50`). Whether "worth keeping" should account for
*direction* of the metric (maximize vs minimize, from `MetricSpec`) matters here — the
delta sign convention must be normalized so "improved" always means positive before this
function runs (reuse the same normalization Plan 1's `best_path()` needs).

`Inference: No Change` from the brief's example is deliberately **not modeled as a real field**
in v1: LabPilot doesn't measure inference latency anywhere today (no template records it).
Document this explicitly in the CLI output as `inference: not tracked` rather than fabricating
a number — see Non-goals.

### 4. Persistence — auto-write on child completion

`Pipeline.improve()` (in `orchestrator/pipeline.py`), after the `write_reflection` stage,
calls `comparator.compare(parent_experiment, child_experiment)` and writes
`runs/<child_id>/comparison.json`. This makes the comparison a durable fact (used by Plan 5's
knowledge base and Plan 8's dashboard) instead of something recomputed on every dashboard
render. Root runs (no parent) simply get no `comparison.json` — nothing to compare against.

### 5. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `ChangeCategory`, `ConfigChange`, `Verdict`, `ExperimentComparison` |
| `src/labpilot/experiments/comparator.py` | new — `compare(base, compare) -> ExperimentComparison`, category rules, verdict logic |
| `src/labpilot/tracking/index.py` | `diff_runs()` becomes a thin wrapper: build two `Experiment`s, call `comparator.compare`, adapt to the existing `RunDiff` shape so `research runs diff` output is unchanged |
| `src/labpilot/orchestrator/pipeline.py` | `improve()` writes `comparison.json` after `write_reflection` |
| `src/labpilot/cli/main.py` | + `experiments compare` subcommand |
| `configs/default.yaml` | + `experiments.comparator.{noise_epsilon,max_runtime_increase_pct}` |

### 6. CLI

```
research experiments compare <base_id> <compare_id> [--format table|json]
```

Renders the brief's exact mockup shape: a `Changes` list, `Validation`/metric deltas, training
time delta, an explicit `inference: not tracked` line, and `Conclusion`.

## Non-goals

- **No inference-latency tracking.** Would require instrumenting every template's train.py to
  time a batch of predictions — real work, not needed to prove the comparator. Flagged as a
  template-level follow-up, not blocking this plan.
- **No multi-way comparison** (A vs B vs C at once) — pairwise only in v1; Plan 8's dashboard
  handles "best pipeline across N experiments" via `ExperimentGraph.best_path()` (Plan 1), not
  via this comparator.
- Verdict thresholds are heuristic, not statistically rigorous (no variance estimate from
  repeated seeds) — acceptable for v1 given LabPilot doesn't run repeated-seed trials today.

## Open questions

1. Should `research runs diff`'s output format change at all, or must it be byte-for-byte
   identical after the refactor? → Byte-for-byte identical; this is a pure internals swap,
   the new richer view is only exposed via the new `experiments compare` command.
2. Config diffing today only looks at `experiment/record.json.params` — should it also diff
   `baseline_choice.json` (template name, problem type) so template swaps show up as changes?
   → Yes, add `template_name` as a compared field; it's exactly the kind of change the brief's
   example (`+ Mixup`, `+ EMA`) is gesturing at when the change is more structural than a
   single hyperparameter.

## Acceptance criteria

- `comparator.compare()` on a fixture pair with a known metric improvement and a known
  `feature_recipes` addition produces `changes` containing the recipe as an `augmentation` or
  `feature_engineering` category entry, and `verdict == WORTH_KEEPING`.
- A fixture pair with a metric delta inside the noise band returns `INCONCLUSIVE`.
- A fixture pair with a metric improvement but runtime increase above the configured
  threshold returns `NOT_WORTH_KEEPING`.
- After `research improve --strategy tune`, the child run directory contains a valid
  `comparison.json`.
- `research runs diff --base <a> --compare <b>` output is unchanged from before this plan
  (regression-tested against the existing `test_tracking_index.py` fixtures).

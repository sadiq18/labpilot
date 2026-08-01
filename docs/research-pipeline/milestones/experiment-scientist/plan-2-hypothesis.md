# Plan 2 — Structured Hypothesis

Back to [Milestone 2](README.md).

**Status:** Shipped. **Depends on:** Plan 1 (`Experiment.hypothesis_id` field). **Unlocks:**
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

## Current state (at ship)

- Manual authoring via `research hypothesis add|list|show|update` (all require `--competition`).
- Optional `--hypothesis H-NNN` on `research run` and `research improve`.
- Attaching auto-flips `proposed` → `testing` (other statuses left alone).
- Evidence routing on update: `confirmed` → `evidence_for`; `rejected` → `evidence_against`;
  other statuses change status only.
- `Experiment.description` prefers `hypothesis.prediction` when the link resolves.

Automatic hypothesis drafting remains Plan 4 (`source="reflection"` / `"llm"`).

## Design

### 1. Storage: per-competition, not per-run

Hypotheses live under `knowledge/<competition-slug>/hypotheses/H-NNN.json` (configured via
`knowledge_dir`, default `knowledge/`, gitignored like `runs/`).

### 2. Data model

`Hypothesis` + `HypothesisStatus` in `experiments/models.py`. Linked experiments are derived
via `linked_experiments(hypothesis_id, graph)` — not stored on the hypothesis file.

### 3. Files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `Hypothesis`, `HypothesisStatus` |
| `src/labpilot/experiments/hypothesis.py` | `HypothesisStore`, `linked_experiments()` |
| `src/labpilot/improvement/fork.py` | optional `hypothesis_id` in child metadata |
| `src/labpilot/orchestrator/pipeline.py` | `run`/`improve` accept `hypothesis_id`; validate + mark testing |
| `src/labpilot/experiments/graph.py` | description prefers prediction; `knowledge_dir` on assemble/build |
| `src/labpilot/cli/main.py` | `hypothesis` app; `--hypothesis` on run/improve |
| `src/labpilot/config.py` | `knowledge_dir` / `LABPILOT_KNOWLEDGE_DIR` |

## Open questions (resolved)

1. `--hypothesis` on root `research run`? → **Yes**, both root and improve.
2. Stale/`testing` TTL? → **Not for v1**; list by status for human cleanup.
3. Evidence lists on update? → **Infer from status**: confirmed → `evidence_for`, rejected →
   `evidence_against`, else status-only.
4. `hypothesis show` without competition? → **Require `--competition`** (multi-competition safe).

## Acceptance criteria

- `research hypothesis add` creates `knowledge/<slug>/hypotheses/H-NNN.json` with a
  correctly incremented id.
- `research improve --run-id <parent> --hypothesis H-001` writes `hypothesis_id: "H-001"` into
  the child's `manifest.json` metadata.
- Attaching `--hypothesis` to a `proposed` hypothesis flips it to `testing`.
- `experiments/hypothesis.py:linked_experiments("H-001", graph)` returns exactly the runs whose
  manifest metadata references it, using the Plan-1 `ExperimentGraph`.
- `research hypothesis update H-001 --status confirmed --evidence-run <id>` persists the new
  status and appends `<id>` to `evidence_for` (`rejected` → `evidence_against`).
- All hypothesis CLI commands require `--competition`.
- `Experiment.description` prefers the linked hypothesis prediction when that file exists.

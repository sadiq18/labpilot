# M8 — Close the objective feedback loop

**Status:** not started · **Blocked by:** M7 (scores must be able to differ)

---

## Purpose

The Conductor cannot tell whether it is making progress. `evaluate_stops` reads
a metric to decide *when to quit*, but nothing feeds score history into the
decision about *what to try next*.

Consequence, observed directly: the policy optimises **process completion**. It
stopped at step 4 of 12 with the target 39× away, reasoning that it had used
each tool once. Goal persistence (already shipped) overrides that stop, but
overriding a bad decision is not the same as making a good one — the policy
still has no idea the score has not moved.

## Goal

Hypothesis N+1 is demonstrably caused by the result of experiment N.

## Approach

Three parts, in order:

**1. One comparable score series.** Every experiment appends
`(experiment_id, hypothesis_id, technique, metric_name, value, timestamp)` to a
single series. Today `metrics.json` is overwritten in place and the metric key
varies (`cv_rmse` vs `cv_mse` depending on selector defaults), so history is
neither retained nor comparable.

**2. Score history in the observe bundle.** The policy should see
`best_so_far`, `last_3_scores`, `delta_vs_best`, and `steps_since_improvement`.
It already receives `untested_hypotheses` and `hours_since_last_artifact`; this
is the same mechanism applied to the thing that actually matters.

**3. Reflection → hypothesis.** The missing edge. Today candidates are mined
only from kernels, papers and repositories — *other people's work*. Nothing
turns "we tried `target_encoding` and the score got worse" into "try
`feature_interactions` instead".

This is the `learn()` step in the original infinite-loop design and it is the
only path where the system gets **smarter** rather than merely busier.

## Exit criteria

1. A campaign log in which a hypothesis's `reason` field cites a *prior
   experiment's result* by id.
2. `steps_since_improvement` visible in observe, and a policy decision that
   demonstrably changes when it is high.
3. Re-running a campaign on a workspace with history picks a different first
   move than on an empty one.

## Traps

- **Do not conflate "a metric exists" with "the loop uses it".** The metric has
  been in `evaluate_stops` since M3 and changed no decision in nine campaigns.
- **Beware metric-key drift.** The selector defaults `tabular_regression` to
  `rmse` while the competition scores `mse`; both appeared in `metrics.json` as
  `cv_rmse` and `mse` in the same session. Normalise before comparing or the
  series will silently mix scales.
- **Reflection already runs and produces nothing actionable.** The pipeline
  writes assessments, beliefs and lessons today. The gap is not "add
  reflection", it is "make reflection emit a hypothesis with a technique the
  recipe layer can execute".

## Related code

- `src/labpilot/research_engine/conductor/budgets.py` — `evaluate_stops`, the only current metric consumer
- `src/labpilot/research_engine/conductor/policy.py` — `build_observe_bundle`
- `src/labpilot/research_engine/reflection/pipeline.py` — `run_reflection`
- `src/labpilot/research_engine/intelligence/hypothesis/candidates.py` — `generate_candidates`, currently evidence-only

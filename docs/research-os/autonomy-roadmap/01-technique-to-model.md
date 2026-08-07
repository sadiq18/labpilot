# M7 — A technique must change the model

**Status:** **done 2026-08-07** — MSE 194.80 → 190.97, the first distinct
scores · **Blocks:** everything · **Blocked by:** ~~[M10](04-llm-tiering.md)~~
(shipped)

> A differ-table taken 2026-08-07 confirmed the technique path is sound: every
> `applied` technique changes the emitted code, and every `not_applicable` one
> reports why. The campaign blocker turned out to be undeclared dependencies —
> generated code importing `catboost`, which killed eight consecutive runs — not
> techniques failing to reach the model. See
> [M19](14-experiments-as-deltas.md), which retires the template path M7
> exposed as the narrow one.

**Design:** [design/01-technique-to-model.md](design/01-technique-to-model.md)

---

## Purpose

Hypotheses cannot currently be *tested*. `technique` is carried on the plan and
ignored by codegen, so every experiment reproduces the baseline. Observed on
rogii: twelve hypotheses, one reachable model state, **MSE 194.80 every time**.

Until this lands, the Conductor is a very sophisticated way of running the same
experiment repeatedly, and no other milestone can pay off.

## Goal

Two different hypotheses produce two different CV scores.

## Approach

**Recipe layer first, LLM codegen second.**

A technique becomes a declarative transform over the rendered template, not
free-form generated code:

```yaml
target_encoding:
  applies_to: [tabular_regression, tabular_classification]
  adds_features: [target_encoded]
  requires: [categorical_columns]
feature_interactions:
  applies_to: [tabular_*]
  adds_features: [pairwise_products]
model_swap_catboost:
  applies_to: [tabular_*]
  model: catboost
  params: {depth: 8}
ensemble_blend:
  applies_to: [tabular_*]
  combines: [previous_best, current]
```

Most useful techniques (`target_encoding`, `feature_interactions`, `SWA`,
`ensemble`, hyperparameter moves, model swaps) are expressible this way.
Reserve LLM codegen for what recipes cannot express.

**Why recipes first — sequencing, not preference:**

1. **Unblocked.** Recipes need no LLM, so M7 stops blocking M8 and M13 without
   waiting on the routing work.
2. **Deterministic and testable.** A recipe either changed the feature set or it
   did not; assertable in a unit test and reproducible byte-for-byte.
3. **Attributable.** Recording `technique_origin` (`registry` | `llm`) keeps a
   poor implementation distinguishable from a genuine negative result.

The LLM path already receives the technique in full and is not broken; the
defect is entirely in the template fallback. Both paths are kept.

**Wiring:** `generate_plan(hypothesis_id=X)` must carry `technique` through to
the renderer, and the renderer must apply the matching recipe. The seam already
exists — `BaselineChoice` reaches `CodeRenderer.render()`, and hypothesis
metadata already flows into `code_engineering/capability.py` as `technique` /
`technique_stack` / `combo_techniques`. It is read into the prompt and then
dropped on the template path.

## Exit criteria

Not "recipes exist". Run a campaign and assert:

1. Three hypotheses with different techniques produce **three different**
   `cv_*` values in `metrics.json`.
2. Each experiment record names the technique that produced it.
3. A recipe that cannot apply to the problem type is rejected at plan time, not
   discovered at train time.

Criterion 1 is the whole milestone. It cannot be satisfied by accident.

## Traps

- **Start with recipes, but do not reject LLM codegen.** On
  `qwen2.5-coder:14b` it produced no usable training code across an entire day,
  and the emergency stub that replaced it wrote `cv_accuracy: 0.0` on a
  regression task while reporting success. That is evidence about *a 14B local
  model*, not about the approach: once [M10](04-llm-tiering.md) routes `codegen`
  to a frontier model, the LLM path covers the long tail no registry will
  enumerate. Recipes go first because they are unblocked, not because the LLM
  path is worse — see [the design](design/01-technique-to-model.md) §8.2.
- **A weak model implementing a technique is worse than not running it.** The
  result is recorded as "technique X did not help": a false negative
  indistinguishable from a real one, which then poisons hypothesis generation
  permanently. This is why M10 routes codegen with *wait*, never *degrade*.
- **Do not let recipes bypass the validation plan.** Anything that adds features
  must respect `validation.exclude_features`, or a technique will quietly
  reintroduce the leakage columns the profiler worked out (`ANCC`, `ASTNU`,
  `TVT` …). A derived feature inherits its parents' availability.

## Related code

- `src/labpilot/research_engine/execution/baseline/selector.py` — `BaselineChoice`, `ValidationPlan`
- `src/labpilot/research_engine/execution/capabilities/code_engineering/capability.py` — technique already in context, dropped on the template path
- `.../code_engineering/offline_codegen/renderer.py` — where a recipe would apply
- `.../templates/tabular_regression_partitioned/` — the template to transform
